import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from utils.data_manager import DataManager
from helper import Model

# =====================================================================================
# Label mapping (orig CIFAR id -> model id) from DataManager._class_order
# =====================================================================================
def build_label_mapper_from_dm(data_manager) -> np.ndarray:
    class_order = np.array(data_manager._class_order)  # e.g., [12, 5, 87, ...]
    mapper = np.empty(100, dtype=np.int64)
    for new_id, orig_id in enumerate(class_order):
        mapper[orig_id] = new_id
    return mapper

# =====================================================================================
# Common eval utility
# =====================================================================================
@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(1, total)

# =====================================================================================
# Clean CIFAR-100 @224 (NO normalization; labels remapped to model indexing)
# =====================================================================================
def eval_cifar100_clean(model, batch_size=256, num_workers=4, device="cuda",
                        data_root="/home/lis/data", label_mapper: np.ndarray = None):
    base = datasets.CIFAR100(root=data_root, train=False, download=True)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # NO Normalize (matches your training)
    ])

    class Wrap(Dataset):
        def __init__(self, base, tf, mapper):
            self.data, self.targets = base.data, base.targets
            self.tf, self.mapper = tf, mapper
        def __len__(self): return len(self.targets)
        def __getitem__(self, i):
            x = self.tf(self.data[i])
            y_orig = int(self.targets[i])
            y = int(self.mapper[y_orig]) if self.mapper is not None else y_orig
            return x, y

    test_ds = Wrap(base, tf, label_mapper)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    acc = eval_loader(model, test_loader, torch.device(device))
    print(f"\n=== Clean CIFAR-100 @224 (no norm, remapped) === {acc:.2f}%")
    return acc

# =====================================================================================
# CIFAR-100-C @224 (NO normalization; labels remapped)
# =====================================================================================
class CIFAR100C224(Dataset):
    def __init__(self, images_npy, labels_npy, severity: int, label_mapper: np.ndarray):
        assert 1 <= severity <= 5
        self.imgs = np.load(images_npy, mmap_mode="r")         # (50000,32,32,3)
        self.lbls = np.load(labels_npy, mmap_mode="r")         # (50000,)
        assert self.imgs.shape == (50000, 32, 32, 3)
        assert self.lbls.shape == (50000,)
        self.lo, self.hi = (severity - 1) * 10000, severity * 10000
        self.label_mapper = label_mapper

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # NO Normalize
        ])

    def __len__(self): return self.hi - self.lo
    def __getitem__(self, idx):
        i = self.lo + idx
        x = self.tf(self.imgs[i])
        y = int(self.label_mapper[int(self.lbls[i])])
        return x, y

def eval_cifar100c(model, cifar_c_dir, batch_size=256, num_workers=4,
                   severities=(1,2,3,4,5), device="cuda", label_mapper: np.ndarray = None):
    labels_path = os.path.join(cifar_c_dir, "labels.npy")
    assert os.path.isfile(labels_path), f"labels.npy not found in {cifar_c_dir}"
    img_npys = sorted(p for p in glob.glob(os.path.join(cifar_c_dir, "*.npy"))
                      if os.path.basename(p) != "labels.npy")

    # keep only expected arrays
    corruptions = []
    for p in img_npys:
        arr = np.load(p, mmap_mode="r")
        if arr.ndim == 4 and arr.shape == (50000, 32, 32, 3):
            corruptions.append(p)

    device = torch.device(device)
    model.to(device).eval()

    per_corr_means, per_sev_means = {}, {s: [] for s in severities}
    for p in corruptions:
        corr_name = os.path.splitext(os.path.basename(p))[0]
        sev_accs = []
        for s in severities:
            dl = DataLoader(CIFAR100C224(p, labels_path, s, label_mapper),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
            acc = eval_loader(model, dl, device)
            sev_accs.append(acc); per_sev_means[s].append(acc)
            print(f"[C-C {corr_name:18s} | sev {s}] acc = {acc:.2f}%")
        per_corr_means[corr_name] = sum(sev_accs)/len(sev_accs)

    print("\n=== CIFAR-100-C: Per-corruption mean over severities ===")
    for k,v in per_corr_means.items(): print(f"{k:24s}: {v:.2f}%")

    print("\n=== CIFAR-100-C: Per-severity mean over corruptions ===")
    for s, vals in per_sev_means.items():
        m = sum(vals)/len(vals) if vals else float('nan')
        print(f"severity {s}: {m:.2f}%")

    global_mean = sum(per_corr_means.values())/len(per_corr_means) if per_corr_means else float('nan')
    print(f"\n=== CIFAR-100-C global mean accuracy === {global_mean:.2f}%")
    return global_mean

# =====================================================================================
# CIFAR-100-P @224 (NO normalization; labels remapped) + mFR
# - Official perturbations: files at the top-level of CIFAR-100-P directory.
# - Extra perturbations: files inside CIFAR-100-P/extra/.
# Computes:
#   • mean Flip Rate (mFR): average normalized flip count across sequences
#   • mean Accuracy over all frames (optional diagnostic)
# Auto-detects data layout: (N, L, H, W, C) or (N, H, W, C, L)
# =====================================================================================
class CIFAR100PSequences:
    """Memory-mapped iterator over sequences for a single perturbation file."""
    def __init__(self, p_path: str, labels_path: str, label_mapper: np.ndarray):
        self.arr = np.load(p_path, mmap_mode="r")     # shape (N, L, 32,32,3) or (N,32,32,3,L)
        self.lbls = np.load(labels_path, mmap_mode="r")  # shape (N,)
        self.mapper = label_mapper
        assert self.lbls.shape[0] == self.arr.shape[0], "labels length mismatch"

        # Detect sequence axis
        # Candidates: (N,L,H,W,C) or (N,H,W,C,L)
        if self.arr.ndim != 5:
            raise ValueError(f"Unexpected ndim for CIFAR-P array: {self.arr.shape}")

        # Decide which axis is sequence length
        # Prefer axis=1 if reasonably sized; else try axis=4
        if 4 <= self.arr.shape[1] <= 100 and self.arr.shape[2:] == (32, 32, 3):
            self.seq_axis = 1
            self.N, self.L = self.arr.shape[0], self.arr.shape[1]
        elif 4 <= self.arr.shape[4] <= 100 and self.arr.shape[1:4] == (32, 32, 3):
            self.seq_axis = 4
            self.N, self.L = self.arr.shape[0], self.arr.shape[4]
        else:
            raise ValueError(f"Cannot infer sequence axis from shape {self.arr.shape}")

        # Basic transform (no norm)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def iter_sequences(self, batch_size: int = 256, device: torch.device = torch.device("cuda")):
        """Yield (preds over L frames, target) per item in batches for efficiency."""
        model = None  # filled externally in eval function; kept here for interface clarity
        raise NotImplementedError("Use eval_cifar100p(...) which handles batching with the model.")

def _to_chw_tensor(x_hwc_uint8, tf):
    # x_hwc_uint8: (32,32,3) uint8
    return tf(x_hwc_uint8)

@torch.no_grad()
def eval_cifar100p(model, cifar_p_dir, batch_size=256, device="cuda",
                   label_mapper: np.ndarray = None, include_extra: bool = False,
                   data_root: str = "/home/lis/data"):
    """
    Evaluate CIFAR-100-P perturbation robustness.
    Computes:
      - mFR (mean Flip Rate): average over sequences of (#prediction changes) / (L-1)
      - mean accuracy across all frames (diagnostic)
    Uses CIFAR-100 test labels from torchvision (10k samples).
    """

    device = torch.device(device)
    model = model.to(device).eval()

    # ---- Load CIFAR-100 test labels (10k) ----
    from torchvision import datasets
    test_set = datasets.CIFAR100(root=data_root, train=False, download=True)
    labels = np.array(test_set.targets, dtype=np.int64)  # shape (10000,)
    assert labels.shape[0] == 10000, "Expected 10k CIFAR-100 test labels."

    # ---- Find perturbation files ----
    def list_perturbation_files(root, include_extra):
        main = sorted(p for p in glob.glob(os.path.join(root, "*.npy")))
        extras = []
        if include_extra:
            extra_dir = os.path.join(root, "extra")
            if os.path.isdir(extra_dir):
                extras = sorted(glob.glob(os.path.join(extra_dir, "*.npy")))
        return main, extras

    main_files, extra_files = list_perturbation_files(cifar_p_dir, include_extra)

    # ---- Evaluation helper ----
    def evaluate_file(p_path, tag):
        arr = np.load(p_path, mmap_mode="r")  # (N,L,32,32,3) or (N,32,32,3,L)
        assert arr.shape[0] == 10000, f"{tag}: expected N=10000, got {arr.shape}"
        N = arr.shape[0]

        # Infer sequence axis
        if arr.ndim != 5:
            raise ValueError(f"Unexpected ndim in {tag}: {arr.shape}")
        if 4 <= arr.shape[1] <= 100 and arr.shape[2:] == (32,32,3):
            # (N,L,H,W,C)
            L = arr.shape[1]
            get_frame = lambda n, t: arr[n, t]
        elif 4 <= arr.shape[4] <= 100 and arr.shape[1:4] == (32,32,3):
            # (N,H,W,C,L)
            L = arr.shape[4]
            get_frame = lambda n, t: arr[n, :, :, :, t]
        else:
            raise ValueError(f"Cannot infer sequence axis from shape {arr.shape} in {tag}")

        # Transform (no norm)
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        mfr_sum = 0.0
        acc_sum = 0.0
        idxs = np.arange(N)

        for start in range(0, N, batch_size):
            chunk = idxs[start:start+batch_size]
            frames, targets = [], []
            for n in chunk:
                y_orig = int(labels[n])
                y = int(label_mapper[y_orig]) if label_mapper is not None else y_orig
                targets.append(y)
                seq = [tf(get_frame(n, t)) for t in range(L)]  # list of (3,224,224)
                frames.append(torch.stack(seq, dim=0))
            frames = torch.stack(frames, dim=0)       # (B,L,3,224,224)
            B = frames.shape[0]
            frames = frames.view(B*L, *frames.shape[2:]).to(device, non_blocking=True)

            logits = model(frames)                    # (B*L, C)
            preds = logits.argmax(1).view(B, L).cpu().numpy()
            targets = np.array(targets, dtype=np.int64)

            flips = (preds[:, 1:] != preds[:, :-1]).sum(axis=1).astype(np.float32) / max(1, L-1)
            mfr_sum += flips.sum()
            acc_sum += (preds == targets[:, None]).mean(axis=1).sum()

        mfr = mfr_sum / N
        mean_acc = acc_sum / N
        print(f"[C-P {tag:18s}] mFR = {mfr:.4f} | mean frame-acc = {mean_acc*100:.2f}%")
        return mfr, mean_acc

    # ---- Run official perturbations ----
    print("\n=== CIFAR-100-P (official perturbations) ===")
    mfrs_main, accs_main = [], []
    for p in main_files:
        name = os.path.splitext(os.path.basename(p))[0]
        mfr, macc = evaluate_file(p, name)
        mfrs_main.append(mfr); accs_main.append(macc)
    if mfrs_main:
        print(f"\n[C-P] mFR (OFFICIAL, mean over {len(mfrs_main)}): {np.mean(mfrs_main):.4f}")
        print(f"[C-P] Mean frame-acc (OFFICIAL): {np.mean(accs_main)*100:.2f}%")

    # ---- Run extra perturbations ----
    if include_extra and extra_files:
        print("\n=== CIFAR-100-P (EXTRA perturbations) ===")
        mfrs_ex, accs_ex = [], []
        for p in extra_files:
            name = f"extra/{os.path.splitext(os.path.basename(p))[0]}"
            mfr, macc = evaluate_file(p, name)
            mfrs_ex.append(mfr); accs_ex.append(macc)
        if mfrs_ex:
            print(f"\n[C-P] mFR (EXTRA, mean over {len(mfrs_ex)}): {np.mean(mfrs_ex):.4f}")
            print(f"[C-P] Mean frame-acc (EXTRA): {np.mean(accs_ex)*100:.2f}%")

# =====================================================================================
# AutoAttack evaluation (L_inf and L2)
# =====================================================================================
def eval_autoattack(model, batch_size=128, device="cuda",
                    data_root="/home/lis/data", label_mapper: np.ndarray = None,
                    norm="Linf", eps=None, version="standard"):
    """
    Evaluate adversarial robustness with AutoAttack.
    Args:
        model: your trained model
        batch_size: eval batch size
        device: "cuda" or "cpu"
        data_root: CIFAR-100 root dir
        label_mapper: mapping from orig CIFAR labels -> model labels
        norm: "Linf" or "L2"
        eps: perturbation budget
        version: AutoAttack version, usually "standard"
    """

    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError("Please install autoattack first: pip install git+https://github.com/fra31/auto-attack")

    # --- Dataset (clean CIFAR-100 test, remapped, same transforms as training) ---
    base = datasets.CIFAR100(root=data_root, train=False, download=True)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    class Wrap(Dataset):
        def __init__(self, base, tf, mapper):
            self.data, self.targets = base.data, base.targets
            self.tf, self.mapper = tf, mapper
        def __len__(self): return len(self.targets)
        def __getitem__(self, i):
            x = self.tf(self.data[i])
            y_orig = int(self.targets[i])
            y = int(self.mapper[y_orig]) if self.mapper is not None else y_orig
            return x, y

    test_ds = Wrap(base, tf, label_mapper)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # --- Stack all test data (AutoAttack expects tensors) ---
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    x_test = torch.cat(xs, dim=0).to(device)
    y_test = torch.cat(ys, dim=0).to(device)

    # --- Default eps values if not set ---
    if eps is None:
        if norm == "Linf":
            eps = 8/255  # standard for CIFAR
        elif norm == "L2":
            eps = 3.0
        else:
            raise ValueError(f"Unknown norm {norm}")

    # --- Run AutoAttack ---
    adversary = AutoAttack(model, norm=norm, eps=eps, version=version)
    print(f"\n=== Running AutoAttack ({norm}, eps={eps}) ===")
    adv_acc = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

    print(f"AutoAttack {norm} @ eps={eps}: accuracy = {adv_acc:.2f}%")
    return adv_acc


# =====================================================================================
# Wiring it all together (example main)
# =====================================================================================
def prefix(config):
    parts = [str(config["seed"]), config["dataset_name"], str(config["dataset_num_task"]),
             config["model_backbone"], config["train_merge"], "lca"]
    return "_".join(parts)

def alignment_checkpoint(config):
    postfix = config.get("train_checkpoint_postfix", "")
    return os.path.join("checkpoints", f"{prefix(config)}_alignment_{postfix}.pt")

if __name__ == "__main__":
    config = {
        "seed": 1993,
        "dataset_name": "cifar224",
        "dataset_num_task": 10,
        "dataset_init_cls": 10,
        "dataset_increment": 10,
        "model_backbone": "vit_base_patch16_224_lora",
        "model_lora_r": 64,
        "model_lora_alpha": 128,
        "model_lora_dropout": 0.0,
        "model_lora_target_modules": ["qkv"],
        "model_classifier": ["mlp"],
        "train_merge": "ties",
        "train_checkpoint_postfix": "simple_cil",
    }

    # DataManager (for class order)
    data_manager = DataManager(
        config["dataset_name"],
        True,
        config["seed"],
        config["dataset_init_cls"],
        config["dataset_increment"],
        False,
    )
    label_mapper = build_label_mapper_from_dm(data_manager)

    # Model
    model = Model(config)
    for _ in range(config["dataset_num_task"]):
        model.update_classifier(10, True)
    model.cuda()

    # Load aligned model
    filename = alignment_checkpoint(config)
    print(f"Loading aligned model from {filename}")
    state = torch.load(filename, map_location="cuda")
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:    print("[load] missing:", missing)
    if unexpected: print("[load] unexpected:", unexpected)

    # Clean CIFAR-100 @224 (no norm, remapped)
    eval_cifar100_clean(model, batch_size=256, num_workers=4, device="cuda",
                        data_root="/home/lis/data", label_mapper=label_mapper)

    # CIFAR-100-C @224 (no norm, remapped)  -- optional
    cifar_c_dir = "/home/lis/data/CIFAR-100-C"
    eval_cifar100c(model, cifar_c_dir, batch_size=256, num_workers=4,
                   severities=(1,2,3,4,5), device="cuda", label_mapper=label_mapper)

    # CIFAR-100-P @224 (no norm, remapped): official only
    cifar_p_dir = "/home/lis/data/CIFAR-100-P/CIFAR-100-P"
    eval_cifar100p(model, cifar_p_dir, batch_size=64, 
                   device="cuda", label_mapper=label_mapper, include_extra=False)

    # CIFAR-100-P @224 including EXTRA perturbations
    # eval_cifar100p(model, cifar_p_dir, batch_size=64, 
    #                device="cuda", label_mapper=label_mapper, include_extra=True)

    # # AutoAttack (L_inf)
    # eval_autoattack(model, batch_size=8, device="cuda",
    #                 data_root="/home/lis/data",
    #                 label_mapper=label_mapper,
    #                 norm="Linf", eps=8/255)

    # # AutoAttack (L2)
    # eval_autoattack(model, batch_size=8, device="cuda",
    #                 data_root="/home/lis/data",
    #                 label_mapper=label_mapper,
    #                 norm="L2", eps=3.0)
