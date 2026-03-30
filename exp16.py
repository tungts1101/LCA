from typing import Dict, List
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from utils.data_manager import DataManager
import gc
import time
from helper import (
    Model,
    CosineLinear,
    ContinualLinear,
    compute_metrics,
    accuracy,
    set_random,
    merge,
    count_parameters,
    seed_worker
)
from torch.distributions import MultivariateNormal
import logging
import sys
import optuna
import copy
import random
import math

from scipy.optimize import linear_sum_assignment


def merge_tasks_rankwise(task1, task2, is_perm_one=True, lamb=1.0, scale=1.0):
    def build_lora_AB_groups_from_keys(keys: List[str]) -> List[Dict[str, str]]:
        keyset = set(keys)
        groups = []
        seen = set()

        for k in keys:
            if ".lora_A." in k:
                b = k.replace(".lora_A.", ".lora_B.")
                if b in keyset:
                    if (k, b) not in seen:
                        seen.add((k, b))
                        groups.append({"A": k, "B": b})
            elif "adaptmlp" in k:
                if k not in seen:
                    seen.add(k)
                    groups.append({"A": k, "B": k})
        return groups

    @torch.no_grad()
    def lora_rank_similarity(A1, B1, A2, B2):
        B1n = F.normalize(B1, dim=0)
        B2n = F.normalize(B2, dim=0)
        A1n = F.normalize(A1, dim=1)
        A2n = F.normalize(A2, dim=1)
        return (B1n.T @ B2n) * (A1n @ A2n.T)

    @torch.no_grad()
    def align_lora_ranks_permute_A1B1(A1, B1, A2, B2):
        S = lora_rank_similarity(A1, B1, A2, B2)
        cost = (S.max() - S).cpu().numpy()
        _, col = linear_sum_assignment(cost)
        col = torch.tensor(col, device=A1.device)

        # print(col)

        if is_perm_one:
            A1p = A1[col, :]
            B1p = B1[:, col]

            return A1p, B1p
        else:
            A2p = A2[col, :]
            B2p = B2[:, col]

            return A2p, B2p

    @torch.no_grad()
    def lora_rank_energy(A, B):
        return torch.norm(B, dim=0) * torch.norm(A, dim=1)
        # return torch.sum(torch.abs(B), dim=0) * torch.sum(torch.abs(A), dim=1)

    merged = {}
    keys = list(task1.keys())
    ab_groups = build_lora_AB_groups_from_keys(keys)

    for g in ab_groups:
        A_key, B_key = g["A"], g["B"]

        A1, B1 = task1[A_key], task1[B_key]
        A2, B2 = task2[A_key], task2[B_key]

        # # 1) Git Re-Basin: align ranks by permuting A1,B1
        if is_perm_one:
            A1p, B1p = align_lora_ranks_permute_A1B1(A1, B1, A2, B2)
        else:
            A2p, B2p = align_lora_ranks_permute_A1B1(A1, B1, A2, B2)

        # # 2) Rank-wise structured merge (past-favoring)
        if is_perm_one:
            e1 = lora_rank_energy(A1p, B1p)
            e2 = lora_rank_energy(A2, B2)
        else:
            e1 = lora_rank_energy(A1, B1)
            e2 = lora_rank_energy(A2p, B2p)

        # print(torch.min(e1), torch.max(e1), torch.min(e2), torch.max(e2))
        keep1 = e1 >= lamb * e2

        if is_perm_one:
            A = torch.where(keep1[:, None], A1p, A2)
            B = torch.where(keep1[None, :], B1p, B2)
        else:
            A = torch.where(keep1[:, None], A1, A2p)
            B = torch.where(keep1[None, :], B1, B2p)

        merged[A_key] = A * scale
        merged[B_key] = B * scale

    return merged

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

g = torch.Generator()
g.manual_seed(0)

class Learner:
    def __init__(self, config):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self._mlp_matrix = []
        self._ncm_matrix = []
        self._cls_to_task_idx = {}
        self._acc = 0.0
        self._acc_history = []
        self._anchors = None

        self.model = Model(config)
        self.model.cuda()
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )

        self.ncm_classifier = None
        self._analysis_results = []

        self._feature_dim = self.model.feature_dim
        self.Ws = []
        self.W_rand = None

        # assert config
        train_first_task_only = self._config.get("train_first_task_only", False)
        train_merge = self._config.get("train_merge", "none")
        classifiers = self._config.get("model_classifier", ["mlp"])
        train_ca = self._config.get("train_ca", False)

        if train_first_task_only and train_merge != "none":
            raise ValueError("train_merge must be 'none' when train_first_task_only is True")
        
        if "mlp" not in classifiers and train_ca:
            raise ValueError("train_ca requires 'mlp' classifier")
        

    def learn(self, data_manager):
        self.data_manager = data_manager

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self.model.cuda()

        train_RP = self._config.get("train_RP", False)
        if train_RP:
            self.setup_RP()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()

    def before_task(self, task, data_manager):
        task_size = data_manager.get_task_size(task)
        self._total_classes = self._known_classes + task_size
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

    def after_task(self):
        self._known_classes = self._total_classes

    def update_ncm_classifier(self):
        classifier = CosineLinear(self._feature_dim, self._total_classes)
        if self.ncm_classifier is not None:
            nb_output = self.ncm_classifier.out_features
            weight = copy.deepcopy(self.ncm_classifier.weight.data)
            classifier.weight.data[:nb_output] = weight

        del self.ncm_classifier
        self.ncm_classifier = classifier
        self.ncm_classifier.cuda()

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, 
                                 num_workers=4, worker_init_fn=seed_worker, generator=g)

        self.model.eval()
        
        y_true, y_pred_mlp, y_pred_ncm = [], [], []
        classifiers = self._config.get("model_classifier", ["mlp"])
        
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                
                if "mlp" in classifiers:
                    logits = self.model(x)
                    predicts_mlp = logits.argmax(dim=1)
                    y_pred_mlp.append(predicts_mlp.cpu().numpy())
                
                if "ncm" in classifiers:
                    z = self.get_features(x).cuda()
                    logits = self.ncm_classifier(z)
                    predicts_ncm = logits.argmax(dim=1)
                    y_pred_ncm.append(predicts_ncm.cpu().numpy())
                
                y_true.append(y.cpu().numpy())

        logging.info(f"[Evaluation] Task {self._cur_task}")
        num_tasks = self._cur_task + 1
        y_true = np.concatenate(y_true)

        if y_pred_mlp:
            y_pred_mlp = np.concatenate(y_pred_mlp)
            acc_total_mlp, grouped_mlp = accuracy(y_pred_mlp.T, y_true, self._class_increments)
            grouped_mlp = [float(acc) for acc in grouped_mlp]
            self._mlp_matrix.append(grouped_mlp)
            logging.info(f"[Evaluation MLP] Total Acc: {acc_total_mlp:.2f}, Grouped Acc: {grouped_mlp}")

            mlp_accuracy_matrix = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    mlp_accuracy_matrix[i, j] = self._mlp_matrix[i][j]
            faa_mlp, ffm_mlp, ffd_mlp, asa_mlp = compute_metrics(mlp_accuracy_matrix)
            logging.info(
                f"[Evaluation MLP] FAA: {faa_mlp:.2f}, FFM: {ffm_mlp:.2f}, FFD: {ffd_mlp:.2f}, ASA: {asa_mlp:.2f}"
            )

            self._analysis_results.append(list(grouped_mlp))
        else:
            faa_mlp = asa_mlp = ffm_mlp = 0.0

        if y_pred_ncm:
            y_pred_ncm = np.concatenate(y_pred_ncm)
            acc_total_ncm, grouped_ncm = accuracy(y_pred_ncm.T, y_true, self._class_increments)
            grouped_ncm = [float(acc) for acc in grouped_ncm]
            self._ncm_matrix.append(grouped_ncm)
            logging.info(f"[Evaluation NCM] Total Acc: {acc_total_ncm:.2f}, Grouped Acc: {grouped_ncm}")

            ncm_accuracy_matrix = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    ncm_accuracy_matrix[i, j] = self._ncm_matrix[i][j]
            faa_ncm, ffm_ncm, ffd_ncm, asa_ncm = compute_metrics(ncm_accuracy_matrix)
            logging.info(
                f"[Evaluation NCM] FAA: {faa_ncm:.2f}, FFM: {ffm_ncm:.2f}, FFD: {ffd_ncm:.2f}, ASA: {asa_ncm:.2f}"
            )
        else:
            faa_ncm = asa_ncm = ffm_ncm = 0.0

        self._faa_mlp = faa_mlp
        self._ffm_mlp = ffm_mlp
        self._asa_mlp = asa_mlp
        self._faa_ncm = faa_ncm
        self._ffm_ncm = ffm_ncm
        self._asa_ncm = asa_ncm
        
        self._acc = max(asa_mlp, asa_ncm)
        self._acc_history.append(float(np.round(self._acc, 2)))

    def train(self):
        trainset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        train_loader = DataLoader(
            trainset,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        prototype_set = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )

        prototype_loader = DataLoader(
            prototype_set,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        classifiers = self._config.get("model_classifier", ["mlp"])
        train_first_task_only = self._config.get("train_first_task_only", False)
        if self._cur_task == 0 or not train_first_task_only:
            self.train_mlp(train_loader)
        
        train_merge = self._config.get("train_merge", "none")
        if train_merge != "none":
            self.merge()
        
        self.extract_prototypes()
        
        train_ca = self._config.get("train_ca", False)
        if train_ca:
            self.compute_multivariate_normal()
            self.align(self.model.classifier)

        if "ncm" in classifiers:
            self.update_ncm_classifier()
            self.train_prototype(prototype_loader)
    
    def extract_prototypes(self):
        train_feature_at_layer = self._config.get("train_feature_at_layer", [-1])
        all_keys = train_feature_at_layer + ["final"]
        feature_dim = self.model.feature_dim
        total_class = self._total_classes
        token_length = 197

        # Initialize or extend storage for intermediate layers + final layer
        if not hasattr(self, "_layer_class_means"):
            self._layer_class_means = {k: torch.zeros(total_class, token_length, feature_dim) for k in all_keys}
            self._layer_class_stds  = {k: torch.zeros(total_class, token_length, feature_dim) for k in all_keys}
        else:
            for k in all_keys:
                if k not in self._layer_class_means:
                    self._layer_class_means[k] = torch.zeros(total_class, token_length, feature_dim)
                    self._layer_class_stds[k]  = torch.zeros(total_class, token_length, feature_dim)
                else:
                    new_means = torch.zeros(total_class, token_length, feature_dim)
                    new_means[:self._known_classes] = self._layer_class_means[k]
                    self._layer_class_means[k] = new_means
                    new_stds = torch.zeros(total_class, token_length, feature_dim)
                    new_stds[:self._known_classes] = self._layer_class_stds[k]
                    self._layer_class_stds[k] = new_stds

        for cls_idx in range(self._known_classes, self._total_classes):
            train_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            train_loader = DataLoader(
                train_set, batch_size=512, shuffle=False,
                num_workers=4, worker_init_fn=seed_worker, generator=g
            )

            layer_feats = {k: [] for k in all_keys}

            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(train_loader):
                    x = x.cuda()
                    z = self.model.get_features(x, return_layer_features=True)  # (B, D)

                    for l in train_feature_at_layer:
                        layer_feats[l].append(self.model.layer_features[l].cpu())

                    layer_feats["final"].append(z.cpu())

            for k in all_keys:
                feats = torch.cat(layer_feats[k], dim=0)
                self._layer_class_means[k][cls_idx] = feats.mean(dim=0)
                self._layer_class_stds[k][cls_idx] = feats.std(dim=0, unbiased=True)

        # Release last batch's intermediate tensors captured by hooks
        self.model.layer_features = []
        torch.cuda.empty_cache()


    def train_mlp(self, train_loader):
        logging.info(f"[Training] Task {self._cur_task}")

        self.model.update_classifier(
            self._total_classes - self._known_classes, 
            with_norm=True, with_bias=False, freeze_old=True, norm_layer="ln"
        )
        self.model.cuda()

        reset_train = self._config.get("reset_train", False)
        if not reset_train:
            saved_backbone_checkpoint = self.backbone_checkpoint(self._cur_task)
            if os.path.exists(saved_backbone_checkpoint):
                logging.info(f"[Training] Load backbone checkpoint for task {self._cur_task}")
                backbone_params = torch.load(saved_backbone_checkpoint)
                self.load_backbone(backbone_params)

                saved_mlp_checkpoint = self.mlp_checkpoint(self._cur_task)
                if os.path.exists(saved_mlp_checkpoint):
                    logging.info(f"[Training] Load mlp checkpoint for task {self._cur_task}")
                    self.model.classifier.heads[self._cur_task].load_state_dict(
                        torch.load(saved_mlp_checkpoint), strict=True
                    )

                return

        train_first_task_only = self._config.get("train_first_task_only", False)
        if not train_first_task_only:
            train_checkpoint_from = -1
            train_incremental = self._config.get("train_incremental", False)
            if train_incremental:
                train_checkpoint_from = self._cur_task - 1
            logging.info(f"[Training] Start from checkpoint {train_checkpoint_from}")
            self.load_backbone(torch.load(self.backbone_checkpoint(train_checkpoint_from)), load_norm=False)

        self.model.train()
        logging.info(f"[Training] {self.model}")
        
        epochs = self._config["train_epochs"]
        base_lr = self._config["train_base_lr"]
        weight_decay = self._config["train_weight_decay"]

        parameters = [
            {
                "params": [
                    p for p in self.model.backbone.parameters() if p.requires_grad
                ],
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for p in self.model.classifier.heads[
                        self._cur_task
                    ].parameters()
                    if p.requires_grad
                ],
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]

        optimizer = optim.SGD(
            parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        train_reg_weight = self._config.get("train_reg_weight", 0.2)
        train_feature_at_layer = self._config.get("train_feature_at_layer", [-1])
        train_reg_samples = self._config.get("train_reg_samples", 8)
        train_reg_batch_size = self._config.get("train_reg_batch_size", 64)
        use_reg = train_reg_weight > 0 and self._cur_task > 0 and hasattr(self, "_layer_class_means")

        reg_samples_cpu = {}
        if use_reg:
            n_old = self._known_classes
            for l in train_feature_at_layer:
                layer_samples, layer_targets = [], []
                for clz in range(n_old):
                    clz_mean = self._layer_class_means[l][clz]    # (197, D)
                    clz_std  = self._layer_class_stds[l][clz]     # (197, D)
                    clz_samples = clz_mean.unsqueeze(0) + \
                        torch.randn(train_reg_samples, *clz_mean.shape) * clz_std.unsqueeze(0)  # (N, 197, D)
                    layer_samples.append(clz_samples)
                    clz_target = self._layer_class_means["final"][clz][0]  # (D,) — CLS token
                    layer_targets.append(clz_target.unsqueeze(0).expand(train_reg_samples, -1).clone())
                all_samples = torch.cat(layer_samples, dim=0)   # (n_old*N, 197, D)
                all_targets = torch.cat(layer_targets, dim=0)   # (n_old*N, D)
                perm = torch.randperm(all_samples.shape[0])
                all_samples = all_samples[perm]
                all_targets = all_targets[perm]
                reg_samples_cpu[l] = (all_samples.cpu().pin_memory(), all_targets.cpu().pin_memory())

        for epoch in range(epochs):
            total_loss, total_reg_loss, total_acc, total = 0, 0, 0, 0

            for _, (_, _, x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = x.cuda(), y.cuda()
                y = torch.where(
                    y - self._known_classes >= 0,
                    y - self._known_classes,
                    -100
                )

                z = self.model.get_features(x)
                logits = self.model.classifier.heads[-1](z)
                loss = F.cross_entropy(logits, y, ignore_index=-100)

                if use_reg:
                    reg_loss = torch.tensor(0.0, device="cuda")
                    for l in train_feature_at_layer:
                        samples, target = reg_samples_cpu[l]
                        idx = torch.randint(0, samples.shape[0], (train_reg_batch_size,))
                        chunk        = samples[idx].cuda(non_blocking=True)
                        chunk_target = target[idx].cuda(non_blocking=True)
                        proj = self.model.forward_from_block(chunk, l + 1)
                        reg_loss = reg_loss + F.mse_loss(proj, chunk_target)

                    loss = loss + train_reg_weight * reg_loss / len(train_feature_at_layer)
                    total_reg_loss += reg_loss.item() * len(y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

            scheduler.step()
            if epoch % 5 == 4 or epoch == epochs - 1:
                logging.info(
                    f"[Training] Epoch {epoch + 1}/{epochs}, "
                    f"Total Loss: {total_loss / total:.4f}, "
                    f"Reg Loss: {total_reg_loss / total:.4f}, "
                    f"Acc: {total_acc / total:.4f}"
                )

        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task)
        )
        torch.save(
            self.model.classifier.heads[self._cur_task].state_dict(),
            self.mlp_checkpoint(self._cur_task),
        )

        if train_first_task_only:
            # freeze backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def merge(self):
        logging.info(f"[Merging] Task {self._cur_task}")

        reset_merge = self._config.get("reset_merge", False)
        if not reset_merge:
            saved_merge_checkpoint = self.merged_checkpoint(self._cur_task)
            if os.path.exists(saved_merge_checkpoint):
                logging.info(f"[Merging] Load merged checkpoint for task {self._cur_task}")
                backbone_params = torch.load(self.merged_checkpoint(self._cur_task))
                self.load_backbone(backbone_params)
                return

        if self._cur_task > 0:
            base_params = torch.load(self.backbone_checkpoint(-1))
            num_merged_params = sum(param.numel() for param in base_params.values())
            logging.info(f"[Merging] Merging with {num_merged_params:,} total parameters")

            if self._config.get("train_merge_incremental", False):
                task_params = []
                task_params.append(torch.load(self.merged_checkpoint(self._cur_task - 1)))
                task_params.append(torch.load(self.backbone_checkpoint(self._cur_task)))
            else:
                task_params = [
                    torch.load(self.backbone_checkpoint(task))
                    for task in range(self._cur_task + 1)
                ]
            logging.info(f"[Merging] Loaded {len(task_params)} tasks for merging")

            if self._config["train_merge"] == "ties":
                backbone_params = merge(
                    base_params,
                    task_params,
                    method=self._config["train_merge"],
                    lamb=self._config["train_merge_coef"],
                    topk=self._config["train_merge_topk"],
                )
            elif self._config["train_merge"] == "rankwise":
                rankwise_merge_lamb = self._config.get("train_merge_rankwise_lamb", 1.0)
                backbone_params = merge_tasks_rankwise(
                    task_params[0], task_params[1], is_perm_one=False, lamb=rankwise_merge_lamb, scale=1.0)
                
            self.load_backbone(backbone_params, load_norm=False)
        
        logging.info(
            f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
        )
        torch.save(
            self.model.get_backbone_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )

    def setup_RP(self):
        M = 10000
        self._feature_dim = M
        self.W_rand = torch.randn(self.model.feature_dim, M, generator=g).cuda()

        # prune RP
        print(f"[RP] Pruning")
        p = 0.9
        num_elements = self.W_rand.numel()
        num_keep = int(num_elements * (1 - p))  # Keep 10%
        flat_W = self.W_rand.view(-1)
        
        _, top_indices = torch.topk(torch.abs(flat_W), num_keep, largest=True)
        
        mask = torch.zeros_like(flat_W, dtype=torch.bool)
        mask[top_indices] = True
        flat_W[~mask] = 0
        
        self.W_rand = flat_W.view(self.W_rand.shape)
        
        self.Q = torch.zeros(M, self._total_classnum)
        self.G = torch.zeros(M, M)
        print(f"[RP] Setup random projection with M={M}")
        print(f"[RP] W_rand shape: {self.W_rand.shape}, Q shape: {self.Q.shape}, G shape: {self.G.shape}")

    @torch.no_grad()
    def get_features(self, x):
        f = self.model.get_features(x)
        if self.W_rand != None:
            f = F.relu(f @ self.W_rand)
        return f

    def train_prototype(self, prototype_loader):
        logging.info(f"[Prototype] Task {self._cur_task}")

        self.model.eval()
        Features_h = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(prototype_loader):
                (_,_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self.get_features(data)
                Features_h.append(embedding.cpu())
                label_list.append(label.cpu())

        Features_h = torch.cat(Features_h, dim=0)
        label_list = torch.cat(label_list, dim=0)
        Y = F.one_hot(label_list, num_classes=self._total_classnum).float()

        train_RP = self._config.get("train_RP", False)
        if train_RP:
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better ncmrical stability than .inv
            self.ncm_classifier.weight.data = Wo[0:self.ncm_classifier.weight.shape[0],:].cuda()
        else:
            for class_idx in set(label_list.numpy()):
                class_features = Features_h[label_list == class_idx]
                class_prototype = class_features.mean(dim=0)
                self.ncm_classifier.weight.data[class_idx] = class_prototype.cuda()
    
    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better ncmrical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("[RP] Optimal lambda: "+str(ridge))
        return ridge
    
    def compute_multivariate_normal(self):
        logging.info(
            f"[Alignment] Compute class mean and cov for classes {self._known_classes} - {self._total_classes - 1}"
        )
        total_class = self._total_classes
        feature_dim = self.model.feature_dim
        if not hasattr(self, "_class_means") or not hasattr(self, "_class_covs"):
            self._class_means = torch.zeros((total_class, feature_dim))
            self._class_covs = torch.zeros((total_class, feature_dim, feature_dim))
        else:
            new_class_means = torch.zeros((total_class, feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_covs = torch.zeros((total_class, feature_dim, feature_dim))
            new_class_covs[: self._known_classes] = self._class_covs
            self._class_covs = new_class_covs

        train_ca_load_checkpoint_from_first_task = self._config.get(
            "train_ca_load_checkpoint_from_first_task", False
        )
        if train_ca_load_checkpoint_from_first_task:
            logging.info("[Alignment] Load backbone from first task for computing class statistics")
            self.load_backbone(torch.load(self.backbone_checkpoint(0)))

        for cls_idx in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            proto_loader = DataLoader(
                proto_set, batch_size=512, shuffle=False, 
                num_workers=4, worker_init_fn=seed_worker, generator=g
            )

            features_list = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(proto_loader):
                    x = x.cuda()
                    f = self.model.get_features(x)
                    features_list.append(f.cpu())

            features_list = torch.cat(features_list, dim=0)
            class_mean = torch.mean(features_list, dim=0)
            class_cov = (
                torch.cov(features_list.T) + torch.eye(class_mean.shape[-1]) * 1e-4
            )

            self._class_means[cls_idx, :] = class_mean
            self._class_covs[cls_idx, ...] = class_cov
        
        if train_ca_load_checkpoint_from_first_task:
            logging.info("[Alignment] Restore backbone after computing class statistics")
            self.load_backbone(torch.load(self.backbone_checkpoint(self._cur_task)))

    def align(self, classifier):
        logging.info(f"[Alignment] Task {self._cur_task}")
        samples_per_cls = self._config.get("train_ca_samples_per_cls", 256)

        epochs = self._config.get("train_ca_epochs", 10)
        batch_size = self._config.get("train_ca_batch_size", 64)
        robust_weight_base = self._config.get("train_ca_robust_weight", 0.0)

        for p in classifier.parameters():
            p.requires_grad = False

        trainable_params = []
        for p in classifier.parameters():
            p.requires_grad = True
            trainable_params.append(p)
            
        num_trainable = count_parameters(classifier, trainable=True)
        logging.info(f"[Alignment] Num trainable parameters: {num_trainable:,}")

        optimizer = optim.SGD(trainable_params, lr=1e-2, momentum=0.9, weight_decay=1e-4)

        sampled_data, sampled_label = [], []
        for cls_idx in range(self._total_classes):
            cls_mean = self._class_means[cls_idx].cuda()
            cls_cov = self._class_covs[cls_idx].cuda()

            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_features = m.sample((samples_per_cls,))

            sampled_data.append(sampled_features)
            sampled_label.extend([cls_idx] * samples_per_cls)

        sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
        sampled_label = torch.tensor(sampled_label).long().cuda()

        indexes = torch.randperm(sampled_data.size(0))
        sampled_data = sampled_data[indexes]
        sampled_label = sampled_label[indexes]

        # for epoch in range(epochs):
        #     total_loss = total_acc = total = 0
        #     num_samples = sampled_data.size(0)
        #     num_iterations = (num_samples + batch_size - 1) // batch_size
        #     for _iter in range(num_iterations):
        #         start_idx = _iter * batch_size
        #         end_idx = min((_iter + 1) * batch_size, num_samples)

        #         x = sampled_data[start_idx:end_idx]
        #         y = sampled_label[start_idx:end_idx]

        #         logits = classifier(x)
        #         loss = F.cross_entropy(logits, y)

        #         if torch.isnan(loss):
        #             continue

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         bs = len(y)
        #         total_loss += loss.item() * bs
        #         total_acc += (logits.argmax(dim=1) == y).sum().item()
        #         total += bs
            
        #     logging.info(
        #         f"[Alignment] Epoch {epoch+1}/{epochs}, "
        #         f"Loss: {total_loss/max(total, 1):.4f}, "
        #         f"Accuracy: {total_acc/max(total, 1):.4f}"
        #     )

        for epoch in range(epochs):
            total_loss = 0
            total = 0
            total_ce_loss = total_rb_loss = 0
            total_acc = 0

            num_samples = sampled_data.size(0)
            num_iterations = (num_samples + batch_size - 1) // batch_size

            for _iter in range(num_iterations):
                start_idx = _iter * batch_size
                end_idx = min((_iter + 1) * batch_size, num_samples)

                x = sampled_data[start_idx:end_idx]
                y = sampled_label[start_idx:end_idx]

                logits = classifier(x)
                loss_vec = F.cross_entropy(logits, y, reduction="none")

                base_loss = loss_vec.mean()
                if torch.isnan(base_loss):
                    continue

                reg_loss = torch.tensor(0.0, device=x.device)
                if robust_weight_base > 0:
                    unique_classes = torch.unique(y)
                    class_dist = torch.cdist(
                        x, self._class_means[: self._total_classes].cuda()
                    )
                    class_indices = torch.argmin(class_dist, dim=1)

                    for class_i in unique_classes:
                        label_mask = (y == class_i)
                        distance_mask = (class_indices == class_i)
                        class_mask = distance_mask & label_mask

                        class_samples = torch.where(class_mask)[0]

                        if len(class_samples) == 0:
                            label_only_samples = torch.where(label_mask)[0]
                            if len(label_only_samples) == 0:
                                continue
                            class_losses = loss_vec[label_mask]
                        else:
                            class_losses = loss_vec[class_mask]

                        if len(class_losses) >= 2:
                            pairwise_diffs = torch.abs(
                                class_losses.unsqueeze(1)
                                - class_losses.unsqueeze(0)
                            )
                            # Remove diagonal (self-comparisons)
                            mask = ~torch.eye(
                                len(class_losses), dtype=torch.bool, device=x.device
                            )
                            pairwise_diffs = pairwise_diffs[mask]
                            reg_loss += pairwise_diffs.mean()
                    
                    if len(unique_classes) > 0:
                        reg_loss = reg_loss / len(unique_classes)
                    
                loss = base_loss + robust_weight_base * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = len(y)
                total_loss += loss.item() * bs
                total_ce_loss += base_loss.item() * bs
                total_rb_loss += reg_loss.item() * bs
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += bs

            if epoch % 5 == 4 or epoch == epochs - 1:
                logging.info(
                    f"[Alignment] Epoch {epoch+1}/{epochs}, "
                    f"Base Loss: {total_ce_loss/max(total, 1):.4f}, "
                    f"Robust Term: {total_rb_loss/max(total, 1):.4f}, "
                    f"Total Loss: {total_loss/max(total, 1):.4f}, "
                    f"Accuracy: {total_acc/max(total, 1):.4f}"
                )

    def prefix(self):
        prefix_parts = [
            str(self._config["seed"]),
            self._config["dataset_name"],
            self._config["model_backbone"],
        ]

        train_prefix = self._config.get("train_prefix", "")
        if train_prefix:
            prefix_parts.append(train_prefix)

        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        filename = f"{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )
        return os.path.join(CHECKPOINT_DIR, filename)
    
    def mlp_checkpoint(self, task):
        filename = f"{self.prefix()}_mlp_{task}.pt"
        return os.path.join(CHECKPOINT_DIR, filename)

    def merged_checkpoint(self, task):
        filename = f"{self.prefix()}_merged_{self._config['train_merge']}_{task}.pt"
        return os.path.join(CHECKPOINT_DIR, filename)

    def load_backbone(self, backbone_params, load_norm=True):
        peft_params = {}
        norm_params = {}
        for name, param in backbone_params.items():
            if name.startswith("norm."):
                norm_name = name[5:]
                norm_params[norm_name] = param
            else:
                peft_params[name] = param
        self.model.backbone.load_state_dict(peft_params, strict=False)
        if norm_params and load_norm:
            self.model.norm.load_state_dict(norm_params, strict=True)

DATA_TABLE = {
    # "cifar224": [(10, 10, 10)],
    # "imagenetr": [(10, 20, 20)],
    "imageneta": [(10, 20, 20)],
    # "cub": [(10, 20, 20)],
    # "omnibenchmark": [(10, 30, 30)],
    # "vtab": [(5, 10, 10)],
    # "cars": [(10, 16, 20)]
}

BASE_CONFIG = {
    "seed": [1993],
    "reset_train": False,
    "reset_merge": False,
    "train_epochs": 10,
    "train_batch_size": 64,
    "train_base_lr": 1e-2,
    "train_weight_decay": 5e-4,
    
    "model_backbone": "vit_base_patch16_224_lora",
    "model_lora_r": 64,
    "model_lora_alpha": 128,
    "model_lora_dropout": 0.0,
    "model_lora_target_modules": ["qkv"],
    "model_classifier": ["ncm"],
}

def run_single_experiment(dataset_name, config_name, experiment_config, seed):
    config = copy.deepcopy(BASE_CONFIG)
    config["seed"] = seed

    set_random(config["seed"])
    
    dataset_num_task, dataset_init_cls, dataset_increment = DATA_TABLE[dataset_name][0]
    dataset_config = {
        "dataset_name": dataset_name,
        "dataset_num_task": dataset_num_task,
        "dataset_init_cls": dataset_init_cls,
        "dataset_increment": dataset_increment,
    }
    config.update(dataset_config)
    
    data_manager = DataManager(
        config["dataset_name"],
        True,
        config["seed"],
        config["dataset_init_cls"],
        config["dataset_increment"],
        False,
    )

    config.update(experiment_config)
    
    if dataset_name == "vtab" or dataset_name == "cars":
        config["train_merge_rankwise_lamb"] = 0.15
    
    if dataset_name == "imageneta":
        config["train_batch_size"] = 48
    
    experiment_name = f"{dataset_name}_{config_name}"
    result = {}
    try:
        logging.info("Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

        learner = Learner(config)
        learner.learn(data_manager)
        
        mlp_faa = learner._faa_mlp
        mlp_ffm = learner._ffm_mlp
        mlp_asa = learner._asa_mlp
        ncm_faa = learner._faa_ncm
        ncm_ffm = learner._ffm_ncm
        ncm_asa = learner._asa_ncm

        del learner
        torch.cuda.empty_cache()
        gc.collect()

        result["mlp_faa"] = mlp_faa
        result["mlp_ffm"] = mlp_ffm
        result["mlp_asa"] = mlp_asa
        result["ncm_faa"] = ncm_faa
        result["ncm_ffm"] = ncm_ffm
        result["ncm_asa"] = ncm_asa
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"[Experiment {experiment_name}] Detailed Error:")
        logging.error(f"Exception Type: {type(e).__name__}")
        logging.error(f"Exception Message: {str(e)}")
        logging.error(f"Full Traceback:\n{error_details}")

        result["mlp_faa"] = 0.0
        result["mlp_ffm"] = 0.0
        result["mlp_asa"] = 0.0
        result["ncm_faa"] = 0.0
        result["ncm_ffm"] = 0.0
        result["ncm_asa"] = 0.0

    return result

def run_experiments():
    seeds = [1993]

    experiment_configs = {
        "exp15": {
            "reset_train": True,
            "reset_merge": True,
            "train_epochs": 10,
            "train_batch_size": 64,
            "model_backbone": "vit_base_patch16_224_lora",
            "model_outdim": 768,
            "model_use_norm": False,
            "model_lora_r": 64,
            "model_lora_alpha": 128,
            "ffn_num": 64,
            "model_lora_dropout": 0.0,
            "model_lora_target_modules": ["qkv"],
            "model_classifier": ["mlp"],

            "train_prefix": "exp16",

            "train_feature_at_layer": [1],

            "train_first_task_only": False,
            "train_incremental": False,
            "train_RP": False,

            "train_merge": "ties",
            "train_merge_coef": 1.0,
            "train_merge_topk": 100,
            "train_merge_incremental": False,
            "train_merge_rankwise_lamb": 0.85,

            "train_ca": True,
            "train_ca_load_checkpoint_from_first_task": False,
            "train_ca_samples_per_cls": 512,
            "train_ca_batch_size": 64,
            "train_ca_epochs": 3,
            "train_ca_robust_weight": 0.0,

            # vit_base_patch16_224_lora
            # vit_base_patch16_224_21k_lora
            # pretrained_vit_b16_224_adapter
            # vit_base_patch16_dinov3.lvd1689m_lora
            # vit_large_patch16_dinov3.lvd1689m_lora
            # vit_huge_plus_patch16_dinov3.lvd1689m_lora
        },
    }
    
    for dataset_name in DATA_TABLE.keys():
        print(f"\n{'='*60}")
        print(f"Starting experiments for dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset_results = {}

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        for config_name, config in experiment_configs.items():
            dir_path = os.path.join(LOG_DIR, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            logfilename = os.path.join(dir_path, config_name + ".log")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(filename)s] => %(message)s",
                handlers=[
                    logging.FileHandler(filename=logfilename),
                    logging.StreamHandler(sys.stdout),
                ],
                force=True
            )
            
            for seed in seeds:
                logging.info("\n" + "=" * 80)
                logging.info(f"Starting experiment: {dataset_name} - {config_name} - seed {seed}")
                experiment_start_time = time.time()
                result = run_single_experiment(dataset_name, config_name, config, seed)
                experiment_end_time = time.time()
                logging.info(f"Experiment {dataset_name}_{config_name}_seed{seed} time: {experiment_end_time - experiment_start_time:.2f} seconds")
                
                if config_name not in dataset_results:
                    dataset_results[config_name] = {
                        'seeds': [],
                        'mlp_faa': [],
                        'mlp_ffm': [],
                        'mlp_asa': [],
                        'ncm_faa': [],
                        'ncm_ffm': [],
                        'ncm_asa': []
                    }
                
                dataset_results[config_name]['seeds'].append(seed)
                dataset_results[config_name]['mlp_faa'].append(result.get('mlp_faa', 0.0))
                dataset_results[config_name]['mlp_ffm'].append(result.get('mlp_ffm', 0.0))
                dataset_results[config_name]['mlp_asa'].append(result.get('mlp_asa', 0.0))
                dataset_results[config_name]['ncm_faa'].append(result.get('ncm_faa', 0.0))
                dataset_results[config_name]['ncm_ffm'].append(result.get('ncm_ffm', 0.0))
                dataset_results[config_name]['ncm_asa'].append(result.get('ncm_asa', 0.0))

            logging.info("\n" + "="*80)
            logging.info(f"SUMMARY FOR {dataset_name.upper()} - {config_name.upper()}")
            logging.info("="*80)

            if len(dataset_results[config_name]['mlp_asa']) > 0:
                mlp_asa_mean = np.mean(dataset_results[config_name]['mlp_asa'])
                mlp_asa_std = np.std(dataset_results[config_name]['mlp_asa'])
                mlp_faa_mean = np.mean(dataset_results[config_name]['mlp_faa'])
                mlp_faa_std = np.std(dataset_results[config_name]['mlp_faa'])
                mlp_ffm_mean = np.mean(dataset_results[config_name]['mlp_ffm'])
                mlp_ffm_std = np.std(dataset_results[config_name]['mlp_ffm'])
                logging.info(f"  MLP - ASA: {mlp_asa_mean:.2f} ± {mlp_asa_std:.2f}")
                logging.info(f"  MLP - FAA: {mlp_faa_mean:.2f} ± {mlp_faa_std:.2f}")
                logging.info(f"  MLP - FFM: {mlp_ffm_mean:.2f} ± {mlp_ffm_std:.2f}")
            
            if len(dataset_results[config_name]['ncm_asa']) > 0:
                ncm_asa_mean = np.mean(dataset_results[config_name]['ncm_asa'])
                ncm_asa_std = np.std(dataset_results[config_name]['ncm_asa'])
                ncm_faa_mean = np.mean(dataset_results[config_name]['ncm_faa'])
                ncm_faa_std = np.std(dataset_results[config_name]['ncm_faa'])
                ncm_ffm_mean = np.mean(dataset_results[config_name]['ncm_ffm'])
                ncm_ffm_std = np.std(dataset_results[config_name]['ncm_ffm'])
                logging.info(f"  NCM - ASA: {ncm_asa_mean:.2f} ± {ncm_asa_std:.2f}")
                logging.info(f"  NCM - FAA: {ncm_faa_mean:.2f} ± {ncm_faa_std:.2f}")
                logging.info(f"  NCM - FFM: {ncm_ffm_mean:.2f} ± {ncm_ffm_std:.2f}")
        logging.info("="*80 + "\n")

if __name__ == "__main__":
    start_time = time.time()
    results = run_experiments()
    total_time = time.time() - start_time
    print(f"Total experiment time: {total_time:.2f} seconds")
