import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime
from utils.data_manager import DataManager
import gc
import time
from helper import (
    Model,
    CosineLinear,
    compute_metrics,
    accuracy,
    set_random,
    merge,
    count_parameters
)
from torch.distributions import MultivariateNormal
import logging
import sys
import optuna
import copy


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class Learner:
    def __init__(self, config, trial=None, pruning_thresholds={}):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self._mlp_matrix = []
        self._nme_matrix = []
        self._cls_to_task_idx = {}
        self._acc = 0.0
        self._acc_history = []
        self.trial = trial
        self.pruning_thresholds = pruning_thresholds

        self.model = Model(config)
        self.model.cuda()
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )
        self.nme_classifier = None
    
    def update_nme_classifier(self):
        classifier = CosineLinear(self.model.feature_dim, self._total_classnum)
        if self.nme_classifier is not None:
            nb_output = self.nme_classifier.out_features
            weight = copy.deepcopy(self.nme_classifier.weight.data)
            classifier.weight.data[:nb_output] = weight

        del self.nme_classifier
        self.nme_classifier = classifier

    def learn(self, data_manager):
        self.data_manager = data_manager

        num_tasks = data_manager.nb_tasks
        self._total_classnum = data_manager.get_total_classnum()
        self.model.cuda()

        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()

            if self.trial is not None:
                dataset_name = self._config["dataset_name"]
                if dataset_name in self.pruning_thresholds:
                    thresholds = self.pruning_thresholds[dataset_name]
                    if task in thresholds:
                        threshold = thresholds[task]
                        if self._acc < threshold:
                            logging.info(
                                f"[Pruning] Acc {self._acc:.2f} < {threshold:.2f} at task {task}"
                            )
                            raise optuna.TrialPruned()

        logging.info(f"[Evaluation] Final accuracy history: {self._acc_history}")

    def before_task(self, task, data_manager):
        task_size = data_manager.get_task_size(task)
        self._total_classes = self._known_classes + task_size
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

    def after_task(self):
        self._known_classes = self._total_classes

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

        self.model.eval()
        
        y_true, y_pred_mlp, y_pred_nme = [], [], []
        classifiers = self._config.get("model_classifier", ["mlp"])
        
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                
                if "mlp" in classifiers:
                    logits = self.model(x)
                    predicts_mlp = logits.argmax(dim=1)
                    y_pred_mlp.append(predicts_mlp.cpu().numpy())
                
                if "nme" in classifiers:
                    z = self.model.get_features(x)
                    logits = self.nme_classifier(z)
                    predicts_nme = logits.argmax(dim=1)
                    y_pred_nme.append(predicts_nme.cpu().numpy())
                
                y_true.append(y.cpu().numpy())

        logging.info(f"[Evaluation] Task {self._cur_task}")
        num_tasks = self._cur_task + 1

        if y_pred_mlp:
            y_pred_mlp = np.concatenate(y_pred_mlp)
            y_true = np.concatenate(y_true)
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
        else:
            faa_mlp = asa_mlp = 0.0

        if y_pred_nme:
            y_pred_nme = np.concatenate(y_pred_nme)
            acc_total_nme, grouped_nme = accuracy(y_pred_nme.T, y_true, self._class_increments)
            grouped_nme = [float(acc) for acc in grouped_nme]
            self._nme_matrix.append(grouped_nme)
            logging.info(f"[Evaluation NME] Total Acc: {acc_total_nme:.2f}, Grouped Acc: {grouped_nme}")

            nme_accuracy_matrix = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i + 1):
                    nme_accuracy_matrix[i, j] = self._nme_matrix[i][j]
            faa_nme, ffm_nme, ffd_nme, asa_nme = compute_metrics(nme_accuracy_matrix)
            logging.info(
                f"[Evaluation NME] FAA: {faa_nme:.2f}, FFM: {ffm_nme:.2f}, FFD: {ffd_nme:.2f}, ASA: {asa_nme:.2f}"
            )
        else:
            faa_nme = asa_nme = 0.0

        self._faa_mlp = faa_mlp
        self._asa_mlp = asa_mlp
        self._faa_nme = faa_nme
        self._asa_nme = asa_nme
        
        self._acc = max(asa_mlp, asa_nme)
        self._acc_history.append(float(np.round(self._acc, 2)))

    def train(self):
        self.model.update_classifier(
            self._total_classes - self._known_classes, freeze_old=True
        )
        self.model.train()
        self.model.cuda()

        if not os.path.exists(self.backbone_checkpoint(self._cur_task)):
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
            )

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

            if self.model.norm is not None:
                parameters.append(
                    {
                        "params": [
                            p for p in self.model.norm.parameters() if p.requires_grad
                        ],
                        "lr": base_lr,
                        "weight_decay": weight_decay,
                    }
                )

            optimizer = optim.SGD(
                parameters, lr=base_lr, momentum=0.9, weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )

            logging.info(f"[Training] Task {self._cur_task}")
            logging.info(f"[Training] {self.model}")

            for epoch in range(epochs):
                total_loss, total_acc, total = 0, 0, 0
                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(
                        y - self._known_classes >= 0, y - self._known_classes, -100
                    )
                    z = self.model.get_features(x)
                    logits = self.model.classifier.heads[-1](z)
                    loss = F.cross_entropy(logits, y)

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
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(
                self.model.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
            torch.save(
                self.model.classifier.heads[self._cur_task].state_dict(),
                self.classifier_checkpoint(self._cur_task),
            )
        else:
            logging.info(f"[Training] Load checkpoint for task {self._cur_task}")
            backbone_params = torch.load(self.backbone_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)
            self.model.classifier.heads[self._cur_task].load_state_dict(
                torch.load(self.classifier_checkpoint(self._cur_task)), strict=True
            )
        
        if self._config.get("train_ca", False):
            self.local_align()
        else:
            if "nme" in self._config.get("model_classifier", ["mlp"]):
                self.compute_multivariate_normal()
            self.merge()

    def merge(self):
        if os.path.exists(self.merged_checkpoint(self._cur_task)):
            logging.info(f"[Merging] Load merged checkpoint for task {self._cur_task}")
            backbone_params = torch.load(self.merged_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)
            return

        if self._cur_task == 0:
            logging.info(
                f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
            )
            torch.save(
                self.model.get_backbone_trainable_params(),
                self.merged_checkpoint(self._cur_task),
            )
            return

        logging.info(f"[Merging] Method {self._config['train_merge']}")
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

        # logging.info("[Merging] Norm layer values BEFORE merging:")
        # logging.info(f"  norm.weight: mean={self.model.norm.weight.data.mean():.6f}, std={self.model.norm.weight.data.std():.6f}")
        # logging.info(f"  norm.bias: mean={self.model.norm.bias.data.mean():.6f}, std={self.model.norm.bias.data.std():.6f}")

        backbone_params = merge(
            base_params,
            task_params,
            method=self._config["train_merge"],
            lamb=self._config["train_merge_coef"],
            topk=self._config["train_merge_topk"],
        )
        self.load_backbone(backbone_params)

        # logging.info("[Merging] Norm layer values AFTER merging:")
        # logging.info(f"  norm.weight: mean={self.model.norm.weight.data.mean():.6f}, std={self.model.norm.weight.data.std():.6f}")
        # logging.info(f"  norm.bias: mean={self.model.norm.bias.data.mean():.6f}, std={self.model.norm.bias.data.std():.6f}")

        logging.info(
            f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
        )
        torch.save(
            self.model.get_backbone_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )

    def compute_multivariate_normal(self):
        classifiers = self._config.get("model_classifier", ["mlp"])
        if "nme" in classifiers:
            self.update_nme_classifier()

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

        for cls_idx in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            proto_loader = DataLoader(
                proto_set, batch_size=512, shuffle=False, num_workers=4
            )

            features_list = []
            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, _) in enumerate(proto_loader):
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

            if "nme" in classifiers:
                self.nme_classifier.weight.data[cls_idx, :] = class_mean

    def align(self, classifier):
        # Sample data from Gaussian distributions
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

        # Create optimizer
        epochs = self._config.get("train_ca_epochs", 10)
        samples_per_cls = self._config.get("train_ca_samples_per_cls", 256)
        batch_size = self._config.get("train_ca_batch_size", 64)
        base_lr = self._config.get("train_ca_lr", 1e-2)

        robust_weight_base = self._config.get("train_ca_robust_weight", 0.0)
        entropy_weight = self._config.get("train_ca_entropy_weight", 0.0)
        logit_norm = self._config.get(
            "train_ca_logit_norm", None
        )  # None means no logit norm

        for p in classifier.parameters():
            p.requires_grad = True
            
        num_trainable = count_parameters(classifier, trainable=True)
        logging.info(f"[Alignment] Num trainable parameters: {num_trainable:,}")

        optimizer = optim.SGD(
            classifier.parameters(),
            lr=base_lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

        # Training loop with LCA loss implementation
        for epoch in range(epochs):
            indexes = torch.randperm(sampled_data.size(0))
            sampled_data = sampled_data[indexes]
            sampled_label = sampled_label[indexes]

            total_loss = total = 0
            total_ce_loss = total_rb_loss = total_entropy_loss = 0
            total_acc = 0

            for i in range(0, len(sampled_data), batch_size):
                x = sampled_data[i : i + batch_size]
                y = sampled_label[i : i + batch_size]

                logits = classifier(x)

                if logit_norm is not None:
                    batch_size = logits.size(0)
                    num_tasks = self._cur_task + 1

                    # Compute per-task norms for averaging
                    task_norms = torch.zeros(
                        batch_size, num_tasks, device=logits.device
                    )

                    for task in range(num_tasks):
                        cls_indices = [
                            clz
                            for clz in self._cls_to_task_idx
                            if self._cls_to_task_idx[clz] == task
                        ]
                        if cls_indices:
                            task_logits = logits[
                                :, cls_indices
                            ]  # (batch_size, num_classes_in_task)
                            task_norms[:, task] = (
                                torch.norm(task_logits, p=2, dim=-1) + 1e-7
                            )

                    # Average norms across all tasks
                    avg_norms = (
                        task_norms.sum(dim=-1) / num_tasks
                    )  # Average across all tasks
                    avg_norms = avg_norms.unsqueeze(-1)  # (batch_size, 1)

                    normalized_logits = logits / (avg_norms + 1e-7) / logit_norm
                    loss_vec = F.cross_entropy(normalized_logits, y, reduction="none")
                else:
                    loss_vec = F.cross_entropy(logits, y, reduction="none")

                if robust_weight_base == 0 and entropy_weight == 0:
                    loss = loss_vec.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    bs = len(y)
                    total_loss += loss.item() * bs
                    total_ce_loss += loss.item() * bs
                    total_rb_loss += 0
                    total_entropy_loss += 0
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += bs
                else:
                    L_total = torch.tensor(0.0, device=x.device)  # L = Σ Li
                    total_term1 = torch.tensor(
                        0.0, device=x.device
                    )  # For logging: sum of all term1
                    total_term2 = torch.tensor(
                        0.0, device=x.device
                    )  # For logging: sum of all term2
                    total_term3 = torch.tensor(
                        0.0, device=x.device
                    )  # For logging: sum of all term3 (entropy)

                    unique_classes = torch.unique(y)
                    class_dist = torch.cdist(
                        x, self._class_means[: self._total_classes].cuda()
                    )
                    class_indices = torch.argmin(class_dist, dim=1)
                    for class_i in unique_classes:
                        label_mask = y == class_i
                        distance_mask = class_indices == class_i
                        class_mask = distance_mask & label_mask

                        class_samples = torch.where(class_mask)[0]

                        # If no samples meet the conditions, fall back to label-only (term1 only)
                        if len(class_samples) == 0:
                            # Fall back to using only label condition for term1
                            label_only_samples = torch.where(label_mask)[0]
                            if len(label_only_samples) == 0:
                                continue  # Skip if no samples with this label at all

                            label_losses = loss_vec[label_mask]
                            term1 = label_losses.mean()
                            term2 = torch.tensor(0.0).cuda()
                            term3 = torch.tensor(0.0).cuda()
                        else:
                            class_losses = loss_vec[class_mask]
                            term1 = class_losses.mean()

                            # Second term: E_{x,x'~Ni}[|ℓ(yi, ht+1(x)) - ℓ(yi, ht+1(x'))|] where x,x' ∈ Ai
                            if len(class_samples) >= 2:
                                pairwise_diffs = torch.abs(
                                    class_losses.unsqueeze(1)
                                    - class_losses.unsqueeze(0)
                                )
                                # Remove diagonal (self-comparisons)
                                mask = ~torch.eye(
                                    len(class_losses), dtype=torch.bool, device=x.device
                                )
                                pairwise_diffs = pairwise_diffs[mask]
                                term2 = pairwise_diffs.mean()
                            else:
                                term2 = torch.tensor(0.0, device=x.device)

                            # Third term: Cluster entropy minimization
                            if len(class_samples) >= 1 and entropy_weight != 0:
                                cluster_logits = logits[
                                    class_mask
                                ]  # Shape: (n_cluster_samples, n_classes)
                                cluster_probs = F.softmax(
                                    cluster_logits, dim=1
                                )  # Shape: (n_cluster_samples, n_classes)

                                # Compute entropy for each sample: -Σ p_i * log(p_i)
                                # Add small epsilon to prevent log(0)
                                cluster_entropy = -torch.sum(
                                    cluster_probs * torch.log(cluster_probs + 1e-8),
                                    dim=1,
                                )
                                term3 = (
                                    cluster_entropy.mean()
                                )  # Average entropy across cluster samples
                            else:
                                term3 = torch.tensor(0.0, device=x.device)

                        Li = term1 + robust_weight_base * term2 + entropy_weight * term3
                        L_total += Li
                        total_term1 += term1
                        total_term2 += robust_weight_base * term2
                        total_term3 += entropy_weight * term3

                    num_classes_in_batch = len(unique_classes)
                    if num_classes_in_batch > 0:
                        loss = L_total / num_classes_in_batch
                    else:
                        loss = loss_vec.mean()  # fallback

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    bs = len(y)

                    # Average the terms by number of classes to get per-sample equivalent
                    if num_classes_in_batch > 0:
                        avg_term1 = total_term1 / num_classes_in_batch
                        avg_term2 = total_term2 / num_classes_in_batch
                        avg_term3 = total_term3 / num_classes_in_batch
                        avg_loss = L_total / num_classes_in_batch
                    else:
                        avg_term1 = torch.tensor(0.0, device=x.device)
                        avg_term2 = torch.tensor(0.0, device=x.device)
                        avg_term3 = torch.tensor(0.0, device=x.device)
                        avg_loss = loss_vec.mean()

                    total_loss += avg_loss.item() * bs
                    total_ce_loss += avg_term1.item() * bs
                    total_rb_loss += avg_term2.item() * bs
                    total_entropy_loss += avg_term3.item() * bs
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += bs

            scheduler.step()

            if epoch % 5 == 4 or epoch == epochs - 1:
                logging.info(
                    f"[Alignment] Epoch {epoch+1}/{epochs}, "
                    f"Base Loss: {total_ce_loss/total:.4f}, "
                    f"Robust Term: {total_rb_loss/total:.4f}, "
                    f"Entropy Term: {total_entropy_loss/total:.4f}, "
                    f"Total Loss: {total_loss/total:.4f}, "
                    f"Accuracy: {total_acc/total:.4f}"
                )

    def local_align(self):
        self.compute_multivariate_normal()
        
        self.merge()
        
        if self._cur_task == 0:
            return
        
        for classifier_name in self._config.get("model_classifier", ["mlp"]):
            if classifier_name == "mlp":
                logging.info(f"[Alignment] Aligning MLP classifier")
                self.align(self.model.classifier)
            elif classifier_name == "nme":
                logging.info(f"[Alignment] Aligning NME classifier")
                self.align(self.nme_classifier)

    def prefix(self):
        prefix_parts = [
            str(self._config["seed"]),
            self._config["dataset_name"],
            str(self._config["dataset_num_task"]),
            self._config["model_backbone"],
            self._config["train_merge"],
            "lca",
        ]
        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        filename = f"{self.prefix()}_backbone" + (
            f"_{task}.pt" if task >= 0 else "_base.pt"
        )
        return os.path.join(CHECKPOINT_DIR, filename)

    def merged_checkpoint(self, task):
        filename = f"{self.prefix()}_merged_{task}.pt"
        return os.path.join(CHECKPOINT_DIR, filename)

    def classifier_checkpoint(self, task):
        filename = f"{self.prefix()}_head_{task}.pt"
        return os.path.join(CHECKPOINT_DIR, filename)

    def load_backbone(self, backbone_params):
        peft_params = {}
        norm_params = {}
        for name, param in backbone_params.items():
            if name.startswith("norm."):
                norm_name = name[5:]
                norm_params[norm_name] = param
            else:
                peft_params[name] = param
        self.model.backbone.load_state_dict(peft_params, strict=False)
        if norm_params:
            self.model.norm.load_state_dict(norm_params, strict=True)


DATA_TABLE = {
    # "cifar224": [(10, 10, 10)],
    # "imagenetr": [(10, 20, 20)],
    # "imageneta": [(10, 20, 20)],
    # "cub": [(10, 20, 20)],
    # "omnibenchmark": [(10, 30, 30)],
    "vtab": [(5, 10, 10)],
}

BASE_CONFIG = {
    "seed": 1993,
    "reset": True,
    "train_epochs": 10,
    "train_batch_size": 64,
    "train_base_lr": 1e-2,
    "train_weight_decay": 5e-4,
    "train_merge": "ties",
    "train_merge_coef": 1.0,
    "train_merge_topk": 100,
    "train_merge_incremental": True,
    "model_backbone": "vit_base_patch16_224_lora",
    "model_lora_r": 64,
    "model_lora_alpha": 128,
    "model_lora_dropout": 0.0,
    "model_lora_target_modules": ["qkv"],
    "model_classifier": ["mlp"],
    "train_ca": True,
    "train_ca_samples_per_cls": 512,
    "train_ca_batch_size": 128,
    "train_ca_epochs": 10,
}

def run_single_experiment(dataset_name, config_name, experiment_config):
    config = copy.deepcopy(BASE_CONFIG)

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
    
    experiment_name = f"{dataset_name}_{config_name}"
    
    try:
        logging.info(config)

        learner = Learner(config)
        learner.learn(data_manager)
        
        mlp_faa = learner._faa_mlp
        mlp_asa = learner._asa_mlp
        nme_faa = learner._faa_nme
        nme_asa = learner._asa_nme
        
        logging.info(f"[Experiment {experiment_name}]")
        logging.info(f"  Configuration: {experiment_config}")
        classifiers = config.get("model_classifier", ["mlp"])
        if "mlp" in classifiers:
            logging.info(f"  MLP - FAA: {mlp_faa:.2f}, ASA: {mlp_asa:.2f}")
        if "nme" in classifiers:
            logging.info(f"  NME - FAA: {nme_faa:.2f}, ASA: {nme_asa:.2f}")
        del learner
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"[Experiment {experiment_name}] Detailed Error:")
        logging.error(f"Exception Type: {type(e).__name__}")
        logging.error(f"Exception Message: {str(e)}")
        logging.error(f"Full Traceback:\n{error_details}")


def run_experiments():
    experiment_configs = {
        # "simple_cil": {
        #     "train_ca": False,
        # },
        # "simple_ca": {
        #     "train_ca": True,
        #     "train_ca_epochs": 10,
        #     "train_ca_lr": 1e-2,
        #     "train_ca_samples_per_cls": 512,
        #     "train_ca_batch_size": 128,
        # },
        "simple_nme": {
            "train_ca": False,
            "model_classifier": ["nme"],
        },
        "nme_lca": {
            "train_ca": True,
            "model_classifier": ["nme"],
            "train_ca_epochs": 10,
            "train_ca_lr": 1e-3,
            "train_ca_samples_per_cls": 512,
            "train_ca_batch_size": 128,
            "train_ca_robust_weight": 0.1,
            "train_ca_entropy_weight": 0.1,
            "train_ca_logit_norm": 0.1,
        }
    }
    
    for dataset_name in DATA_TABLE.keys():
        print(f"{'='*60}")
        print(f"Starting experiments for dataset: {dataset_name}")
        print(f"{'='*60}")

        logfilename = os.path.join(LOG_DIR, f"nme_{dataset_name}.log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename),
                logging.StreamHandler(sys.stdout),
            ],
            force=True
        )

        for config_name, config in experiment_configs.items():
            logging.info("=" * 80)
            logging.info(f"Starting experiment: {dataset_name} - {config_name}")
            experiment_start_time = time.time()
            run_single_experiment(dataset_name, config_name, config)
            experiment_end_time = time.time()
            logging.info(f"Experiment {dataset_name}_{config_name} time: {experiment_end_time - experiment_start_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    results = run_experiments()
    total_time = time.time() - start_time
    print(f"Total experiment time: {total_time:.2f} seconds")
