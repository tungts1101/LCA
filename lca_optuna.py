import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from utils.data_manager import DataManager
from helper import (
    Model,
    compute_metrics,
    count_parameters,
    accuracy,
    set_random,
    merge,
)
import gc
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse
import traceback
from torch.distributions.multivariate_normal import MultivariateNormal
import logging
import sys


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

OPTUNA_DIR = "optuna"
os.makedirs(OPTUNA_DIR, exist_ok=True)


DATA_TABLE = {
    "cifar224": {
        "dataset_num_task": 10,
        "dataset_init_cls": 10,
        "dataset_increment": 10,
    },
    "imagenetr": {
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,
    },
    "imageneta": {
        "dataset_num_task": 10,
        "dataset_init_cls": 20,
        "dataset_increment": 20,
    },
    "cub": {"dataset_num_task": 10, "dataset_init_cls": 20, "dataset_increment": 20},
    "omnibenchmark": {
        "dataset_num_task": 10,
        "dataset_init_cls": 30,
        "dataset_increment": 30,
    },
    "vtab": {"dataset_num_task": 5, "dataset_init_cls": 10, "dataset_increment": 10},
}

# in-r
# 94, 92, 90, 89, 87, 86, 85, 84, 84, 83 (83.52)
# 94, 92, 91, 89, 88, 87, 86, 85, 84, 83 (83.69)

# in-a
# 87, 84, 81, 78, 75, 72, 70, 69, 67, 66 (66.52)
# 87, 86, 84, 82, 80,

# simple-cil, rank 64, incrementally merge
# in-r
# 94.63, 92.47, 91.05, 89.75, 88.48, 87.55, 86.82, 86.19, 85.60, 85.03
# in-a
# 87.43, 84.56, 81.13, 78.00, 75.16, 72.74, 70.74, 69.16, 67.73, 66.52

# base alignment, rank 64, incrementally merge
# in-r
# 94.63, 92.58, 91.31, 90.11, 88.91, 88.04, 87.31, 86.66, 86.08, 85.52
# in-a
# 87.43, 86.20, 84.11, 82.12, 80.25, 78.59, 77.22, 76.10, 74.82, 73.76

EARLY_PRUNING_THRESHOLDS = {
    "imagenetr": {2: 92, 4: 90, 6: 88, 8: 86},
    "imageneta": {2: 86, 4: 82, 6: 78, 8: 76},
}


class Learner:
    def __init__(self, config, trial):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self._acc_matrix = []
        self._cls_to_task_idx = {}
        self.trial = trial

        self.model = Model(config)
        self.model.cuda()
        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint()
        )

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

            dataset_name = self._config["dataset_name"]
            if dataset_name in EARLY_PRUNING_THRESHOLDS:
                thresholds = EARLY_PRUNING_THRESHOLDS[dataset_name]
                if (task + 1) in thresholds:
                    threshold = thresholds[task + 1]
                    if self._asa < threshold:
                        logging.info(
                            f"[Early Stopping] ASA {self._asa:.2f} < {threshold:.2f} at task {task + 1}"
                        )
                        if self.trial is not None:
                            logging.info(
                                f"[Pruning] Trial pruned due to poor performance at task {task + 1}"
                            )
                            raise optuna.TrialPruned()

        if self.trial is not None:
            self.trial.report(self._asa, self.trial.number)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

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

        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                logits = self.model(x)
                predicts = logits.argmax(dim=1)
                y_pred.append(predicts.cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc_total, grouped = accuracy(y_pred.T, y_true, self._class_increments)

        logging.info(f"[Evaluation] Task {self._cur_task + 1}")
        logging.info(f"[Evaluation] Total Acc: {acc_total:.2f}, Grouped Acc: {grouped}")

        self._acc_matrix.append(grouped)

        num_tasks = len(self._acc_matrix)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                accuracy_matrix[i, j] = self._acc_matrix[i][j]

        faa, ffm, ffd, asa = compute_metrics(accuracy_matrix)
        self._faa = faa
        self._asa = asa
        self._grouped = grouped
        logging.info(
            f"[Evaluation] FAA: {faa:.2f}, FFM: {ffm:.2f}, FFD: {ffd:.2f}, ASA: {asa:.2f}"
        )

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

            logging.info(f"[Training] Task {self._cur_task + 1}")
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
            logging.info(f"[Training] Load checkpoint for task {self._cur_task + 1}")
            backbone_params = torch.load(self.backbone_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)
            self.model.classifier.heads[self._cur_task].load_state_dict(
                torch.load(self.classifier_checkpoint(self._cur_task)), strict=True
            )

        if self._config.get("train_ca", False):
            self.local_align()
        else:
            self.merge()

    def merge(self):
        if self._cur_task == 0:
            logging.info(
                f"[Merging] Save merged backbone checkpoint for task {self._cur_task + 1}"
            )
            torch.save(
                self.model.get_backbone_trainable_params(),
                self.merged_checkpoint(self._cur_task),
            )
            return
        if os.path.exists(self.merged_checkpoint(self._cur_task)):
            logging.info(
                f"[Merging] Load merged checkpoint for task {self._cur_task + 1}"
            )
            backbone_params = torch.load(self.merged_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)
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
            f"[Merging] Save merged backbone checkpoint for task {self._cur_task + 1}"
        )
        torch.save(
            self.model.get_backbone_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )

    def local_align(self):
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

        self.merge()

        if self._cur_task == 0:
            return

        for p in self.model.classifier.parameters():
            p.requires_grad = True
        num_trainable = count_parameters(self.model.classifier, trainable=True)
        logging.info(f"[Alignment] Num trainable parameters: {num_trainable:,}")

        epochs = self._config.get("train_ca_epochs", 10)
        samples_per_cls = self._config.get("train_ca_samples_per_cls", 256)
        batch_size = self._config.get("train_ca_batch_size", 64)
        base_lr = self._config.get("train_ca_lr", 1e-2)

        robust_weight_base = self._config.get("train_ca_robust_weight", 0.0)
        entropy_weight = self._config.get("train_ca_entropy_weight", 0.0)
        logit_norm = self._config.get(
            "train_ca_logit_norm", None
        )  # None means no logit norm

        # param_groups = []
        # for task_id in range(self._cur_task + 1):
        #     task_params = [p for p in self.model.classifier.heads[task_id].parameters() if p.requires_grad]
        #     if task_params:
        #         task_age = self._cur_task - task_id  # 0 for current task, increases for older tasks
        #         task_lr_multiplier = 0.5 ** task_age
        #         task_lr = base_lr * task_lr_multiplier

        #         param_groups.append({
        #             "params": task_params,
        #             "lr": task_lr,
        #             "weight_decay": self._config.get("train_ca_weight_decay", 5e-4)
        #         })
        # optimizer = optim.SGD(
        #     param_groups, lr=base_lr, weight_decay=5e-4, momentum=0.9
        # )

        optimizer = optim.SGD(
            self.model.classifier.parameters(),
            lr=base_lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

        # Sample data from Gaussian distributions
        sampled_data, sampled_label = [], []
        for cls_idx in range(total_class):
            cls_mean = self._class_means[cls_idx].cuda()
            cls_cov = self._class_covs[cls_idx].cuda()

            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_features = m.sample((samples_per_cls,))

            sampled_data.append(sampled_features)
            sampled_label.extend([cls_idx] * samples_per_cls)

        sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
        sampled_label = torch.tensor(sampled_label).long().cuda()

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

                logits = self.model.classifier(x)

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


# Base configuration for Optuna optimization (dataset-independent)
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
    "train_ca": True,
    "train_ca_samples_per_cls": 512,
    "train_ca_batch_size": 128,
    "train_ca_epochs": 10,
    "train_ca_logit_norm": 0.1,  # None means no logit norm
}


def suggest_hyperparameters(trial):
    # ca_lr = trial.suggest_categorical("train_ca_lr", [1e-4, 1e-3, 1e-2])
    ca_lr = trial.suggest_float("train_ca_lr", 1e-4, 1e-2)

    # robust_weight_log = trial.suggest_categorical("robust_weight_log", [-3, -2, -1, 0, 1, 2, 3])
    robust_weight_log = trial.suggest_float("robust_weight_log", -3, 3)
    robust_weight = 10**robust_weight_log

    # entropy_weight_log = trial.suggest_categorical("entropy_weight_log", [-2, -1, 0, 1, 2])
    entropy_weight_log = trial.suggest_float("entropy_weight_log", -2, 2)
    entropy_weight = 10**entropy_weight_log

    ca_lr = round(ca_lr, 5)
    robust_weight = round(robust_weight, 5)
    entropy_weight = round(entropy_weight, 5)

    return {
        "train_ca_lr": ca_lr,
        "train_ca_rb_weight": robust_weight,
        "train_ca_entropy_weight": entropy_weight,
    }


def objective(trial, dataset_name):
    trial_start_time = time.time()
    try:
        hyperparams = suggest_hyperparameters(trial)
        config = BASE_CONFIG.copy()
        config.update(hyperparams)

        if dataset_name not in DATA_TABLE:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Available: {list(DATA_TABLE.keys())}"
            )

        dataset_config = DATA_TABLE[dataset_name]
        config["dataset_name"] = dataset_name
        config.update(
            {
                "dataset_num_task": dataset_config["dataset_num_task"],
                "dataset_init_cls": dataset_config["dataset_init_cls"],
                "dataset_increment": dataset_config["dataset_increment"],
            }
        )

        trial_id = trial.number
        logging.info(
            f"\n[Trial {trial_id + 1}] Starting optimization with hyperparameters: {hyperparams}"
        )

        data_manager = DataManager(
            config["dataset_name"],
            True,
            config["seed"],
            config["dataset_init_cls"],
            config["dataset_increment"],
            False,
        )

        set_random(config["seed"])

        learner = Learner(config, trial=trial)
        learner.learn(data_manager)

        asa_score = learner._asa

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.info(
            f"[Trial {trial_id + 1}] ASA Score: {asa_score:.2f}, Duration: {trial_duration:.2f}s"
        )

        del learner
        torch.cuda.empty_cache()
        gc.collect()

        return asa_score

    except optuna.TrialPruned:
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.info(
            f"[Trial {trial.number + 1}] Trial was pruned, Duration: {trial_duration:.2f}s"
        )

        torch.cuda.empty_cache()
        gc.collect()

        raise

    except Exception as e:
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.error(f"[Trial {trial.number + 1}] Error during optimization: {str(e)}")
        logging.error(
            f"[Trial {trial.number + 1}] Duration before error: {trial_duration:.2f}s"
        )
        logging.error(f"Error details: {traceback.format_exc()}")

        torch.cuda.empty_cache()
        gc.collect()

        return 0.0


def run_optuna_optimization(
    dataset_name, n_trials=150, early_stop_patience=None, max_time_hours=None
):
    logfilename = os.path.join(LOG_DIR, f"optuna_{dataset_name}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    optimization_start_time = time.time()
    logging.info(
        f"Starting Optuna optimization with {n_trials} trials for {dataset_name}"
    )

    if early_stop_patience is not None:
        logging.info(
            f"Early stopping enabled: patience = {early_stop_patience} trials without improvement"
        )
    if max_time_hours is not None:
        logging.info(f"Time limit enabled: maximum {max_time_hours} hours")

    sampler = TPESampler(seed=BASE_CONFIG["seed"])
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study_name = f"optuna_{dataset_name}"
    storage_name = os.path.join(OPTUNA_DIR, f"{study_name}.db")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    logging.info(f"Study created: {study_name}")
    logging.info(f"Storage: {storage_name}")

    best_value = -float("inf")
    min_delta = 0.01
    no_improvement_trials = 0

    def early_stopping_callback(study, trial):
        nonlocal best_value, no_improvement_trials, min_delta
        if trial is not None and trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value is not None and trial.value - min_delta > best_value:
                best_value = trial.value
                no_improvement_trials = 0
                logging.info(
                    f"New best value: {best_value:.2f} at trial {trial.number}"
                )
            else:
                no_improvement_trials += 1
                logging.info(
                    f"No improvement in trial {trial.number}. Count: {no_improvement_trials}/{early_stop_patience}"
                )
                if (
                    early_stop_patience is not None
                    and no_improvement_trials >= early_stop_patience
                ):
                    logging.info(
                        f"Early stopping: No improvement in the last {early_stop_patience} trials."
                    )
                    study.stop()

        if max_time_hours is not None:
            elapsed_time = time.time() - optimization_start_time
            elapsed_hours = elapsed_time / 3600
            if elapsed_hours >= max_time_hours:
                logging.info(
                    f"Early stopping: Time limit reached! Elapsed: {elapsed_hours:.2f}/{max_time_hours} hours"
                )
                study.stop()
                return

    try:
        callbacks = [early_stopping_callback]
        study.optimize(
            lambda trial: objective(trial, dataset_name),
            n_trials=n_trials,
            callbacks=callbacks,
        )

        optimization_end_time = time.time()
        total_optimization_time = optimization_end_time - optimization_start_time

        logging.info(f"\nOptimization completed!")
        logging.info(
            f"Total optimization time: {total_optimization_time:.2f} seconds ({total_optimization_time/3600:.2f} hours)"
        )

        completed_successful_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_successful_trials) > 0:
            logging.info(
                f"Successfully completed trials: {len(completed_successful_trials)}"
            )
            logging.info(f"Best ASA score: {study.best_value:.2f}")
            logging.info(f"Best hyperparameters:")
            for key, value in study.best_params.items():
                logging.info(f"  {key}: {value}")
        else:
            logging.warning("No trials completed successfully!")

    except KeyboardInterrupt:
        optimization_end_time = time.time()
        total_optimization_time = optimization_end_time - optimization_start_time

        logging.info("Optimization interrupted by user")
        logging.info(
            f"Total time before interruption: {total_optimization_time:.2f} seconds ({total_optimization_time/3600:.2f} hours)"
        )

        completed_successful_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_successful_trials) > 0:
            logging.info(f"Current best ASA score: {study.best_value:.2f}")
        else:
            logging.warning("No trials completed successfully before interruption!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCA Optuna Optimization")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenetr",
        choices=list(DATA_TABLE.keys()),
        help=f"Dataset to optimize on. Available: {list(DATA_TABLE.keys())}",
    )
    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Stop whole study after N trials without improvement",
    )
    parser.add_argument(
        "--max_time_hours", type=float, default=24, help="Stop early after N hours"
    )

    args = parser.parse_args()

    run_optuna_optimization(
        dataset_name=args.dataset,
        n_trials=args.n_trials,
        early_stop_patience=args.early_stop_patience,
        max_time_hours=args.max_time_hours,
    )
