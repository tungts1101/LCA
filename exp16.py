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
    seed_worker,
    IntermediateFeatureSampler,
)
from torch.distributions import MultivariateNormal
import logging
import sys
import optuna
import copy
import random
import math

from scipy.optimize import linear_sum_assignment



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
        classifiers = self._config.get("model_classifier", ["mlp"])
        train_ca = self._config.get("train_ca", False)
        
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

        stop_at_task = self._config.get("train_stop_at_task", None)
        if stop_at_task == -1:
            stop_at_task = None
        for task in range(num_tasks):
            self.before_task(task, data_manager)
            self.train()
            self.eval()
            self.after_task()
            if stop_at_task is not None and task >= stop_at_task:
                break
        
        torch.save(
            self.model.state_dict(),
            self.model_checkpoint()
        )

    def before_task(self, task, data_manager):
        task_size = data_manager.get_task_size(task)
        self._total_classes = self._known_classes + task_size
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

        for clz in range(self._known_classes, self._total_classes):
            self._cls_to_task_idx[clz] = self._cur_task

        if task > 0:
            merged = self.merged_checkpoint(task - 1)
            if os.path.exists(merged):
                model_use_norm = self._config.get("model_use_norm", False)
                logging.info(f"[Before Task {task}] Loading merged checkpoint from task {task - 1}")
                self.load_backbone(torch.load(merged), load_norm=model_use_norm)

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
        self.train_mlp(train_loader)
        
        train_merge = self._config.get("train_merge", "none")
        if train_merge != "none":
            self.merge()
        
        if self._config.get("train_reg_weight", 0.2) > 0:
            self.extract_prototypes()

        # train_merge = self._config.get("train_merge", "none")
        # if train_merge != "none":
        #     self.merge()
        
        train_ca = self._config.get("train_ca", False)
        if train_ca:
            self.compute_multivariate_normal()
            self.align(self.model.classifier)

        if "ncm" in classifiers:
            self.update_ncm_classifier()
            self.train_prototype(prototype_loader)
    
    def extract_prototypes(self):
        logging.info(f"[Training] Extracting intermediate features for task {self._cur_task}")
        L = self._config.get("train_feature_at_layer", -1)

        if not hasattr(self, "_sampler"):
            self._sampler = IntermediateFeatureSampler(
                total_classes=self._total_classes,
                token_length=197,
                feature_dim=self.model.feature_dim,
            )
        else:
            self._sampler.expand_to(self._total_classes)

        for cls_idx in range(self._known_classes, self._total_classes):
            train_set = self.data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            train_loader = DataLoader(
                train_set, batch_size=512, shuffle=False,
                num_workers=4, worker_init_fn=seed_worker, generator=g
            )

            layer_feats = {L: [], "final": []}

            self.model.eval()
            with torch.no_grad():
                for _, (_, _, x, y) in enumerate(train_loader):
                    x = x.cuda()
                    z = self.model.get_features(x, return_layer_features=True)  # (B, D)
                    layer_feats[L].append(self.model.layer_features[L].cpu())
                    layer_feats["final"].append(z.cpu())

            for k in (L, "final"):
                feats = torch.cat(layer_feats[k], dim=0)
                self._sampler.update(feats, cls_idx, k)

        # Release last batch's intermediate tensors captured by hooks
        self.model.layer_features = []
        torch.cuda.empty_cache()

    def train_mlp(self, train_loader):
        logging.info(f"[Training] Task {self._cur_task}")

        model_use_norm = self._config.get("model_use_norm", False)
        model_classifier_use_norm = not model_use_norm
        model_classifier_norm_layer = self._config.get("model_classifier_norm_layer", "ln")

        self.model.update_classifier(
            self._total_classes - self._known_classes, 
            with_norm=model_classifier_use_norm, 
            with_bias=False, freeze_old=True, 
            norm_layer=model_classifier_norm_layer
        )
        self.model.cuda()

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
        
        lam = self._config.get("train_reg_weight", 0.2)
        lam2 = self._config.get("train_reg_weight_intra_class", 0.0)
        L = self._config.get("train_feature_at_layer", -1)
        K = self._config.get("train_reg_num_sampling", 4)
        N = self._config.get("train_reg_num_classes", 5)
        train_reg_loss = self._config.get("train_reg_loss", "mse")
        train_reg_mag_weight = self._config.get("train_reg_mag_weight", 0.1)
        use_reg = lam > 0 and self._cur_task > 0 and hasattr(self, "_sampler")


        train_minibatch_uniform = self._config.get("train_minibatch_uniform", False)
        train_minibatch_num_cls = self._config.get("train_minibatch_num_cls", 2)

        if train_minibatch_uniform:
            num_new_cls = self._total_classes - self._known_classes
            n_cls_per_batch = min(train_minibatch_num_cls, num_new_cls)
            samples_per_cls_in_batch = max(1, self._config["train_batch_size"] // n_cls_per_batch)

            all_x_list, all_y_list = [], []
            for _, (_, _, xb, yb) in enumerate(train_loader):
                all_x_list.append(xb)
                all_y_list.append(yb)
            all_x_cpu = torch.cat(all_x_list, dim=0)
            all_y_cpu = torch.cat(all_y_list, dim=0)

            cls_data_mlp = []
            for c in range(num_new_cls):
                mask = all_y_cpu == (c + self._known_classes)
                cls_data_mlp.append(all_x_cpu[mask])

            total_n_mlp = sum(len(d) for d in cls_data_mlp)
            num_batches_mlp = max(1, total_n_mlp // self._config["train_batch_size"])

        train_reg_at_each_n_batch = self._config.get("train_reg_at_each_n_batch", 1)

        for epoch in range(epochs):
            total_loss, total_ce_loss, total_reg_loss, total_intra_loss, total_acc, total = 0, 0, 0, 0, 0, 0

            if train_minibatch_uniform:
                cls_shuffled_mlp = [d[torch.randperm(len(d))] for d in cls_data_mlp]
                sample_ptrs_mlp = [0] * num_new_cls
                cls_order_mlp = torch.randperm(num_new_cls).tolist()

                def _mlp_batch_iter():
                    classes_seen = set()
                    for bi in range(num_batches_mlp):
                        cls_start = (bi * n_cls_per_batch) % num_new_cls
                        selected = [cls_order_mlp[(cls_start + j) % num_new_cls] for j in range(n_cls_per_batch)]
                        xs, ys = [], []
                        for c in selected:
                            sz = len(cls_shuffled_mlp[c])
                            p = sample_ptrs_mlp[c]
                            idxs = torch.arange(p, p + samples_per_cls_in_batch) % sz
                            chunk = cls_shuffled_mlp[c][idxs]
                            sample_ptrs_mlp[c] = (p + samples_per_cls_in_batch) % sz
                            xs.append(chunk)
                            ys.append(torch.full((samples_per_cls_in_batch,), c, dtype=torch.long))
                        classes_seen.update(selected)
                        # logging.info(
                        #     f"[Training/UniformMB] epoch={epoch+1} batch={bi+1}/{num_batches_mlp} "
                        #     f"classes={[c + self._known_classes for c in selected]} "
                        #     f"samples_per_cls={samples_per_cls_in_batch} batch_size={len(selected)*samples_per_cls_in_batch}"
                        # )
                        yield torch.cat(xs, 0), torch.cat(ys, 0)
                    # logging.info(
                    #     f"[Training/UniformMB] epoch={epoch+1} covered {len(classes_seen)}/{num_new_cls} classes "
                    #     f"over {num_batches_mlp} batches ({n_cls_per_batch} cls/batch, {samples_per_cls_in_batch} samples/cls)"
                    # )

                batch_iter = tqdm(_mlp_batch_iter(), total=num_batches_mlp, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            else:
                def _std_batch_iter():
                    for _, (_, _, xb, yb) in enumerate(train_loader):
                        yb = torch.where(yb - self._known_classes >= 0, yb - self._known_classes, -100)
                        yield xb, yb
                batch_iter = tqdm(_std_batch_iter(), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_num, (x, y) in enumerate(batch_iter):
                x, y = x.cuda(), y.cuda()

                z = self.model.get_features(x)
                logits = self.model.classifier.heads[-1](z)
                ce_loss = F.cross_entropy(logits, y, ignore_index=-100)
                loss = ce_loss

                if use_reg and (batch_num % train_reg_at_each_n_batch == 0):
                    n_old = self._known_classes
                    reg_loss = torch.tensor(0.0, device="cuda")
                    classes = torch.randperm(n_old)[:N].tolist()
                    chunk = torch.cat([self._sampler.sample(c, L, K) for c in classes], dim=0).cuda(non_blocking=True)  # (N*K, T, D)
                    proj = self.model.forward_from_block(chunk, L + 1)

                    if train_reg_loss == "log_likelihood":
                        means = torch.cat([self._sampler.get_cls_mean(c, "final").unsqueeze(0).expand(K, -1) for c in classes], dim=0).cuda(non_blocking=True)  # (N*K, D)
                        var = torch.cat([self._sampler.get_sigma(c, "final").clamp(min=1e-3).pow(2).unsqueeze(0).expand(K, -1) for c in classes], dim=0).cuda(non_blocking=True)  # (N*K, D)
                        reg_loss = reg_loss + F.gaussian_nll_loss(proj, means, var)
                        if lam2 > 0:
                            proj_by_class = proj.view(len(classes), K, -1)  # (N, K, D)
                            diff = proj_by_class.unsqueeze(2) - proj_by_class.unsqueeze(1)  # (N, K, K, D)
                            mse_class = diff.pow(2).mean()
                            reg_loss = reg_loss + lam2 * mse_class
                            total_intra_loss += mse_class.item() * len(y)
                    else:
                        chunk_target = torch.cat([self._sampler.sample(c, "final", K).squeeze(1) for c in classes], dim=0).cuda(non_blocking=True)  # (N*K, D)
                        if train_reg_loss == "l1":
                            reg_loss = reg_loss + F.l1_loss(proj, chunk_target)
                        elif train_reg_loss == "smooth_l1":
                            reg_loss = reg_loss + F.smooth_l1_loss(proj, chunk_target)
                        elif train_reg_loss == "cosine":
                            reg_loss = reg_loss + (1 - F.cosine_similarity(proj, chunk_target, dim=-1)).mean()
                        elif train_reg_loss == "normalized_smooth_l1":
                            proj_n = F.normalize(proj, dim=-1)
                            tgt_n  = F.normalize(chunk_target, dim=-1)
                            reg_loss = reg_loss + F.smooth_l1_loss(proj_n, tgt_n)
                        elif train_reg_loss == "normalized_l2":
                            proj_n = F.normalize(proj, dim=-1)
                            tgt_n  = F.normalize(chunk_target, dim=-1)
                            reg_loss = reg_loss + F.mse_loss(proj_n, tgt_n)
                        elif train_reg_loss == "cosine_magnitude":
                            cos = (1 - F.cosine_similarity(proj, chunk_target, dim=-1)).mean()
                            mag = F.mse_loss(proj.norm(dim=-1), chunk_target.norm(dim=-1))
                            reg_loss = reg_loss + cos + train_reg_mag_weight * mag
                        else:
                            reg_loss = reg_loss + F.mse_loss(proj, chunk_target)

                    loss = loss + lam * reg_loss
                    total_reg_loss += reg_loss.item() * len(y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                total_ce_loss += ce_loss.item() * len(y)
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

            scheduler.step()
            # if epoch % 5 == 4 or epoch == epochs - 1:
            logging.info(
                f"[Training] Epoch {epoch + 1}/{epochs}, "
                f"Total Loss: {total_loss / total:.4f}, "
                f"CE Loss: {total_ce_loss / total:.4f}, "
                f"Reg Loss: {total_reg_loss / total:.4f}, "
                f"Intra Loss: {total_intra_loss / total:.4f}, "
                f"Acc: {total_acc / total:.4f}"
            )

        torch.save(
            self.model.get_backbone_trainable_params(), self.backbone_checkpoint(self._cur_task)
        )

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

            model_use_norm = self._config.get("model_use_norm", False)    
            self.load_backbone(backbone_params, load_norm=model_use_norm)
        
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

    # def align(self, classifier):
    #     logging.info(f"[Alignment] Task {self._cur_task}")

    #     samples_per_cls = self._config.get("train_ca_samples_per_cls", 256)
    #     mc_z_per_class = self._config.get("train_ca_mc_z_per_class", 64)
    #     epochs = self._config.get("train_ca_epochs", 10)
    #     batch_size = self._config.get("train_ca_batch_size", 64)
    #     robust_weight = self._config.get("train_ca_robust_weight", 0.0)
    #     align_minibatch_uniform = self._config.get("align_minibatch_uniform", False)
    #     align_minibatch_num_cls = self._config.get("align_minibatch_num_cls", 2)

    #     device = next(classifier.parameters()).device

    #     for p in classifier.parameters():
    #         p.requires_grad = True
        
    #     num_trainable = count_parameters(classifier, trainable=True)
    #     logging.info(f"[Alignment] Num trainable parameters: {num_trainable:,}")

    #     optimizer = optim.SGD(
    #         classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
    #     )

    #     # ------------------------------------------------------------------
    #     # Build a fixed synthetic dataset S.
    #     # Region/class i defines Z_i through the fixed Gaussian N(mu_i, Sigma_i).
    #     # We sample S_i from each region once and keep them fixed across training.
    #     # This matches the theorem much better than redefining regions on the fly.
    #     # ------------------------------------------------------------------
    #     sampled_data = []
    #     sampled_label = []

    #     dists = []
    #     for cls_idx in range(self._total_classes):
    #         mu_i = self._class_means[cls_idx].to(device).float()
    #         cov_i = self._class_covs[cls_idx].to(device).float()
    #         dist_i = MultivariateNormal(mu_i, cov_i)
    #         dists.append(dist_i)

    #         s_i = dist_i.sample((samples_per_cls,))  # S_i
    #         sampled_data.append(s_i)
    #         sampled_label.append(
    #             torch.full((samples_per_cls,), cls_idx, dtype=torch.long, device=device)
    #         )

    #     sampled_data = torch.cat(sampled_data, dim=0)   # (n, D)
    #     sampled_label = torch.cat(sampled_label, dim=0) # (n,)

    #     # # Fixed partition anchors if you still want Voronoi regions.
    #     # # But note: if Z_i is defined by N(mu_i, Sigma_i), the cleanest choice is:
    #     # # S_i = samples drawn from dist_i, without extra Voronoi reassignment.
    #     # partition_means = self._class_means[: self._total_classes].clone().detach().to(device).float()

    #     n_total = sampled_data.size(0)

    #     if align_minibatch_uniform:
    #         n_cls_align = self._total_classes
    #         n_cls_per_batch_align = min(align_minibatch_num_cls, n_cls_align)
    #         spc_align = max(1, batch_size // n_cls_per_batch_align)
    #         cls_data_align = []
    #         for c in range(n_cls_align):
    #             mask = sampled_label == c
    #             cls_data_align.append(sampled_data[mask])
    #         num_batches_align = max(1, n_total // batch_size)

    #     for epoch in range(epochs):
    #         total_loss = 0.0
    #         total_ce_loss = 0.0
    #         total_rb_loss = 0.0
    #         total_acc = 0
    #         total = 0

    #         if align_minibatch_uniform:
    #             cls_shuffled_align = [d[torch.randperm(len(d), device=device)] for d in cls_data_align]
    #             sample_ptrs_align = [0] * n_cls_align
    #             num_iterations = num_batches_align
    #         else:
    #             # Shuffle S each epoch for SGD on F(S,h)
    #             perm = torch.randperm(n_total, device=device)
    #             epoch_data = sampled_data[perm]
    #             epoch_label = sampled_label[perm]
    #             num_iterations = (n_total + batch_size - 1) // batch_size

    #         for it in range(num_iterations):
    #             if align_minibatch_uniform:
    #                 cls_start = (it * n_cls_per_batch_align) % n_cls_align
    #                 selected = [(cls_start + j) % n_cls_align for j in range(n_cls_per_batch_align)]
    #                 xs, ys = [], []
    #                 for c in selected:
    #                     sz = len(cls_shuffled_align[c])
    #                     p = sample_ptrs_align[c]
    #                     idxs = torch.arange(p, p + spc_align) % sz
    #                     chunk = cls_shuffled_align[c][idxs]
    #                     sample_ptrs_align[c] = (p + spc_align) % sz
    #                     xs.append(chunk)
    #                     ys.append(torch.full((spc_align,), c, dtype=torch.long, device=device))
    #                 x = torch.cat(xs, 0)
    #                 y = torch.cat(ys, 0)
    #             else:
    #                 start = it * batch_size
    #                 end = min((it + 1) * batch_size, n_total)
    #                 x = epoch_data[start:end]
    #                 y = epoch_label[start:end]
    #             n_batch = y.size(0)

    #             logits = classifier(x)
    #             loss_vec = F.cross_entropy(logits, y, reduction="none")
    #             base_loss = loss_vec.mean()

    #             if torch.isnan(base_loss):
    #                 continue

    #             reg_loss = torch.tensor(0.0, device=device)

    #             if robust_weight > 0:
    #                 # ----------------------------------------------------------
    #                 # Estimate:
    #                 #   sum_i (n_i / n) * bar_epsilon_i(h)
    #                 #
    #                 # where
    #                 #   bar_epsilon_i(h)
    #                 #   = (1 / n_i) sum_{s in S_i} E_{z ~ Z_i} |l(h,z) - l(h,s)|
    #                 #
    #                 # We do this by:
    #                 #   1) taking the current batch samples x as candidate s
    #                 #   2) grouping them by region i
    #                 #   3) drawing fresh Monte Carlo samples z ~ Z_i
    #                 #   4) computing all pairwise |l(z) - l(s)| for that region
    #                 #
    #                 # This is faithful to Theorem 5.
    #                 # ----------------------------------------------------------

    #                 unique_classes = torch.unique(y)

    #                 for class_i in unique_classes.tolist():
    #                     s_mask = (y == class_i)
    #                     x_i = x[s_mask]
    #                     y_i = y[s_mask]

    #                     if x_i.size(0) == 0:
    #                         continue

    #                     # # Optional: strict geometric membership under a fixed partition.
    #                     # # If you want theorem-faithful regions and your Z_i is "Gaussian i",
    #                     # # skip this block and use all x_i directly.
                        
    #                     # with torch.no_grad():
    #                     #     dist_to_centers = torch.cdist(x_i, partition_means)
    #                     #     vor_idx = torch.argmin(dist_to_centers, dim=1)
    #                     # keep = (vor_idx == class_i)
    #                     # x_i = x_i[keep]
    #                     # y_i = y_i[keep]
    #                     # if x_i.size(0) == 0:
    #                     #     continue

    #                     # Losses l(h,s), s in S_i
    #                     logits_s = classifier(x_i)
    #                     loss_s = F.cross_entropy(logits_s, y_i, reduction="none")  # (n_i_batch,)

    #                     # Monte Carlo samples z ~ Z_i
    #                     z_i = dists[class_i].sample((mc_z_per_class,))  # (m, D)
    #                     y_z = torch.full(
    #                         (mc_z_per_class,), class_i, dtype=torch.long, device=device
    #                     )

    #                     logits_z = classifier(z_i)
    #                     loss_z = F.cross_entropy(logits_z, y_z, reduction="none")  # (m,)

    #                     # Pairwise absolute differences:
    #                     #   (1/n_i) sum_s (1/m) sum_z |l(z) - l(s)|
    #                     pairwise_abs = torch.abs(loss_z[:, None] - loss_s[None, :])  # (m, n_i_batch)
    #                     bar_eps_i_hat = pairwise_abs.mean()

    #                     # Use the actual effective weight from the current S.
    #                     # Since S was built with equal samples_per_cls per class,
    #                     # n_i / n is exactly 1 / total_classes if you do NOT
    #                     # discard samples afterwards.
    #                     region_weight = float(samples_per_cls) / float(n_total)

    #                     reg_loss = reg_loss + region_weight * bar_eps_i_hat

    #             loss = base_loss + robust_weight * reg_loss

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item() * n_batch
    #             total_ce_loss += base_loss.item() * n_batch
    #             total_rb_loss += reg_loss.item() * n_batch
    #             total_acc += (logits.argmax(dim=1) == y).sum().item()
    #             total += n_batch

    #         if epoch % 5 == 4 or epoch == epochs - 1:
    #             logging.info(
    #                 f"[Alignment] Epoch {epoch + 1}/{epochs}, "
    #                 f"Base Loss: {total_ce_loss / max(total, 1):.4f}, "
    #                 f"Robust Term: {total_rb_loss / max(total, 1):.4f}, "
    #                 f"Total Loss: {total_loss / max(total, 1):.4f}, "
    #                 f"Accuracy: {total_acc / max(total, 1):.4f}"
    #             )

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

    def model_checkpoint(self):
        filename = f"{self.prefix()}_model.pt"
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
    # "imageneta": [(10, 20, 20)],
    # "cub": [(10, 20, 20)],
    # "omnibenchmark": [(10, 30, 30)],
    # "vtab": [(5, 10, 10)],
    "cars": [(10, 16, 20)]
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
    seeds = [1993, 1994, 1995]

    experiment_configs = {
        "exp16": {
            "reset_train": True,
            "reset_merge": True,
            "train_epochs": 10,
            "train_batch_size": 64,
            "model_backbone": "vit_base_patch16_224_lora",
            "model_outdim": 768,
            "model_use_norm": True,
            "model_lora_r": 64,
            "model_lora_alpha": 128,
            "model_classifier_norm_layer": "ln", # "ln" | "bn"
            "ffn_num": 64,
            "model_lora_dropout": 0.0,
            "model_lora_target_modules": ["qkv"],
            "model_classifier": ["mlp"],

            "train_prefix": "exp16",
            "train_stop_at_task": 4,

            "train_minibatch_uniform": False,
            "train_minibatch_num_cls": 4,
            "align_minibatch_uniform": False,
            "align_minibatch_num_cls": 4,

            "train_feature_at_layer": 8,    # L
            "train_reg_weight": 1e-2,        # lambda
            "train_reg_num_classes": 5,     # N
            "train_reg_num_sampling": 16,   # K

            "train_reg_loss": "log_likelihood", # "mse" | "l1" | "smooth_l1" | "cosine" | "normalized_smooth_l1" | "normalized_l2" | "cosine_magnitude"
            
            "train_reg_mag_weight": 1.0,  # weight for magnitude term in cosine_magnitude loss
            "train_reg_at_each_n_batch": 1,

            "train_RP": False,

            "train_merge": "ties",
            "train_merge_coef": 1.0,
            "train_merge_topk": 100,
            "train_merge_incremental": True,

            "train_ca": True,
            "train_ca_load_checkpoint_from_first_task": False,
            "train_ca_samples_per_cls": 512,
            "train_ca_batch_size": 64,
            "train_ca_epochs": 3,
            "train_ca_robust_weight": 1.0,

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
