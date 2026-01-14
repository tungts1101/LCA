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
    count_parameters
)
import logging
import sys
import copy
import math
from collections import defaultdict

import timm
from peft import get_peft_model, LoraConfig
from typing import Dict, List, Tuple, Optional, Set


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

EPSILON = 1e-8

def get_backbone(config):
    # timm/vit_base_patch16_dinov3.lvd1689m
    # timm/vit_large_patch16_dinov3.lvd1689m
    # timm/vit_huge_plus_patch16_dinov3.lvd1689m
    # timm/vit_base_patch16_clip_224.openai
    # timm/vit_base_patch16_224.mae
    model = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
    model.requires_grad_(False)
    model.out_dim = 1280
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    return model    

# ============================================================
# Max-abs merge + mask (for stats)
# ============================================================
def maxabs_merge_with_mask(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    base: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Elementwise max-abs merge with LEGACY tie-breaking.

    Rules:
      1) If |x1| > |x2|  -> choose x1
      2) If |x2| > |x1|  -> choose x2
      3) If |x1| == |x2| -> compare |x1 - base| vs |x2 - base|
      4) If still tie   -> choose x1 (legacy default)

    Returns:
      merged tensor
      mask where True means x1 was selected
    """
    # abs1 = x1.abs()
    # abs2 = x2.abs()

    # # primary decision
    # take1 = abs1 > abs2
    # take2 = abs2 > abs1
    # tie = ~(take1 | take2)

    # if base is not None and tie.any():
    #     d1 = (x1 - base).abs()
    #     d2 = (x2 - base).abs()

    #     # legacy delta-based tie-break
    #     take1_tie = d1 >= d2   # >= keeps legacy "task1 wins ties"
    #     take1 = torch.where(tie, take1_tie, take1)
    # else:
    #     # no base → default legacy: task1 wins ties
    #     take1 = take1 | tie

    # merged = torch.where(take1, x1, x2)
    # return merged, take1

    # take1 = x1 > x2
    take1 = x1.abs() >= x2.abs()
    merged = torch.where(take1, x1, x2)
    return merged, take1


# ============================================================
# Group lora_A / lora_B pairs from keys
# ============================================================
def build_lora_AB_groups_from_keys(keys: List[str]) -> List[Dict[str, str]]:
    """
    Detect LoRA pairs:
      ...lora_A...  <->  ...lora_B...
    Returns list of dicts: [{"A": a_name, "B": b_name}, ...]
    """
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
    return groups


# ============================================================
# LoRA gauge reparam:  B@A == (B@S) @ (S^{-1}@A), S diagonal
# We will optimize S to make TASK1 win max_abs MORE THAN BASELINE
# ============================================================
def _soft_win_prob(x1: torch.Tensor, x2: torch.Tensor, tau: float, eps: float) -> torch.Tensor:
    # probability-like surrogate for "abs(x1) > abs(x2)"
    margin = (x1.abs() - x2.abs()) / (x2.abs() + eps)
    return torch.sigmoid(margin / tau).mean()

def optimize_S_diagonal_prefer_task1(
    B1, A1, B2, A2,
    *,
    tau: float = 0.05,
    eps: float = 1e-8,
    steps: int = 200,
    lr: float = 1e-2,
    s_min: float = 1e-2,
    s_max: float = 1e2,
    # weights between making B win vs making A win (because scaling trades off)
    wB: float = 1.0,
    wA: float = 1.0,
    # "more often than baseline" targets (fractions in [0,1])
    target_pB: Optional[float] = None,
    target_pA: Optional[float] = None,
    target_margin: float = 0.02,   # require +2% above baseline by default
    target_strength: float = 10.0, # penalty weight for missing targets
    # keep S not too extreme
    reg_identity: float = 1e-3,    # penalty on log_s^2
):
    """
    Learn s (diag(S)) to increase the chance that TASK1 wins max_abs vs TASK2
    for BOTH B and A (surrogate), while preserving TASK1 function:
        B1@A1 == (B1S)(S^{-1}A1).

    Shapes:
      B*: (..., r)  shared dim is last
      A*: (r, ...)  shared dim is first
    """
    device = B1.device
    r = B1.shape[-1]

    # optimize in fp32
    B1_ = B1.detach().to(torch.float32)
    A1_ = A1.detach().to(torch.float32)
    B2_ = B2.detach().to(torch.float32)
    A2_ = A2.detach().to(torch.float32)

    log_s = torch.zeros(r, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_s], lr=lr)

    b_scale_shape = [1] * (B1_.ndim - 1) + [r]
    a_scale_shape = [r] + [1] * (A1_.ndim - 1)

    for _ in range(steps):
        s = torch.exp(log_s).clamp(s_min, s_max)

        # reparam task1
        B1p = B1_ * s.view(*b_scale_shape)     # B1S
        A1p = A1_ / s.view(*a_scale_shape)     # S^{-1}A1

        pB = _soft_win_prob(B1p, B2_, tau=tau, eps=eps)
        pA = _soft_win_prob(A1p, A2_, tau=tau, eps=eps)

        # maximize pB and pA => minimize negative
        loss = -(wB * pB + wA * pA)

        # enforce "more often than baseline"
        if target_pB is not None:
            tgtB = min(1.0, float(target_pB) + target_margin)
            loss = loss + target_strength * torch.relu(torch.tensor(tgtB, device=device) - pB) ** 2
        if target_pA is not None:
            tgtA = min(1.0, float(target_pA) + target_margin)
            loss = loss + target_strength * torch.relu(torch.tensor(tgtA, device=device) - pA) ** 2

        # keep near identity to avoid extreme scaling
        if reg_identity > 0:
            loss = loss + reg_identity * (log_s ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    s = torch.exp(log_s).clamp(s_min, s_max).detach().to(dtype=B1.dtype)
    return s


def reparam_lora_with_s(B, A, s):
    r = B.shape[-1]
    Bp = B * s.view(*([1] * (B.ndim - 1)), r)
    Ap = A / s.view(r, *([1] * (A.ndim - 1)))
    return Bp, Ap


def compute_s_simple(B1, B2, eps=1e-8):
    # B*: (out_dim, rank)
    n1 = torch.norm(B1, dim=0)
    n2 = torch.norm(B2, dim=0)
    s = torch.sqrt((n2 + eps) / (n1 + eps))
    return s

# ============================================================
# FIXED merge: for LoRA params, choose from PARAMS (not deltas)
# + also report baseline max_abs selection percentages (params)
# ============================================================
def merge_lora_optS_max_abs_and_stats(
    base_params: Dict[str, torch.Tensor],
    tasks_params: List[Dict[str, torch.Tensor]],
    *,
    lamb: float = 1.0,
    steps: int = 200,
    lr: float = 1e-2,
    tau: float = 0.05,
    eps: float = 1e-8,
    s_min: float = 1e-2,
    s_max: float = 1e2,
    wB: float = 1.0,
    wA: float = 1.0,
    target_margin: float = 0.02,
    target_strength: float = 10.0,
    reg_identity: float = 1e-3,
):
    """
    Returns:
        merged_optS   : dict[str, Tensor]   # opt-S + max_abs (PARAM-based for LoRA)
        merged_base   : dict[str, Tensor]   # baseline max_abs (PARAM-based for LoRA)
        stats         : dict                # stats for analysis/comparison
    """
    assert len(tasks_params) == 2, "Expected exactly 2 tasks"
    t1, t2 = tasks_params

    keys = list(t1.keys())
    ab_groups = build_lora_AB_groups_from_keys(keys)
    A_to_B = {g["A"]: g["B"] for g in ab_groups}

    merged_optS = {}
    merged_base = {}
    handled = set()

    stats = {
        "baseline_lora": {"task1": 0, "task2": 0, "total": 0},
        "opt_lora": {"task1": 0, "task2": 0, "total": 0},
        "per_pair": [],
    }

    for name in keys:
        if name in handled:
            continue

        # --------------------------------------------------
        # LoRA parameters: PARAM-based merging
        # --------------------------------------------------
        if name in A_to_B:
            a_name = name
            b_name = A_to_B[name]
            handled.add(a_name)
            handled.add(b_name)

            A1, B1 = t1[a_name], t1[b_name]
            A2, B2 = t2[a_name], t2[b_name]

            # ---------- baseline max_abs (PARAMS) ----------
            mA_base, maskA_base = maxabs_merge_with_mask(A1, A2, base=base_params[a_name])
            mB_base, maskB_base = maxabs_merge_with_mask(B1, B2, base=base_params[b_name])

            merged_base[a_name] = base_params[a_name] + lamb * (mA_base - base_params[a_name])
            merged_base[b_name] = base_params[b_name] + lamb * (mB_base - base_params[b_name])

            baseA_t1 = int(maskA_base.sum().item())
            baseB_t1 = int(maskB_base.sum().item())
            baseA_tot = maskA_base.numel()
            baseB_tot = maskB_base.numel()

            stats["baseline_lora"]["task1"] += baseA_t1 + baseB_t1
            stats["baseline_lora"]["task2"] += (baseA_tot - baseA_t1) + (baseB_tot - baseB_t1)
            stats["baseline_lora"]["total"] += baseA_tot + baseB_tot

            # ---------- opt-S ----------
            s = optimize_S_diagonal_prefer_task1(
                B1, A1, B2, A2,
                tau=tau,
                eps=eps,
                steps=steps,
                lr=lr,
                s_min=s_min,
                s_max=s_max,
                wB=wB,
                wA=wA,
                target_pB=baseB_t1 / max(1, baseB_tot),
                target_pA=baseA_t1 / max(1, baseA_tot),
                target_margin=target_margin,
                target_strength=target_strength,
                reg_identity=reg_identity,
            )

            # s = compute_s_simple(B1, B2, eps=eps)  # Simple fallback

            # B1p, A1p = reparam_lora_with_s(B1, A1, s)

            # mA_opt, maskA_opt = maxabs_merge_with_mask(A1p, A2, base=base_params[a_name])
            # mB_opt, maskB_opt = maxabs_merge_with_mask(B1p, B2, base=base_params[b_name])

            # B2p, A2p = reparam_lora_with_s(B2, A2, s)
            # mA_opt, maskA_opt = maxabs_merge_with_mask(A1, A2p, base=base_params[a_name])
            # mB_opt, maskB_opt = maxabs_merge_with_mask(B1, B2p, base=base_params[b_name])


            # rotational matrix
            # Diagonal scaling D
            n1 = torch.linalg.norm(B1, dim=0)
            n2 = torch.linalg.norm(B2, dim=0)
            d = (n2 + eps) / (n1 + eps)
            sqrt_d = torch.sqrt(d)
            inv_sqrt_d = 1.0 / (sqrt_d + EPSILON)

            # --- Orthogonal rotation R
            B1f = B1.to(torch.float32)
            B2f = B2.to(torch.float32)

            U1 = B1f.transpose(-2, -1) @ B1f
            U2 = B2f.transpose(-2, -1) @ B2f

            # Symmetrize to avoid tiny numerical asymmetry
            U1 = 0.5 * (U1 + U1.T)
            U2 = 0.5 * (U2 + U2.T)

            _, Q1 = torch.linalg.eigh(U1)
            _, Q2 = torch.linalg.eigh(U2)

            # R = Q2 @ Q1.transpose(-2, -1)

            # # S = D^{1/2} * R * D^{1/2}
            # r = B1.shape[-1]
            # S = (sqrt_d.view(r, 1) * R) * inv_sqrt_d.view(1, r)
            # B1p = B1 @ S
            # A1p = torch.linalg.solve(S, A1)

            R = Q2 @ Q1.T
            B1p = B1 @ R
            A1p = R.T @ A1

            mA_opt, maskA_opt = maxabs_merge_with_mask(A1p, A2, base=base_params[a_name])
            mB_opt, maskB_opt = maxabs_merge_with_mask(B1p, B2, base=base_params[b_name])


            merged_optS[a_name] = base_params[a_name] + lamb * (mA_opt - base_params[a_name])
            merged_optS[b_name] = base_params[b_name] + lamb * (mB_opt - base_params[b_name])

            optA_t1 = int(maskA_opt.sum().item())
            optB_t1 = int(maskB_opt.sum().item())
            optA_tot = maskA_opt.numel()
            optB_tot = maskB_opt.numel()

            stats["opt_lora"]["task1"] += optA_t1 + optB_t1
            stats["opt_lora"]["task2"] += (optA_tot - optA_t1) + (optB_tot - optB_t1)
            stats["opt_lora"]["total"] += optA_tot + optB_tot

            stats["per_pair"].append({
                "A": a_name,
                "B": b_name,
                "baseline_A_p": baseA_t1 / max(1, baseA_tot),
                "baseline_B_p": baseB_t1 / max(1, baseB_tot),
                "opt_A_p": optA_t1 / max(1, optA_tot),
                "opt_B_p": optB_t1 / max(1, optB_tot),
            })
            continue

        # --------------------------------------------------
        # Non-LoRA parameters: delta-based max_abs (same for both)
        # --------------------------------------------------
        handled.add(name)
        base = base_params[name]
        d1 = t1[name] - base
        d2 = t2[name] - base

        m, _ = maxabs_merge_with_mask(d1, d2, base=base)
        merged_val = base + lamb * m

        merged_base[name] = merged_val
        merged_optS[name] = merged_val

    return merged_optS, merged_base, stats
# ============================================================


@torch.no_grad()
def select_ranks_by_scale_normalized_contrast(
    e1: torch.Tensor,
    e2: torch.Tensor,
    tau: float = 0.0,
    eps: float = 1e-8,
):
    """
    Scale-aware rank selection using normalized energy contrast.

    Args:
        e1, e2: (r,) rank energies from past and current tasks
        tau: threshold (0 = fair, >0 favors past, <0 favors current)
    """
    # joint scale
    scale = e1.std(unbiased=False) + e2.std(unbiased=False) + eps

    # normalized contrast
    score = (e1 - e2) / scale

    keep1 = score >= tau
    return keep1


from scipy.optimize import linear_sum_assignment
def merge_tasks_rankwise(task1, task2, lamb=1.0):
    @torch.no_grad()
    def lora_rank_similarity(A1, B1, A2, B2):
        B1n = F.normalize(B1, dim=0)
        B2n = F.normalize(B2, dim=0)
        A1n = F.normalize(A1, dim=1)
        A2n = F.normalize(A2, dim=1)
        return (B1n.T @ B2n) * (A1n @ A2n.T)

    @torch.no_grad()
    def align_lora_ranks(A1, B1, A2, B2):
        S = lora_rank_similarity(A1, B1, A2, B2)
        cost = (S.max() - S).cpu().numpy()
        _, col = linear_sum_assignment(cost)
        col = torch.tensor(col, device=A2.device)
        return A2[col, :], B2[:, col]

    @torch.no_grad()
    def lora_rank_energy(A, B):
        # return torch.norm(B, dim=0) * torch.norm(A, dim=1)
        return torch.sum(torch.abs(B), dim=0) * torch.sum(torch.abs(A), dim=1)

    merged = {}
    keys = list(task1.keys())
    ab_groups = build_lora_AB_groups_from_keys(keys)

    for g in ab_groups:
        A_key, B_key = g["A"], g["B"]

        A1, B1 = task1[A_key], task1[B_key]
        A2, B2 = task2[A_key], task2[B_key]

        # 1) Git Re-Basin: align LoRA ranks
        A2p, B2p = align_lora_ranks(A1, B1, A2, B2)

        # 2) Rank-wise structured merge (past-favoring)
        e1 = lora_rank_energy(A1, B1)
        e2 = lora_rank_energy(A2p, B2p)
        # print(torch.min(e1), torch.max(e1), torch.min(e2), torch.max(e2))

        keep1 = e1 >= lamb * e2
        # keep1 = select_ranks_by_scale_normalized_contrast(e1, e2, tau=0.2)

        A = torch.where(keep1[:, None], A1, A2p)
        B = torch.where(keep1[None, :], B1, B2p)

        merged[A_key] = A
        merged[B_key] = B

    return merged

def merge_lora_tasks_rank_permuted_max(task1, task2, bias=0.0):
    """
    Rank-wise max merge with heuristic permutation bias (no similarity matching).
    Structure-safe: outputs valid LoRA A/B.
    """
    import torch

    def build_lora_AB_groups_from_keys(keys):
        keyset = set(keys)
        groups = []
        seen = set()
        for k in keys:
            if ".lora_A." in k:
                b = k.replace(".lora_A.", ".lora_B.")
                if b in keyset and (k, b) not in seen:
                    seen.add((k, b))
                    groups.append({"A": k, "B": b})
        return groups

    @torch.no_grad()
    def rank_energy(A, B):
        return torch.norm(B, dim=0) * torch.norm(A, dim=1)

    merged = {}
    keys = list(task1.keys())
    ab_groups = build_lora_AB_groups_from_keys(keys)

    for g in ab_groups:
        A_key, B_key = g["A"], g["B"]

        A1, B1 = task1[A_key], task1[B_key]
        A2, B2 = task2[A_key], task2[B_key]

        # heuristic permutation: sort ranks by dominance
        e1 = rank_energy(A1, B1)
        e2 = rank_energy(A2, B2)
        dominance = e1 - e2

        perm = torch.argsort(dominance, descending=True)

        A1p, B1p = A1[perm], B1[:, perm]
        A2p, B2p = A2[perm], B2[:, perm]

        # rank-wise max merge
        keep1 = e1[perm] >= (e2[perm] - bias)

        A = torch.where(keep1[:, None], A1p, A2p)
        B = torch.where(keep1[None, :], B1p, B2p)

        merged[A_key] = A
        merged[B_key] = B

    return merged



def merge_lora_tasks_mask_aware_rankwise(task1, task2, k=4, bias=0.0):
    """
    Mask-aware rank-wise LoRA merge.
    Structure-safe and function-preserving.
    """
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment

    def build_lora_AB_groups_from_keys(keys):
        keyset = set(keys)
        groups = []
        seen = set()
        for k in keys:
            if ".lora_A." in k:
                b = k.replace(".lora_A.", ".lora_B.")
                if b in keyset and (k, b) not in seen:
                    seen.add((k, b))
                    groups.append({"A": k, "B": b})
        return groups

    @torch.no_grad()
    def rank_similarity(A1, B1, A2, B2):
        return (
            F.normalize(B1, dim=0).T @ F.normalize(B2, dim=0)
        ) * (
            F.normalize(A1, dim=1) @ F.normalize(A2, dim=1).T
        )

    @torch.no_grad()
    def align(A1, B1, A2, B2):
        S = rank_similarity(A1, B1, A2, B2)
        cost = (S.max() - S).cpu().numpy()
        _, col = linear_sum_assignment(cost)
        col = torch.tensor(col, device=A2.device)
        return A2[col], B2[:, col]

    @torch.no_grad()
    def masked_rank_energy(A, B, k):
        W = B @ A
        vals, _ = torch.topk(torch.abs(W), k, dim=1)
        return vals.sum(dim=1).mean() * torch.ones(A.shape[0], device=A.device)

    merged = {}
    keys = list(task1.keys())
    ab_groups = build_lora_AB_groups_from_keys(keys)

    for g in ab_groups:
        A_key, B_key = g["A"], g["B"]

        A1, B1 = task1[A_key], task1[B_key]
        A2, B2 = task2[A_key], task2[B_key]

        A2p, B2p = align(A1, B1, A2, B2)

        e1 = masked_rank_energy(A1, B1, k)
        e2 = masked_rank_energy(A2p, B2p, k)

        keep1 = e1 >= (e2 - bias)

        A = torch.where(keep1[:, None], A1, A2p)
        B = torch.where(keep1[None, :], B1, B2p)

        merged[A_key] = A
        merged[B_key] = B

    return merged



class Learner:
    def __init__(self, config,):
        self._config = config
        self._known_classes = 0
        self._total_classes = 0
        self._class_increments = []
        self._cur_task = -1
        self._acc_matrix = []

        self._backbone = get_backbone(config)
        self._backbone.cuda()
        torch.save(self.get_trainable_params(), self.backbone_checkpoint(-1))
        self._classifier = None

    def update_classifier(self):
        classifier = CosineLinear(self._backbone.out_dim, self._total_classes).cuda()

        if self._classifier is not None:
            old_weights = self._classifier.weight.data
            classifier.weight.data[: old_weights.size(0)] = old_weights
        self._classifier = classifier
        del classifier
    
    def get_features(self, x):
        z = self._backbone(x)
        return z

    def infer(self, x):
        z = self.get_features(x)
        logits = self._classifier(z)
        return logits

    def learn(self, data_manager):
        self.data_manager = data_manager
        num_tasks = data_manager.nb_tasks

        for task in range(num_tasks):
            self.before_task(task)
            self.train()
            self.eval()
            self.after_task()

    def before_task(self, task):
        task_size = self.data_manager.get_task_size(task)
        self._total_classes = self._known_classes + task_size
        self._class_increments.append((self._known_classes, self._total_classes - 1))
        self._cur_task = task

    def after_task(self):
        self._known_classes = self._total_classes

    def train(self):
        self._backbone.train()
        self._backbone.cuda()

        checkpoint_path = self.backbone_checkpoint(self._cur_task)
        if not os.path.exists(checkpoint_path) or self._config["reset_train"]:
            # self.load_backbone(torch.load(self.backbone_checkpoint(self._cur_task - 1)))

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
            base_lr = 1e-2
            weight_decay = 5e-4

            feature_dim = self._backbone.out_dim
            num_classes = self._total_classes - self._known_classes
            classifier = ContinualLinear(feature_dim, num_classes, with_norm=True).cuda()

            parameters = [
                {
                    "params": [
                        p for p in self._backbone.parameters() if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for p in classifier.parameters() if p.requires_grad
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

            logging.info(f"[Training] Task {self._cur_task}")

            for epoch in range(epochs):
                total_loss, total_acc, total = 0, 0, 0
                for _, (_, _, x, y) in enumerate(train_loader):
                    x, y = x.cuda(), y.cuda()
                    y = torch.where(
                        y - self._known_classes >= 0, y - self._known_classes, -100
                    )
                    z = self._backbone(x)
                    logits = classifier(z)
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
                self.get_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )

            del classifier
        else:
            logging.info(f"[Training] Load checkpoint for task {self._cur_task}")
            backbone_params = torch.load(self.backbone_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)

        self.merge()

        feature_dim = self._backbone.out_dim

        self.update_classifier()

        for cls in range(self._known_classes, self._total_classes):
            proto_set = self.data_manager.get_dataset(
                np.arange(cls, cls + 1),
                source="train",
                mode="test"
            )
            loader = DataLoader(proto_set, batch_size=512, shuffle=False)

            feats = []
            self._backbone.eval()
            with torch.no_grad():
                for _, (_, _, x, _) in enumerate(loader):
                    x = x.cuda()
                    z = self.get_features(x)
                    feats.append(z.cpu())

            feats = torch.cat(feats, dim=0)
            self._classifier.weight.data[cls] = feats.mean(dim=0)

    # ============================================================
    # Your method wrapper (prints baseline vs opt, and returns merged)
    # ============================================================
    def merge_v2(
        self,
        base_params: Dict[str, torch.Tensor],
        tasks_params: List[Dict[str, torch.Tensor]],
        lamb: float = 1.0,
        steps: int = 200,
        lr: float = 1e-2,
        tau: float = 0.05,
        eps: float = 1e-8,
        s_min: float = 1e-2,
        s_max: float = 1e2,
        wB: float = 1.0,
        wA: float = 1.0,
        target_margin: float = 0.02,
        target_strength: float = 10.0,
        reg_identity: float = 1e-3,
        verbose: bool = True,
    ):
        merged_opt, merged_base, stats = merge_lora_optS_max_abs_and_stats(
            base_params=base_params,
            tasks_params=tasks_params,
            lamb=lamb,
            steps=steps,
            lr=lr,
            tau=tau,
            eps=eps,
            s_min=s_min,
            s_max=s_max,
            wB=wB,
            wA=wA,
            target_margin=target_margin,
            target_strength=target_strength,
            reg_identity=reg_identity,
        )

        if verbose:
            def pct(a, tot): 
                return 0.0 if tot == 0 else 100.0 * a / tot

            bl = stats["baseline_lora"]
            op = stats["opt_lora"]

            print(
                "[merge_v2] LoRA baseline (PARAM max_abs): "
                f"task1={pct(bl['task1'], bl['total']):.2f}% "
                f"task2={pct(bl['task2'], bl['total']):.2f}% "
                f"total={bl['total']}"
            )

            print(
                "[merge_v2] LoRA optS (PARAM max_abs):      "
                f"task1={pct(op['task1'], op['total']):.2f}% "
                f"task2={pct(op['task2'], op['total']):.2f}% "
                f"total={op['total']}"
            )

            # Optional: per-pair diagnostic (VERY useful for debugging)
            if "per_pair" in stats:
                for i, g in enumerate(stats["per_pair"][:5]):  # cap prints
                    print(
                        f"  [pair {i:02d}] "
                        f"A: {g['baseline_A_p']*100:.1f}% → {g['opt_A_p']*100:.1f}% | "
                        f"B: {g['baseline_B_p']*100:.1f}% → {g['opt_B_p']*100:.1f}%"
                    )

        return merged_opt

    def merge(self):
        # if os.path.exists(self.merged_checkpoint(self._cur_task)):
        #     logging.info(f"[Merging] Load merged checkpoint for task {self._cur_task}")
        #     backbone_params = torch.load(self.merged_checkpoint(self._cur_task))
        #     self.load_backbone(backbone_params)
        #     return

        if self._cur_task == 0:
            logging.info(
                f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
            )
            torch.save(
                self.get_trainable_params(),
                self.merged_checkpoint(self._cur_task),
            )
            return

        base_params = torch.load(self.backbone_checkpoint(-1))

        tasks_params = []
        tasks_params.append(torch.load(self.merged_checkpoint(self._cur_task - 1)))
        # tasks_params.append(torch.load(self.backbone_checkpoint(self._cur_task - 1)))
        tasks_params.append(torch.load(self.backbone_checkpoint(self._cur_task)))

        # logging.info(f"[Merging] Loaded {len(tasks_params)} tasks for merging")
        
        # merged_params = merge(
        #     base_params, tasks_params, method="ties"
        # )

        # merged_params = self.merge_v2(
        #     base_params, tasks_params, target_strength=1.0)

        merged_params = merge_tasks_rankwise(
            tasks_params[0], tasks_params[1], lamb=0.85)

        # merged_params = merge_lora_tasks_rank_permuted_max(
        #     tasks_params[0], tasks_params[1])

        # merged_params = merge_lora_tasks_mask_aware_rankwise(
        #     tasks_params[0], tasks_params[1], k=10)


        # total_elements = 0
        # greater_count = 0
        # all_diffs = []
        # all_merged_vals = []
        # all_base_vals = []
        
        # for key in merged_params.keys():
        #     if key in base_params:
        #         merged_tensor = merged_params[key].detach().float()
        #         base_tensor = base_params[key].detach().float()
                
        #         total_elements += merged_tensor.numel()
        #         greater_count += (merged_tensor > base_tensor).sum().item()
                
        #         diff = (merged_tensor - base_tensor).cpu().numpy().flatten()
        #         all_diffs.extend(diff)
        #         all_merged_vals.extend(merged_tensor.cpu().numpy().flatten())
        #         all_base_vals.extend(base_tensor.cpu().numpy().flatten())
        
        # all_diffs = np.array(all_diffs)
        # all_merged_vals = np.array(all_merged_vals)
        # all_base_vals = np.array(all_base_vals)
        
        # pct_greater = 100.0 * greater_count / total_elements if total_elements > 0 else 0.0
        
        # logging.info(f"[Merge Diagnostics]")
        # logging.info(f"  Total elements: {total_elements}")
        # logging.info(f"  merged > base: {pct_greater:.2f}%")
        # logging.info(f"  Difference (merged - base):")
        # logging.info(f"    Mean: {all_diffs.mean():.6e}, Std: {all_diffs.std():.6e}")
        # logging.info(f"  Merged values:")
        # logging.info(f"    Mean: {all_merged_vals.mean():.6e}, Std: {all_merged_vals.std():.6e}")
        # logging.info(f"  Base values:")
        # logging.info(f"    Mean: {all_base_vals.mean():.6e}, Std: {all_base_vals.std():.6e}")
        
        
        self.load_backbone(merged_params)

        logging.info(
            f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
        )
        torch.save(
            self.get_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )

    def eval(self):
        test_set = self.data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )

        loader = DataLoader(test_set, batch_size=256, shuffle=False)
        self._backbone.eval()
        self._classifier.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for _, (_, _, x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                logits = self.infer(x)

                y_pred.append(logits.argmax(dim=1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        _, grouped = accuracy(y_pred.T, y_true, self._class_increments)
        grouped = [float(acc) for acc in grouped]
        self._acc_matrix.append(grouped)
        logging.info(f"[Evaluation] Grouped: {grouped}")

        num_tasks = self._cur_task + 1
        acc_matrix = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(i + 1):
                acc_matrix[i, j] = self._acc_matrix[i][j]
        faa_nme, ffm_nme, ffd_nme, asa_nme = compute_metrics(acc_matrix)
        logging.info(
            f"[Evaluation] FAA: {faa_nme:.2f}, FFM: {ffm_nme:.2f}, FFD: {ffd_nme:.2f}, ASA: {asa_nme:.2f}"
        )

        self._faa_nme = faa_nme
        self._asa_nme = asa_nme

    def prefix(self):
        prefix = "_".join([
            str(self._config["seed"]),
            self._config["dataset_name"],
            str(self._config["dataset_num_task"]),
            self._config["model_backbone"],
        ])

        train_prefix = self._config.get("train_prefix", None)
        if train_prefix is not None:
            prefix = f"{train_prefix}_{prefix}"
        
        return prefix

    def backbone_checkpoint(self, task=-1):
        name = f"{self.prefix()}_backbone"
        if task >= 0:
            name += f"_{task}"
        return os.path.join(CHECKPOINT_DIR, name + ".pt")

    def merged_checkpoint(self, task):
        return os.path.join(CHECKPOINT_DIR, f"{self.prefix()}_merged_{task}.pt")

    def get_trainable_params(self):
        return {
            name: param
            for name, param in self._backbone.named_parameters()
            if param.requires_grad
        }

    def load_backbone(self, backbone_params):
        self._backbone.load_state_dict(backbone_params, strict=False)


# ------------------------------------------------------------------
# Experiment runner (unchanged)
# ------------------------------------------------------------------
# (Keep your DATA_TABLE, BASE_CONFIG, run_experiments, etc. as-is)

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
    "reset_train": True,
    "train_epochs": 10,
    "train_batch_size": 16,
    "lora_r": 16,
    "lora_alpha": 32,
    "train_incremental": True,
    "train_merge_incremental": True,
    "model_backbone": "vit_base",
    "train_prefix": "10e_16r",
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
    
    experiment_name = f"{dataset_name}_{config_name}"
    result = {}
    try:
        logging.info("Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

        learner = Learner(config)
        learner.learn(data_manager)
        
        nme_faa = learner._faa_nme
        nme_asa = learner._asa_nme
        
        logging.info(f"[Experiment {experiment_name}]")
        logging.info(f"  Configuration: {experiment_config}")
        logging.info(f"  NME - FAA: {nme_faa:.2f}, ASA: {nme_asa:.2f}")

        del learner
        torch.cuda.empty_cache()
        gc.collect()

        result["nme_faa"] = nme_faa
        result["nme_asa"] = nme_asa
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"[Experiment {experiment_name}] Detailed Error:")
        logging.error(f"Exception Type: {type(e).__name__}")
        logging.error(f"Exception Message: {str(e)}")
        logging.error(f"Full Traceback:\n{error_details}")

        result["nme_faa"] = 0.0
        result["nme_asa"] = 0.0

    return result


def run_experiments():
    seeds = [1993]

    experiment_configs = {
        "default": {},
    }
    
    for dataset_name in DATA_TABLE.keys():
        print(f"\n{'='*60}")
        print(f"Starting experiments for dataset: {dataset_name}")
        print(f"{'='*60}")

        # Collect results for this dataset
        dataset_results = {}
        dataset_results["faa"] = []
        dataset_results["asa"] = []

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

                dataset_results["faa"].append(result["nme_faa"])
                dataset_results["asa"].append(result["nme_asa"])
                

if __name__ == "__main__":
    start_time = time.time()
    results = run_experiments()
    total_time = time.time() - start_time
    print(f"Total experiment time: {total_time:.2f} seconds")
