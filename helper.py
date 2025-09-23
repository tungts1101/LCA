import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers.weight_init import trunc_normal_
from peft import get_peft_model, LoraConfig
import numpy as np
import logging
import random
import math


# Helper functions ============================================================
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_metrics(accuracy_matrix):
    faa = np.mean(accuracy_matrix[-1])

    session_averages = []
    for i in range(accuracy_matrix.shape[0]):
        session_avg = np.mean(
            accuracy_matrix[i, : i + 1]
        )  # calculate total accuracy per session
        session_averages.append(session_avg)
    asa = np.mean(session_averages)

    if accuracy_matrix.shape[0] == 1:
        return faa, 0.0, 0.0, asa

    final_acc_per_task = accuracy_matrix[-1]
    max_acc_per_task = np.max(accuracy_matrix, axis=0)
    ffm = np.mean(max_acc_per_task[:-1] - final_acc_per_task[:-1])
    ffd = np.max(max_acc_per_task[:-1] - final_acc_per_task[:-1]) - np.min(
        max_acc_per_task[:-1] - final_acc_per_task[:-1]
    )

    return faa, ffm, ffd, asa


def setup_logger(log_file=f"logs/default.log", logger_name=None):
    if logger_name is None:
        logger_name = f"logger_{log_file.replace('/', '_').replace('.', '_')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def trim(tensor, topk=100):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * topk / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))

    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)

    return (trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor))


def merge_task_vectors(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = gamma_tvs == gamma
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(
        dim=0
    ) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs


def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in base_params:
        base_tv = base_params[name].clone()
        task_vectors = [task_params[name] for task_params in tasks_params]

        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]

        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "min":
            merged_tv = torch.min(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "max_abs":
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = torch.abs(stacked)
            max_idx = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, max_idx.unsqueeze(0)).squeeze(0)

        params[name] = base_tv + lamb * merged_tv

    return params


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(y_pred, y_true, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = []
    acc_total = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    for task_id, classes in enumerate(class_increments):
        idxes = np.where(np.logical_and(y_true >= classes[0], y_true <= classes[1]))[0]
        all_acc.append(
            np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

    return acc_total, all_acc


# =============================================================================


# Backbone ====================================================================
def get_backbone(args):
    name = args["model_backbone"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif "lora" in name:
        model = timm.create_model(name[:-5], pretrained=True, num_classes=0)
        model.requires_grad_(False)
        model.out_dim = 768
        lora_config = LoraConfig(
            r=args["model_lora_r"],
            lora_alpha=args["model_lora_alpha"],
            target_modules=args["model_lora_target_modules"],
            lora_dropout=args["model_lora_dropout"],
            bias="none",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        return model
    elif '_ssf' in name:
        from backbone import vit_ssf
        if name == "pretrained_vit_b16_224_ssf":
            model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        elif name == "pretrained_vit_b16_224_in21k_ssf":
            model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        return model.eval()
    elif '_vpt' in name:
        from backbone.vpt import build_promptmodel
        if name == "pretrained_vit_b16_224_vpt":
            basicmodelname = "vit_base_patch16_224" 
        elif name == "pretrained_vit_b16_224_in21k_vpt":
            basicmodelname = "vit_base_patch16_224_in21k"
        
        print("modelname,", name, "basicmodelname", basicmodelname)
        VPT_type = "Deep"
        if args["vpt_type"] == 'shallow':
            VPT_type = "Shallow"
        Prompt_Token_num = args["prompt_token_num"]

        model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
        prompt_state_dict = model.obtain_prompt()
        model.load_prompt(prompt_state_dict)
        model.out_dim = 768
        return model.eval()
    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        from backbone import vit_adapter
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        if name == "pretrained_vit_b16_224_adapter":
            model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        elif name == "pretrained_vit_b16_224_in21k_adapter":
            model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim=768
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


# =============================================================================


# Classifier ==================================================================
class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, with_norm=False, with_bias=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.with_norm = with_norm
        self.with_bias = with_bias

        self.heads = nn.ModuleList([])
        self.update(nb_classes)

    def create_head(self, nb_classes):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
        fc = nn.Linear(self.embed_dim, nb_classes, bias=self.with_bias)
        trunc_normal_(fc.weight, std=0.02)
        if self.with_bias:
            nn.init.constant_(fc.bias, 0)
        single_head.append(fc)
        head = nn.Sequential(*single_head)
        return head

    def update(self, nb_classes, freeze_old=True):
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
        single_head = self.create_head(nb_classes)
        self.heads.append(single_head)

    def forward(self, x, return_dict=False):
        out = []
        for i, head in enumerate(self.heads):
            logits_i = head(x)

            out.append(logits_i)

        if return_dict:
            return {"logits": torch.cat(out, dim=1)}
        return torch.cat(out, dim=1)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, return_dict=False):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if return_dict:
            return {'logits': out}
        return out
# =============================================================================


# Model =======================================================================
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.backbone = get_backbone(config)
        self.norm = nn.LayerNorm(self.backbone.num_features)
        self.classifier = None

    @property
    def feature_dim(self):
        return self.backbone.num_features

    def update_classifier(self, num_classes, freeze_old=True):
        if self.classifier == None:
            self.classifier = ContinualLinear(
                self.feature_dim, num_classes, with_norm=False, with_bias=False
            )
        else:
            self.classifier.update(num_classes, freeze_old=freeze_old)

    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param

        if self.norm is not None:
            for name, param in self.norm.named_parameters():
                if param.requires_grad:
                    params[f"norm.{name}"] = param

        return params

    def get_features(self, x):
        z = self.backbone(x)
        if self.norm is not None:
            z = self.norm(z)
        return z

    def forward(self, x):
        z = self.get_features(x)
        y = self.classifier(z)
        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        return f"Model(trainable_params={trainable_params:,}, total_params={total_params:,}, percentage={trainable_params * 100 / total_params:.2f})"


# =============================================================================
