import copy
import gc
import logging
import os
import sys

import optuna
import torch

from exp16 import Learner, run_single_experiment, DATA_TABLE, LOG_DIR, BASE_CONFIG

SEED = 1
DATASET_NAME = "imagenetr"

LOG_DIR_OPTUNA = os.path.join(LOG_DIR, "optuna")
os.makedirs(LOG_DIR_OPTUNA, exist_ok=True)


BASE_EXPERIMENT_CONFIG = {
    "reset_train": True,
    "reset_merge": True,
    "train_epochs": 10,
    "train_batch_size": 64,
    "model_backbone": "vit_base_patch16_224_lora",
    "model_outdim": 768,
    "model_use_norm": True,
    "model_lora_r": 64,
    "model_lora_alpha": 128,
    "model_classifier_norm_layer": "ln",
    "ffn_num": 64,
    "model_lora_dropout": 0.0,
    "model_lora_target_modules": ["qkv"],
    "model_classifier": ["mlp"],

    "train_prefix": "exp16_optuna",
    "train_stop_at_task": 4,

    "train_minibatch_uniform": False,
    "train_minibatch_num_cls": 4,
    "align_minibatch_uniform": False,
    "align_minibatch_num_cls": 4,

    "train_reg_num_batch": 1,
    "train_reg_loss": "log_likelihood",
    "train_reg_mag_weight": 1.0,
    "train_reg_at_each_n_batch": 1,

    "train_first_task_only": False,
    "train_incremental": True,
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
}


SEARCH_SPACE = {
    "train_feature_at_layer": [8],
    "train_reg_weight":       [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    "train_reg_num_classes":  [5],
    "train_reg_num_sampling": [16],
}


def objective(trial: optuna.Trial) -> float:
    L   = trial.suggest_categorical("train_feature_at_layer", SEARCH_SPACE["train_feature_at_layer"])
    lam = trial.suggest_categorical("train_reg_weight",       SEARCH_SPACE["train_reg_weight"])
    N   = trial.suggest_categorical("train_reg_num_classes",  SEARCH_SPACE["train_reg_num_classes"])
    K   = trial.suggest_categorical("train_reg_num_sampling", SEARCH_SPACE["train_reg_num_sampling"])

    config = copy.deepcopy(BASE_EXPERIMENT_CONFIG)
    config["train_feature_at_layer"] = L
    config["train_reg_weight"]       = lam
    config["train_reg_num_classes"]  = N
    config["train_reg_num_sampling"] = K

    trial_name = f"trial{trial.number}_L{L}_lam{lam:.0e}_N{N}_K{K}"
    logging.info(f"\n{'='*60}\nTrial {trial.number}: {trial_name}\n{'='*60}")

    result = run_single_experiment(DATASET_NAME, trial_name, config, SEED)

    mlp_asa = result.get("mlp_asa", 0.0)
    logging.info(f"Trial {trial.number} mlp_asa={mlp_asa:.4f}")
    return mlp_asa


def run_optuna():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile = os.path.join(LOG_DIR_OPTUNA, f"{DATASET_NAME}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfile),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    n_trials = 1
    for v in SEARCH_SPACE.values():
        n_trials *= len(v)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE),
        study_name=f"exp16_{DATASET_NAME}",
        storage=f"sqlite:///{LOG_DIR_OPTUNA}/{DATASET_NAME}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    logging.info("\n" + "=" * 60)
    logging.info("Best trial:")
    best = study.best_trial
    logging.info(f"  mlp_asa : {best.value:.4f}")
    logging.info("  Params  :")
    for k, v in best.params.items():
        logging.info(f"    {k}: {v}")


if __name__ == "__main__":
    run_optuna()
