import torch
import os
import time
from utils.data_manager import DataManager
from helper import (
    set_random,
)
import gc
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse
import traceback
import logging
import sys
from lca import Learner, DATA_TABLE, BASE_CONFIG
import copy
from pathlib import Path


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

OPTUNA_DIR = "optuna"
os.makedirs(OPTUNA_DIR, exist_ok=True)

# cifar224: 98.8, 98.1, 97.71, 97.23, 96.74, 96.28, 95.81, 95.27, 94.8, 94.4
# imagenetr: 94.63, 92.46, 91.2, 90.22, 88.88, 87.9, 87.16, 86.58, 86.03, 85.47
# imageneta: 87.43, 86.06, 84.14, 81.79, 80.06, 78.4, 77.07, 75.89, 74.79, 73.72
# cub: 98.38, 97.72, 96.47, 95.35, 94.39, 93.37, 92.57, 91.72, 90.94, 90.18
# omnibenchmark: 94.5, 93.25, 91.57, 89.34, 87.5, 85.89, 84.52, 83.13, 81.78, 80.5
# vtab: 99.52, 98.64, 97.06, 95.92, 94.57

EARLY_PRUNING_THRESHOLDS = {
    "cifar224": {1: 98, 3: 97, 5: 96, 7: 95},
    "imagenetr": {1: 92, 3: 90, 5: 87, 7: 86},
    "imageneta": {1: 86, 3: 81, 5: 78, 7: 75},
    "cub": {1: 97, 3: 95, 5: 93, 7: 91},
    "omnibenchmark": {1: 93, 3: 89, 5: 85, 7: 83},
    "vtab": {1: 98, 3: 95}
}


def suggest_hyperparameters(trial):
    # ca_lr = trial.suggest_categorical("train_ca_lr", [1e-4, 1e-3, 1e-2])
    ca_lr = trial.suggest_float("train_ca_lr", 1e-4, 1e-2)

    # robust_weight_log = trial.suggest_categorical("robust_weight_log", [-3, -2, -1, 0, 1, 2, 3])
    robust_weight_log = trial.suggest_float("robust_weight_log", -2, 0)
    robust_weight = 10**robust_weight_log

    # entropy_weight_log = trial.suggest_categorical("entropy_weight_log", [-2, -1, 0, 1, 2])
    entropy_weight_log = trial.suggest_float("entropy_weight_log", -2, 0)
    entropy_weight = 10**entropy_weight_log

    ca_logit_norm = trial.suggest_float("train_ca_logit_norm", 0.1, 0.5)

    ca_lr = round(ca_lr, 5)
    robust_weight = round(robust_weight, 5)
    entropy_weight = round(entropy_weight, 5)
    ca_logit_norm = round(ca_logit_norm, 2)

    return {
        "train_ca_lr": ca_lr,
        "train_ca_rb_weight": robust_weight,
        "train_ca_entropy_weight": entropy_weight,
        "train_ca_logit_norm": ca_logit_norm,
    }


def objective(data_manager, trial, config):
    trial_start_time = time.time()
    try:
        hyperparams = suggest_hyperparameters(trial)
        config.update(hyperparams)

        logging.info(
            f"\n[Trial {trial.number}] Starting optimization with hyperparameters: {hyperparams}"
        )

        learner = Learner(
            config, trial=trial, pruning_thresholds=EARLY_PRUNING_THRESHOLDS.copy()
        )
        learner.learn(data_manager)

        acc = learner._acc

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.info(
            f"[Trial {trial.number}] Accuracy: {acc:.2f}, Duration: {trial_duration:.2f}s"
        )

        del learner
        torch.cuda.empty_cache()
        gc.collect()

        return acc

    except optuna.TrialPruned:
        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time

        logging.info(
            f"[Trial {trial.number}] Trial was pruned, Duration: {trial_duration:.2f}s"
        )

        torch.cuda.empty_cache()
        gc.collect()

        raise

    except Exception as e:
        logging.error(f"[Trial {trial.number}] Error during optimization: {str(e)}")
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

    logging.info("=" * 80)
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
    db_path = Path(OPTUNA_DIR) / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path}"

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

    try:
        best_value = study.best_value if study.best_value is not None else -float("inf")
        if best_value != -float("inf"):
            logging.info(f"Resuming study with existing best value: {best_value:.2f}")
    except ValueError:
        # No trials have been completed yet
        best_value = -float("inf")
        logging.info("Starting fresh study (no previous trials found)")
    
    min_delta = 0.01
    no_improvement_trials = 0

    def early_stopping_callback(study, trial):
        nonlocal best_value, no_improvement_trials, min_delta
        if trial is not None:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None and trial.value - min_delta > best_value:
                best_value = trial.value
                no_improvement_trials = 0
                logging.info(
                    f"New value: {best_value:.2f} at trial {trial.number}"
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

    config = copy.deepcopy(BASE_CONFIG)
    config["train_ca"] = True
    config["train_ca_samples_per_cls"] = 512
    config["train_ca_batch_size"] = 128
    config["train_ca_epochs"] = 10

    if dataset_name not in DATA_TABLE:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Available: {list(DATA_TABLE.keys())}"
        )

    dataset_num_task, dataset_init_cls, dataset_increment = DATA_TABLE[dataset_name][0]
    config["dataset_name"] = dataset_name
    config.update(
        {
            "dataset_num_task": dataset_num_task,
            "dataset_init_cls": dataset_init_cls,
            "dataset_increment": dataset_increment,
        }
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

    try:
        callbacks = [early_stopping_callback]
        study.optimize(
            lambda trial: objective(data_manager, trial, config),
            n_trials=n_trials,
            callbacks=callbacks,
            timeout=max_time_hours * 3600 if max_time_hours else None
        )
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCA Optuna Optimization")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATA_TABLE.keys()) + ["all"],
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
    
    if args.dataset == "all":
        for dataset in DATA_TABLE.keys():
            args.dataset = dataset
            run_optuna_optimization(
                dataset_name=args.dataset,
                n_trials=args.n_trials,
                early_stop_patience=args.early_stop_patience,
                max_time_hours=args.max_time_hours,
            )
    else:
        run_optuna_optimization(
            dataset_name=args.dataset,
            n_trials=args.n_trials,
            early_stop_patience=args.early_stop_patience,
            max_time_hours=args.max_time_hours,
        )
