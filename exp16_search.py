import copy
import itertools
import logging
import os
import sys

import numpy as np

from exp16 import run_single_experiment, LOG_DIR

SEEDS = [1993, 1994, 1995]
DATASET_NAME = "cars"

LOG_DIR_SEARCH = os.path.join(LOG_DIR, "search")
os.makedirs(LOG_DIR_SEARCH, exist_ok=True)


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

    "train_prefix": "exp16_search",
    "train_stop_at_task": -1,

    "train_minibatch_uniform": False,
    "train_minibatch_num_cls": 4,
    "align_minibatch_uniform": False,
    "align_minibatch_num_cls": 4,

    "train_reg_loss": "log_likelihood",
    "train_reg_mag_weight": 1.0,
    "train_reg_at_each_n_batch": 1,

    "train_RP": False,

    "train_merge": "ties",
    "train_merge_coef": 1.0,
    "train_merge_topk": 100,
    "train_merge_incremental": True,

    "train_ca": False,
    "train_ca_load_checkpoint_from_first_task": False,
    "train_ca_samples_per_cls": 512,
    "train_ca_batch_size": 64,
    "train_ca_epochs": 3,
    "train_ca_robust_weight": 1.0,
}

# Values are tried in the order listed.
SEARCH_SPACE = {
    "train_feature_at_layer": [8],
    "train_reg_weight_intra_class": [1],
    "train_reg_weight":       [1e-3, 5e-3, 1e-2],
    "train_reg_num_classes":  [10],
    "train_reg_num_sampling": [8],
}


def run_search():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile = os.path.join(LOG_DIR_SEARCH, f"{DATASET_NAME}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfile),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    keys = list(SEARCH_SPACE.keys())
    # trial_name -> {metric: [values across seeds]}
    trial_results = {}

    for i, values in enumerate(itertools.product(*SEARCH_SPACE.values())):
        params = dict(zip(keys, values))

        config = copy.deepcopy(BASE_EXPERIMENT_CONFIG)
        config.update(params)

        L   = params["train_feature_at_layer"]
        lam = params["train_reg_weight"]
        N   = params["train_reg_num_classes"]
        K   = params["train_reg_num_sampling"]
        trial_name = f"trial{i}_L{L}_lam{lam:.0e}_N{N}_K{K}"

        trial_results[trial_name] = {
            "params": params,
            "seeds": [],
            "mlp_faa": [],
            "mlp_ffm": [],
            "mlp_asa": [],
            "ncm_faa": [],
            "ncm_ffm": [],
            "ncm_asa": [],
        }

        for seed in SEEDS:
            logging.info(f"\n{'='*60}\nTrial {i}: {trial_name} | seed {seed}\n{'='*60}")

            result = run_single_experiment(DATASET_NAME, trial_name, config, seed)

            trial_results[trial_name]["seeds"].append(seed)
            trial_results[trial_name]["mlp_faa"].append(result.get("mlp_faa", 0.0))
            trial_results[trial_name]["mlp_ffm"].append(result.get("mlp_ffm", 0.0))
            trial_results[trial_name]["mlp_asa"].append(result.get("mlp_asa", 0.0))
            trial_results[trial_name]["ncm_faa"].append(result.get("ncm_faa", 0.0))
            trial_results[trial_name]["ncm_ffm"].append(result.get("ncm_ffm", 0.0))
            trial_results[trial_name]["ncm_asa"].append(result.get("ncm_asa", 0.0))

            logging.info(f"Trial {i} seed {seed} mlp_asa={result.get('mlp_asa', 0.0):.4f}")

        # Per-trial summary
        mlp_asa_vals = trial_results[trial_name]["mlp_asa"]
        logging.info(f"\n{'='*60}")
        logging.info(f"Trial {i} ({trial_name}) summary over {len(SEEDS)} seed(s):")
        logging.info(f"  MLP - ASA: {np.mean(mlp_asa_vals):.4f} ± {np.std(mlp_asa_vals):.4f}")
        logging.info(f"  MLP - FAA: {np.mean(trial_results[trial_name]['mlp_faa']):.4f} ± {np.std(trial_results[trial_name]['mlp_faa']):.4f}")
        logging.info(f"  MLP - FFM: {np.mean(trial_results[trial_name]['mlp_ffm']):.4f} ± {np.std(trial_results[trial_name]['mlp_ffm']):.4f}")
        ncm_asa_vals = trial_results[trial_name]["ncm_asa"]
        if any(v > 0.0 for v in ncm_asa_vals):
            logging.info(f"  NCM - ASA: {np.mean(ncm_asa_vals):.4f} ± {np.std(ncm_asa_vals):.4f}")
            logging.info(f"  NCM - FAA: {np.mean(trial_results[trial_name]['ncm_faa']):.4f} ± {np.std(trial_results[trial_name]['ncm_faa']):.4f}")
            logging.info(f"  NCM - FFM: {np.mean(trial_results[trial_name]['ncm_ffm']):.4f} ± {np.std(trial_results[trial_name]['ncm_ffm']):.4f}")

    # Final best-trial summary ranked by mean mlp_asa
    logging.info("\n" + "=" * 60)
    logging.info("SEARCH SUMMARY (ranked by mean mlp_asa)")
    logging.info("=" * 60)

    ranked = sorted(
        trial_results.items(),
        key=lambda x: np.mean(x[1]["mlp_asa"]),
        reverse=True,
    )

    for rank, (trial_name, data) in enumerate(ranked, 1):
        mean_asa = np.mean(data["mlp_asa"])
        std_asa  = np.std(data["mlp_asa"])
        logging.info(f"#{rank:2d} {trial_name}  mlp_asa={mean_asa:.4f} ± {std_asa:.4f}")
        for k, v in data["params"].items():
            logging.info(f"      {k}: {v}")

    best_name, best_data = ranked[0]
    logging.info("\nBest trial: " + best_name)
    logging.info(f"  mlp_asa : {np.mean(best_data['mlp_asa']):.4f} ± {np.std(best_data['mlp_asa']):.4f}")
    logging.info("  Params  :")
    for k, v in best_data["params"].items():
        logging.info(f"    {k}: {v}")


if __name__ == "__main__":
    run_search()
