import sys
from training.comet_config import get_comet_config
from dotenv import load_dotenv

load_dotenv()

sys.path.append(".")
sys.path.append("..")

import matplotlib

matplotlib.use("Agg")
from pytorch_lightning.loggers import CometLogger
from pathlib import Path
from datetime import datetime, timedelta
from lib.wpt_utils import load_wavelets
from training_pipeline import run_pipeline
from training.objects import PipelineOutput
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scienceplots
from itertools import product
import time
from pydantic import BaseModel
import itertools
import scipy.stats

plt.style.use(["science", "ieee", "grid"])

COMET_CONFIG = get_comet_config("COMET_PROJECT_VAE")

SEQ_1_PATH = Path("./data/wpts/sequence_1_db35")
SEQ_2_PATH = Path("./data/wpts/sequence_2_db35")
SEQ_3_PATH = Path("./data/wpts/sequence_3_db35")

data_seq_1 = load_wavelets(SEQ_1_PATH)
data_seq_2 = load_wavelets(SEQ_2_PATH)
# data_test = load_wavelets(SEQ_3_PATH)


class CorrelationMetrics(BaseModel):
    pearson: float
    spearman: float
    kendalltau: float
    chatterjeexi: float


def compute_correlation_metrics(x: np.ndarray, y: np.ndarray) -> CorrelationMetrics:
    pearson = scipy.stats.pearsonr(x, y)[0]
    spearman = scipy.stats.spearmanr(x, y)[0]
    kendalltau = scipy.stats.kendalltau(x, y)[0]
    chatterjeexi = scipy.stats.chatterjeexi(x, y)[0]
    return CorrelationMetrics(
        pearson=pearson,
        spearman=spearman,
        kendalltau=kendalltau,
        chatterjeexi=chatterjeexi,
    )


def log_metrics(output: PipelineOutput, logger: pl.loggers.Logger):
    train_vae_results = pd.DataFrame([f.model_dump() for f in output.train_vae_results])
    valid_vae_results = pd.DataFrame([f.model_dump() for f in output.valid_vae_results])

    train_dist_results = pd.DataFrame(
        [f.model_dump() for f in output.train_dist_results]
    )
    valid_dist_results = pd.DataFrame(
        [f.model_dump() for f in output.valid_dist_results]
    )

    for cable in train_vae_results.cable.unique():
        train_cable_df = train_vae_results.loc[train_vae_results["cable"] == cable]
        train_values = np.array(train_cable_df["values"].to_list())
        train_tension = np.array(train_cable_df["tension"].to_list())

        train_dist_df = train_dist_results.loc[train_dist_results["cable"] == cable]
        train_dist = np.array(train_dist_df["values"].to_list()).reshape(-1)
        train_dist_tension = np.array(train_dist_df["tension"].to_list()).reshape(-1)

        valid_cable_df = valid_vae_results.loc[valid_vae_results["cable"] == cable]
        valid_values = np.array(valid_cable_df["values"].to_list())
        valid_tension = np.array(valid_cable_df["tension"].to_list())

        valid_dist_df = valid_dist_results.loc[valid_dist_results["cable"] == cable]
        valid_dist = np.array(valid_dist_df["values"].to_list()).reshape(-1)
        valid_dist_tension = np.array(valid_dist_df["tension"].to_list()).reshape(-1)

        # Train vs Valid Embeddings
        fig = plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.scatter(
            train_values[:, 0], train_values[:, 1], c=train_tension, cmap="seismic"
        )
        plt.xlabel("Embedding 1")
        plt.ylabel("Embedding 2")
        plt.title("Training Embeddings")
        plt.colorbar(label="Tension (N)")

        plt.subplot(1, 2, 2)
        plt.scatter(
            valid_values[:, 0], valid_values[:, 1], c=valid_tension, cmap="seismic"
        )
        plt.xlabel("Embedding 1")
        plt.ylabel("Embedding 2")
        plt.title("Validation Embeddings")
        plt.colorbar(label="Tension (N)")
        logger.experiment.log_figure(
            figure=fig, figure_name=f"embeddings_cable_{cable}"
        )

        plt.close("all")

        # Embeddings vs Tension
        fig = plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.plot(train_tension, train_values)
        plt.xlabel("Tension (N)")
        plt.ylabel("Embedding")
        plt.title("Training Embeddings")
        plt.legend([f"Emb. {i + 1}" for i in range(train_values.shape[1])])

        plt.subplot(1, 2, 2)
        plt.plot(valid_tension, valid_values)
        plt.xlabel("Tension (N)")
        plt.ylabel("Embedding")
        plt.title("Validation Embeddings")
        plt.legend([f"Emb. {i + 1}" for i in range(valid_values.shape[1])])
        logger.experiment.log_figure(
            figure=fig, figure_name=f"embeddings_vs_tension_cable_{cable}"
        )
        plt.close("all")

        # Distance vs Tension
        fig = plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.scatter(train_dist_tension, train_dist, color="k")
        plt.xlabel("Tension (N)")
        plt.ylabel("Distance")
        plt.title("Training Distances")

        plt.subplot(1, 2, 2)
        plt.scatter(valid_dist_tension, valid_dist, color="k")
        plt.xlabel("Tension (N)")
        plt.ylabel("Distance")
        plt.title("Validation Distances")
        logger.experiment.log_figure(
            figure=fig, figure_name=f"distance_vs_tension_cable_{cable}"
        )
        plt.close("all")

        # Distance vs Tension
        fig = plt.figure(figsize=(4, 3))
        plt.scatter(train_dist_tension, train_dist, color="k")
        plt.scatter(valid_dist_tension, valid_dist, color="r")
        plt.xlabel("Tension (N)")
        plt.ylabel("Distance")
        plt.legend(["Training", "Validation"])

        logger.experiment.log_figure(
            figure=fig, figure_name=f"full_distance_vs_tension_cable_{cable}"
        )
        plt.close("all")

    plt.close("all")

    # training correlation metrics
    pair_combinations = list(
        itertools.combinations(train_dist_results.cable.unique(), 2)
    )
    for comb in pair_combinations:
        x = np.array(
            train_dist_results.loc[train_dist_results["cable"] == comb[0]][
                "values"
            ].to_list()
        ).reshape(-1)
        y = np.array(
            train_dist_results.loc[train_dist_results["cable"] == comb[1]][
                "values"
            ].to_list()
        ).reshape(-1)
        corr_metrics = compute_correlation_metrics(x, y)
        for metric_name in corr_metrics.model_fields.keys():
            logger.experiment.log_metric(
                f"train_{comb[0]}_vs_{comb[1]}_{metric_name}",
                getattr(corr_metrics, metric_name),
            )

    # validation correlation metrics
    pair_combinations = list(
        itertools.combinations(valid_dist_results.cable.unique(), 2)
    )
    for comb in pair_combinations:
        x = np.array(
            valid_dist_results.loc[valid_dist_results["cable"] == comb[0]][
                "values"
            ].to_list()
        ).reshape(-1)
        y = np.array(
            valid_dist_results.loc[valid_dist_results["cable"] == comb[1]][
                "values"
            ].to_list()
        ).reshape(-1)
        corr_metrics = compute_correlation_metrics(x, y)
        for metric_name in corr_metrics.model_fields.keys():
            logger.experiment.log_metric(
                f"valid_{comb[0]}_vs_{comb[1]}_{metric_name}",
                getattr(corr_metrics, metric_name),
            )


data_train, data_valid = data_seq_2, data_seq_1

param_combinations = list(
    product(
        beta := [1e-3, 1e-2, 1e-1],
        kl_weight := [2,3,4,5],
        latent_dim := [2, 4, 8, 16],
        f_min := [7],
        bandwidth := [113],
        min_level := [3],
        max_level := [5, 6, 7],
    )
)

# filter out parameter with min_level >= max_level

fixed_params = {
    "normalize": True,
    "batch_size": 128,
    "num_epochs": 400,
    "lr": 1e-3,
    "hidden_dim": 32,
    "num_layers": 3,
    "activation": "relu",
    "fs": 256,
}

for k, combination in enumerate(param_combinations):
    (
        beta,
        kl_weight,
        latent_dim,
        f_min,
        bandwidth,
        min_level,
        max_level,
    ) = combination

    variable_params = {
        "kl_weight": kl_weight,
        "latent_dim": latent_dim,
        "f_min": f_min,
        "bandwidth": bandwidth,
        "min_level": min_level,
        "max_level": max_level,
        "beta": beta,
        "num_comb": k,
    }
    params = {**fixed_params, **variable_params}
    logger = CometLogger(
        api_key=COMET_CONFIG["api_key"],
        project_name=COMET_CONFIG["project_name"],
        workspace=COMET_CONFIG["workspace"],
    )
    logger.log_hyperparams(params)
    try:
        output, train_results, valid_results, test_results = run_pipeline(
            data_train, data_valid, params, logger
        )
        log_metrics(output, logger)
        # time.sleep(2)
    except Exception as e:
        print(e)
        continue
    logger.experiment.end()
