import sys
import os

sys.path.append(".")
sys.path.append("..")

from pydantic import BaseModel
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from training.objects import PipelineOutput
import itertools
# import scienceplots

# plt.style.use(["science", "ieee", "grid"])


class CorrelationMetrics(BaseModel):
    pearson: float
    spearman: float
    kendalltau: float
    chatterjeexi: float


def compute_correlation_metrics(x: np.ndarray, y: np.ndarray):
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
