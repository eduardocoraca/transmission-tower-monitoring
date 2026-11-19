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
from train_vae_hmm_pipe import vae_hmm_pipeline
from training.objects import PipelineOutput
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scienceplots
from itertools import product
import time

plt.style.use(["science", "ieee", "grid"])

COMET_CONFIG = get_comet_config("COMET_PROJECT_VAE_HMM")

SEQ_1_PATH = Path("./data/wpts/sequence_1_db35")
SEQ_2_PATH = Path("./data/wpts/sequence_2_db35")
SEQ_3_PATH = Path("./data/wpts/sequence_3_db35")

data_seq_1 = load_wavelets(SEQ_1_PATH)
data_seq_2 = load_wavelets(SEQ_2_PATH)
# data_test = load_wavelets(SEQ_3_PATH)


def log_metrics(output: PipelineOutput, logger: pl.loggers.Logger):
    train_vae_results = pd.DataFrame([f.model_dump() for f in output.train_vae_results])
    valid_vae_results = pd.DataFrame([f.model_dump() for f in output.valid_vae_results])

    has_hmm_result = output.train_hmm_results is not None
    if has_hmm_result:
        train_hmm_results = pd.DataFrame(
            [f.model_dump() for f in output.train_hmm_results]
        )
        valid_hmm_results = pd.DataFrame(
            [f.model_dump() for f in output.valid_hmm_results]
        )

    for cable in train_vae_results.cable.unique():
        train_cable_df = train_vae_results.loc[train_vae_results["cable"] == cable]
        train_values = np.array(train_cable_df["values"].to_list())
        train_tension = np.array(train_cable_df["tension"].to_list())

        valid_cable_df = valid_vae_results.loc[valid_vae_results["cable"] == cable]
        valid_values = np.array(valid_cable_df["values"].to_list())
        valid_tension = np.array(valid_cable_df["tension"].to_list())

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

    if has_hmm_result:
        for cable in train_hmm_results.cable.unique():
            train_cable_df = train_hmm_results.loc[train_hmm_results["cable"] == cable]
            train_values = np.array(train_cable_df["values"].to_list()).reshape(-1)
            train_tension = np.array(train_cable_df["tension"].to_list()).reshape(-1)
            train_sampled_at = np.array(train_cable_df["sampled_at"].to_list()).reshape(
                -1
            )

            valid_cable_df = valid_hmm_results.loc[valid_hmm_results["cable"] == cable]
            valid_values = np.array(valid_cable_df["values"].to_list()).reshape(-1)
            valid_tension = np.array(valid_cable_df["tension"].to_list()).reshape(-1)
            valid_sampled_at = np.array(valid_cable_df["sampled_at"].to_list()).reshape(
                -1
            )

            fig = plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.scatter(train_sampled_at, train_values, c=train_tension, cmap="seismic")
            plt.title("Training HMM States")
            plt.colorbar(label="Tension (N)")
            plt.xticks(rotation=30)
            plt.ylabel("State")

            plt.subplot(1, 2, 2)
            plt.scatter(valid_sampled_at, valid_values, c=valid_tension, cmap="seismic")
            plt.title("Validation HMM States")
            plt.colorbar(label="Tension (N)")
            plt.xticks(rotation=30)
            plt.ylabel("State")

            logger.experiment.log_figure(
                figure=fig, figure_name=f"hmm_states_cable_{cable}"
            )
            plt.close("all")

    for eval in output.train_vae_eval:
        logger.experiment.log_metric(
            f"{eval.cable}_reconstruction_score", eval.reconstruction_score
        )
        logger.experiment.log_metric(
            f"{eval.cable}_classification_accuracy", eval.classification_accuracy
        )

    for eval in output.valid_vae_eval:
        logger.experiment.log_metric(
            f"{eval.cable}_val_reconstruction_score", eval.reconstruction_score
        )
        logger.experiment.log_metric(
            f"{eval.cable}_val_classification_accuracy", eval.classification_accuracy
        )

    if has_hmm_result:
        for eval in output.train_hmm_eval:
            logger.experiment.log_metric(f"{eval.cable}_hmm_score", eval.hmm_score)

        for eval in output.valid_hmm_eval:
            logger.experiment.log_metric(f"{eval.cable}_val_hmm_score", eval.hmm_score)

    plt.close("all")


data_train, data_valid = data_seq_2, data_seq_1


# T1
# param_combinations = list(
#     product(
#         beta := [0.01, 0.1, 0.5],
#         hmm_components := [6, 8],
#         latent_dim := [2, 4, 6],
#         num_bins := [5, 10],
#         f_min := [10, 48, 64],
#         bandwidth := [12, 24, 48],
#         max_level := [4, 5, 6],
#     )
# )

# T2
# param_combinations = list(
#     product(
#         beta := [0.01, 0.1, 0.5],
#         hmm_components := [6],
#         latent_dim := [2, 4, 6],
#         num_bins := [5, 10],
#         f_min := [10, 48, 64],
#         bandwidth := [12, 24, 48],
#         max_level := [4, 5, 6],
#     )
# )

# T3
param_combinations = list(
    product(
        beta := [0.01, 0.1, 0.5],
        hmm_components := [6],
        latent_dim := [2],
        num_bins := [10],
        f_min := [10, 24, 48, 64],
        bandwidth := [12, 24, 48],
        min_level := [1, 2, 3],
        max_level := [4, 5, 6, 7],
    )
)

# filter out parameter with min_level >= max_level

param_combinations = [p for p in param_combinations if p[6] < p[7]]

fixed_params = {
    "normalize": True,
    "batch_size": 128,
    "num_epochs": 150,
    "lr": 1e-3,
    "hidden_dim": 16,
    "num_layers": 3,
    "activation": "relu",
    "hmm_transition_matrix": "left-right",
}

for k, combination in enumerate(param_combinations):
    print(k)

    (
        beta,
        hmm_components,
        latent_dim,
        num_bins,
        f_min,
        bandwidth,
        min_level,
        max_level,
    ) = combination
    variable_params = {
        "hmm_components": hmm_components,
        "latent_dim": latent_dim,
        "num_bins": num_bins,
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
        output = vae_hmm_pipeline(data_train, data_valid, params, logger)
        log_metrics(output, logger)
        time.sleep(2)
    except Exception as e:
        print(e)
        continue
    logger.experiment.end()
