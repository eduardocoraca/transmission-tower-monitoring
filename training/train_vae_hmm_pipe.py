import sys
from pytorch_lightning.loggers import CometLogger

sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from classes.sample.sample_classes import SampleWaveletPacket
from training.feature_processing import (
    new_extract_features,
)
from training.objects import ScalerParams, NewSpectralFeatures, PipelineOutput
from training.constants import FS, CABLES
from training.dataset import (
    build_wpt_dataset_from_features,
    new_build_tensor_stacked_dataset,
)
from torch.utils.data import TensorDataset, DataLoader
from training.dataset import normalize_tensor
from training.models.vae_model import VAE
import pytorch_lightning as pl
from training.evaluate import evaluate_vae
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LogisticRegression


def log_metrics(output: PipelineOutput, logger: pl.loggers.Logger):
    train_vae_results = pd.DataFrame([f.model_dump() for f in output.train_vae_results])
    valid_vae_results = pd.DataFrame([f.model_dump() for f in output.valid_vae_results])

    train_hmm_results = pd.DataFrame([f.model_dump() for f in output.train_hmm_results])
    valid_hmm_results = pd.DataFrame([f.model_dump() for f in output.valid_hmm_results])

    for cable in CABLES:
        train_cable_df = train_vae_results.loc[train_vae_results["cable"] == cable]
        train_values = np.array(train_cable_df["values"].to_list())
        train_tension = np.array(train_cable_df["tension"].to_list())

        valid_cable_df = valid_vae_results.loc[valid_vae_results["cable"] == cable]
        valid_values = np.array(valid_cable_df["values"].to_list())
        valid_tension = np.array(valid_cable_df["tension"].to_list())

        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(train_values[:, 0], train_values[:, 1], c=train_tension, cmap="jet")
        plt.title("Training Embeddings")
        plt.colorbar(label="Tension (N)")
        plt.subplot(1, 2, 2)
        plt.scatter(valid_values[:, 0], valid_values[:, 1], c=valid_tension, cmap="jet")
        plt.title("Validation Embeddings")
        plt.colorbar(label="Tension (N)")
        logger.experiment.log_figure(
            figure=fig, figure_name=f"embeddings_cable_{cable}"
        )
        plt.close()

    for cable in CABLES:
        train_cable_df = train_hmm_results.loc[train_hmm_results["cable"] == cable]
        train_values = np.array(train_cable_df["values"].to_list()).reshape(-1)
        train_tension = np.array(train_cable_df["tension"].to_list()).reshape(-1)
        train_sampled_at = np.array(train_cable_df["sampled_at"].to_list()).reshape(-1)

        valid_cable_df = valid_hmm_results.loc[valid_hmm_results["cable"] == cable]
        valid_values = np.array(valid_cable_df["values"].to_list()).reshape(-1)
        valid_tension = np.array(valid_cable_df["tension"].to_list()).reshape(-1)
        valid_sampled_at = np.array(valid_cable_df["sampled_at"].to_list()).reshape(-1)

        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(train_sampled_at, train_values, c=train_tension, cmap="jet")
        plt.title("Training HMM States")
        plt.colorbar(label="Tension (N)")
        plt.xticks(rotation=30)
        plt.subplot(1, 2, 2)
        plt.scatter(valid_sampled_at, valid_values, c=valid_tension, cmap="jet")
        plt.title("Validation HMM States")
        plt.colorbar(label="Tension (N)")
        plt.xticks(rotation=30)
        logger.experiment.log_figure(
            figure=fig, figure_name=f"hmm_states_cable_{cable}"
        )
        plt.close()

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

    for eval in output.train_hmm_eval:
        logger.experiment.log_metric(f"{eval.cable}_hmm_score", eval.hmm_score)

    for eval in output.valid_hmm_eval:
        logger.experiment.log_metric(f"{eval.cable}_val_hmm_score", eval.hmm_score)


def vae_hmm_pipeline(
    data_train: list[SampleWaveletPacket],
    data_valid: list[SampleWaveletPacket],
    params: dict,
    logger: pl.loggers.Logger,
) -> PipelineOutput:
    # -- Get features --
    f_max = params["f_min"] + params["bandwidth"]

    train_features: list[NewSpectralFeatures] = []
    for d in data_train:
        train_features += new_extract_features(
            d,
            f_min=params["f_min"],
            f_max=f_max,
            fs=FS,
            min_level=params["min_level"],
            max_level=params["max_level"],
            normalize=params["normalize"],
        )

    valid_features: list[NewSpectralFeatures] = []
    for d in data_valid:
        valid_features += new_extract_features(
            d,
            f_min=params["f_min"],
            f_max=f_max,
            fs=FS,
            min_level=params["min_level"],
            max_level=params["max_level"],
            normalize=params["normalize"],
        )

    # -- Remove cable 4 data --
    train_features = [f for f in train_features if f.cable != "c4"]
    valid_features = [f for f in valid_features if f.cable != "c4"]

    if params["num_bins"] == 0:
        num_bins = 10
    else:
        num_bins = params["num_bins"]
    # -- Build dataset --
    train_wpt_dataset = build_wpt_dataset_from_features(
        train_features, num_bins=num_bins
    )
    valid_wpt_dataset = build_wpt_dataset_from_features(
        valid_features, num_bins=num_bins
    )

    # -- Build tensor dataset --
    raw_train_dataset = new_build_tensor_stacked_dataset(train_wpt_dataset)
    raw_valid_dataset = new_build_tensor_stacked_dataset(valid_wpt_dataset)

    # -- Normalize --
    scaler_params = ScalerParams(
        mu=raw_train_dataset.tensors[0].mean(dim=0).reshape(-1).tolist(),
        sigma=raw_train_dataset.tensors[0].std(dim=0).reshape(-1).tolist(),
    )

    train_dataset = TensorDataset(
        normalize_tensor(raw_train_dataset.tensors[0], scaler_params),
        raw_train_dataset.tensors[1],
        raw_train_dataset.tensors[2],
    )

    valid_dataset = TensorDataset(
        normalize_tensor(raw_valid_dataset.tensors[0], scaler_params),
        raw_valid_dataset.tensors[1],
        raw_valid_dataset.tensors[2],
    )

    # -- Dataloaders --
    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0
    )

    # -- VAE Model --
    vae_model = VAE(
        input_dim=raw_train_dataset.tensors[0].shape[1],
        lr=params["lr"],
        beta=params["beta"],
        latent_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        activation=params["activation"],
        num_classes=num_bins,
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=params["num_epochs"],
        accelerator="gpu",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        vae_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )

    vae_model.eval()

    # -- Get embedding features --
    train_embeddings, train_vae_eval = evaluate_vae(
        train_wpt_dataset, vae_model, scaler_params
    )
    valid_embeddings, valid_vae_eval = evaluate_vae(
        valid_wpt_dataset, vae_model, scaler_params
    )

    train_df = pd.DataFrame([f.model_dump() for f in train_embeddings])
    valid_df = pd.DataFrame([f.model_dump() for f in valid_embeddings])

    # -- Get baseline --
    train_start = train_df.sampled_at.min()
    train_end = train_start + timedelta(days=7)

    valid_start = valid_df.sampled_at.min()
    valid_end = valid_start + timedelta(days=7)

    train_baseline_per_cable = (
        train_df.loc[train_df.sampled_at.isin(pd.date_range(train_start, train_end))]
        .groupby("cable")
        .agg({"values": lambda x: np.mean(np.vstack(x), axis=0).tolist()})
        .reset_index()
    )
    valid_baseline_per_cable = (
        valid_df.loc[valid_df.sampled_at.isin(pd.date_range(valid_start, valid_end))]
        .groupby("cable")
        .agg({"values": lambda x: np.mean(np.vstack(x), axis=0).tolist()})
        .reset_index()
    )

    # -- Center embeddings per cable --
    centered_train_embeddings = []
    centered_valid_embeddings = []

    for train_embedding in train_embeddings:
        cable = train_embedding.cable
        baseline = train_baseline_per_cable.loc[
            train_baseline_per_cable["cable"] == cable
        ]["values"].item()
        centered = np.array(train_embedding.values) - np.array(baseline)
        centered_train_embeddings.append(
            NewSpectralFeatures(
                cable=train_embedding.cable,
                sampled_at=train_embedding.sampled_at,
                tension=train_embedding.tension,
                values=centered.tolist(),
                label=train_embedding.label,
            )
        )

    for valid_embedding in valid_embeddings:
        cable = valid_embedding.cable
        baseline = valid_baseline_per_cable.loc[
            valid_baseline_per_cable["cable"] == cable
        ]["values"].item()
        centered = np.array(valid_embedding.values) - np.array(baseline)
        centered_valid_embeddings.append(
            NewSpectralFeatures(
                cable=valid_embedding.cable,
                sampled_at=valid_embedding.sampled_at,
                tension=valid_embedding.tension,
                values=centered.tolist(),
                label=valid_embedding.label,
            )
        )

    centered_train_df = pd.DataFrame(
        [f.model_dump() for f in centered_train_embeddings]
    )
    centered_valid_df = pd.DataFrame(
        [f.model_dump() for f in centered_valid_embeddings]
    )

    # -- Fit Logistic Regression --
    for cable in centered_train_df.cable.unique():
        train_cable_df = centered_train_df.loc[centered_train_df["cable"] == cable]
        valid_cable_df = centered_valid_df.loc[centered_valid_df["cable"] == cable]

        lr_model = LogisticRegression()
        x_train = np.array(train_cable_df["values"].to_list())
        y_train = np.array(train_cable_df["label"].to_list()) - 1
        lr_model.fit(x_train, y_train)
        y_pred_train = lr_model.predict(x_train)
        acc_train = sum(y_pred_train == y_train) / len(y_train)

        x_valid = np.array(valid_cable_df["values"].to_list())
        y_valid = np.array(valid_cable_df["label"].to_list()) - 1
        y_pred_valid = lr_model.predict(x_valid)
        acc_valid = sum(y_pred_valid == y_valid) / len(y_valid)

        for eval in train_vae_eval:
            if eval.cable == cable:
                eval.classification_accuracy = acc_train

        for eval in valid_vae_eval:
            if eval.cable == cable:
                eval.classification_accuracy = acc_valid

    return PipelineOutput(
        train_vae_eval=train_vae_eval,
        train_vae_results=train_embeddings,
        valid_vae_eval=valid_vae_eval,
        valid_vae_results=valid_embeddings,
    )

    # -- Get HMM dataset --
    # train_df_grouped = (
    #     centered_train_df.groupby([centered_train_df["sampled_at"].dt.date, "cable"])
    #     .agg(
    #         {
    #             "tension": "mean",
    #             "values": lambda x: np.mean(np.vstack(x), axis=0).tolist(),
    #         }
    #     )
    #     .reset_index()
    # )

    # valid_df_grouped = (
    #     centered_valid_df.groupby([centered_valid_df["sampled_at"].dt.date, "cable"])
    #     .agg(
    #         {
    #             "tension": "mean",
    #             "values": lambda x: np.mean(np.vstack(x), axis=0).tolist(),
    #         }
    #     )
    #     .reset_index()
    # )
    # # Grouped train embeddings
    # train_embeddings_grouped = [
    #     NewSpectralFeatures(**row) for _, row in train_df_grouped.iterrows()
    # ]
    # valid_embeddings_grouped = [
    #     NewSpectralFeatures(**row) for _, row in valid_df_grouped.iterrows()
    # ]

    # train_embeddings_grouped = [
    #     NewSpectralFeatures(**row) for _, row in train_df_grouped.iterrows()
    # ]
    # valid_embeddings_grouped = [
    #     NewSpectralFeatures(**row) for _, row in valid_df_grouped.iterrows()
    # ]

    # train_seq = []
    # train_lens = []
    # valid_seq = []
    # valid_lens = []
    # for cable in train_df_grouped.cable.unique():
    #     train_df_cable = train_df_grouped.loc[train_df_grouped["cable"] == cable]
    #     valid_df_cable = valid_df_grouped.loc[valid_df_grouped["cable"] == cable]

    #     train_df_cable = train_df_cable.sort_values(by="sampled_at")
    #     valid_df_cable = valid_df_cable.sort_values(by="sampled_at")

    #     train_seq.append(np.array(train_df_cable["values"].to_list()))
    #     train_lens.append(train_df_cable["values"].shape[0])

    #     valid_seq.append(np.array(valid_df_cable["values"].to_list()))
    #     valid_lens.append(valid_df_cable["values"].shape[0])

    # train_seq_stacked = np.vstack(train_seq)

    # # -- Find initial guess for the means --
    # min_date = train_df_grouped.sampled_at.min()
    # max_date = train_df_grouped.sampled_at.max()
    # range_date = max_date - min_date
    # group_size = range_date // params["hmm_components"]
    # group_days = group_size.days
    # train_df_grouped["sampled_at"] = pd.to_datetime(train_df_grouped["sampled_at"])
    # grouped_df = (
    #     train_df_grouped.groupby(
    #         [
    #             pd.Grouper(key="sampled_at", freq=f"{group_days}D"),
    #         ]
    #     )
    #     .agg({"values": lambda x: np.median(np.vstack(x), axis=0).tolist()})
    #     .reset_index()
    # )

    # initial_means = []
    # for i in range(params["hmm_components"]):
    #     values = np.array(grouped_df.iloc[0]["values"])
    #     initial_means.append(values)
    # initial_means = np.array(initial_means)

    # # -- Train HMM --
    # hmm = GaussianHMM(
    #     n_components=params["hmm_components"],
    #     covariance_type="full",
    #     init_params="sc",
    #     params="stmc",
    #     random_state=42,
    # )
    # hmm.means_ = initial_means
    # if params["hmm_transition_matrix"] == "left-right":
    #     hmm.transmat_ = build_left_right_transition_matrix(hmm.n_components)

    # # hmm.means_prior = np.array(train_df_grouped["values"].to_list())
    # try:
    #     hmm.fit(train_seq_stacked, lengths=train_lens)
    # except Exception as e:
    #     print(e)
    #     return PipelineOutput(
    #         train_vae_eval=train_vae_eval,
    #         train_vae_results=train_embeddings,
    #         valid_vae_eval=valid_vae_eval,
    #         valid_vae_results=valid_embeddings,
    #     )

    # ## -- Get scores --
    # train_hmm_results, train_hmm_eval = evaluate_hmm(train_embeddings_grouped, hmm)
    # valid_hmm_results, valid_hmm_eval = evaluate_hmm(valid_embeddings_grouped, hmm)

    # output = PipelineOutput(
    #     train_vae_eval=train_vae_eval,
    #     train_vae_results=train_embeddings,
    #     valid_vae_eval=valid_vae_eval,
    #     valid_vae_results=valid_embeddings,
    #     train_hmm_eval=train_hmm_eval,
    #     train_hmm_results=train_hmm_results,
    #     valid_hmm_eval=valid_hmm_eval,
    #     valid_hmm_results=valid_hmm_results,
    # )

    return output
