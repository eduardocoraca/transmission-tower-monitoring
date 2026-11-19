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
from training.objects import CosinePipelineOutput, MetaData
from training.constants import FS, CABLES
from torch.utils.data import DataLoader
from training.dataset import normalize_tensor
from training.models.ae_regularized import AECosineRegularized
import pytorch_lightning as pl
from training.evaluate import evaluate_ae_cosine
from training.objects import BandRange, Embedding
from datetime import timedelta
from copy import deepcopy
from datetime import datetime
from lib.wpt_utils import combine_wpts, build_index_per_level
from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint


def extract_feature_vectors(
    wpt_dataset: list[SampleWaveletPacket],
) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    cables = [1, 2, 3, 4]
    x_per_cable: dict[int, list[float]] = {cable: [] for cable in cables}
    t_per_cable: dict[int, list[float]] = {cable: [] for cable in cables}
    dates: list[datetime] = []
    for d in wpt_dataset:
        for c in cables:
            cable_data = getattr(d, f"sample_c{c}")
            combined = combine_wpts([cable_data.y, cable_data.z])
            x_per_cable[c].append(combined.energy_map)
            t_per_cable[c].append(cable_data.tension)
        dates.append(datetime.fromisoformat(d.sample_id))

    x_tensor = np.array([np.array(x) for x in x_per_cable.values()]).swapaxes(0, 1)
    t_tensor = np.array([np.array(t) for t in t_per_cable.values()]).swapaxes(0, 1)
    return x_tensor, t_tensor, dates


def normalize_by_max_per_level(x_tensor: np.ndarray, max_level: int) -> np.ndarray:
    num_cables = x_tensor.shape[1]
    x_copy = deepcopy(x_tensor)
    index_per_level = build_index_per_level(0, max_level)
    for c in range(num_cables):
        for _, ind in index_per_level.items():
            level_values = x_tensor[:, c][:, ind]
            level_max = np.max(level_values, axis=1)
            x_copy[:, c, ind] = x_tensor[:, c, ind] / level_max.reshape(-1, 1)
    return x_copy


def get_valid_index(
    f_min: float, f_max: float, min_level: int, max_level: int, fs: float
):
    valid_index = {}
    indexes = build_index_per_level(min_level, max_level)
    for level, ind in indexes.items():
        delta_f = (fs / 2) / (2**level)
        valid_index[level] = []
        for k, i in enumerate(ind):
            freq_range = BandRange(start=k * delta_f, end=(k + 1) * delta_f)
            if freq_range.has_intersection(BandRange(start=f_min, end=f_max)):
                valid_index[level].append(i)
    return valid_index


def run_pipeline(
    data_train: list[SampleWaveletPacket],
    data_valid: list[SampleWaveletPacket],
    params: dict,
    logger: pl.loggers.Logger,
    return_model: bool = False,
    data_test: Optional[list[SampleWaveletPacket]] = None,
) -> CosinePipelineOutput:
    # -- Get initial parameters --
    f_max = params["f_min"] + params["bandwidth"]
    max_level = data_train[0].sample_c1.y.max_level

    # -- Extract feature vectors --
    train_x_tensor, train_t_tensor, train_dates = extract_feature_vectors(data_train)
    valid_x_tensor, valid_t_tensor, valid_dates = extract_feature_vectors(data_valid)
    if data_test is not None:
        test_x_tensor, test_t_tensor, test_dates = extract_feature_vectors(data_test)

    # -- Normalize --
    # 1) max per band
    train_xn = normalize_by_max_per_level(train_x_tensor, max_level=max_level + 1)
    valid_xn = normalize_by_max_per_level(valid_x_tensor, max_level=max_level + 1)
    if data_test is not None:
        test_xn = normalize_by_max_per_level(test_x_tensor, max_level=max_level + 1)

    # 2) log
    # train_xn = np.log(train_xn)
    # valid_xn = np.log(valid_xn)
    # if data_test is not None:
    #     test_xn = np.log(test_xn)

    # -- Select frequency bands --
    valid_index = get_valid_index(
        f_min=params["f_min"],
        f_max=f_max,
        min_level=params["min_level"],
        max_level=params["max_level"],
        fs=params["fs"],
    )
    valid_index_flat = [item for sublist in valid_index.values() for item in sublist]
    temp_train = np.zeros((train_xn.shape[0], train_xn.shape[1], len(valid_index_flat)))
    temp_valid = np.zeros((valid_xn.shape[0], valid_xn.shape[1], len(valid_index_flat)))
    if data_test is not None:
        temp_test = np.zeros(
            (test_xn.shape[0], test_xn.shape[1], len(valid_index_flat))
        )

    for cable in [0, 1, 2, 3]:
        temp_train[:, cable] = train_xn[:, cable][:, valid_index_flat]
        temp_valid[:, cable] = valid_xn[:, cable][:, valid_index_flat]
        if data_test is not None:
            temp_test[:, cable] = test_xn[:, cable][:, valid_index_flat]
    train_xn, valid_xn = temp_train, temp_valid
    if data_test is not None:
        test_xn = temp_test

    # -- Dataloaders --
    train_dataloader: DataLoader = DataLoader(
        train_xn, batch_size=params["batch_size"], shuffle=True, num_workers=0
    )
    train_dataloader_no_shuffle: DataLoader = DataLoader(
        train_xn, batch_size=params["batch_size"], shuffle=False, num_workers=0
    )
    valid_dataloader: DataLoader = DataLoader(
        valid_xn, batch_size=params["batch_size"], shuffle=False, num_workers=0
    )
    if data_test is not None:
        test_dataloader: DataLoader = DataLoader(
            test_xn, batch_size=params["batch_size"], shuffle=False, num_workers=0
        )

    # -- AE Cosine Model --

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="ae_cosine_regularized",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )

    ae_model = AECosineRegularized(
        input_dim=train_xn.shape[2],
        lr=params["lr"],
        latent_dim=params["latent_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        activation=params["activation"],
        beta=params["beta"],
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=params["num_epochs"],
        accelerator="gpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        ae_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )

    ae_model.eval()
    model_path = checkpoint_callback.best_model_path
    logger.experiment.log_model("ae_cosine_regularized", model_path)
    # logger.experiment.log_model("vae_regularized", "../modelfile.h5")

    # -- Get evaluations --
    train_x, train_z, train_mse, train_rec = evaluate_ae_cosine(
        train_dataloader_no_shuffle, ae_model
    )
    valid_x, valid_z, valid_mse, valid_rec = evaluate_ae_cosine(
        valid_dataloader, ae_model
    )
    if data_test is not None:
        test_x, test_z, test_mse, test_rec = evaluate_ae_cosine(
            test_dataloader, ae_model
        )

    train_metadata = []
    for d, t in zip(train_dates, train_t_tensor):
        for cable in [0, 1, 2, 3]:
            train_metadata.append(
                MetaData(cable=str(cable), tension=t[cable], sampled_at=d)
            )

    valid_metadata = []
    for d, t in zip(valid_dates, valid_t_tensor):
        for cable in [0, 1, 2, 3]:
            valid_metadata.append(
                MetaData(cable=str(cable), tension=t[cable], sampled_at=d)
            )

    if data_test is not None:
        test_metadata = []
        for d, t in zip(test_dates, test_t_tensor):
            for cable in [0, 1, 2, 3]:
                test_metadata.append(
                    MetaData(cable=str(cable), tension=t[cable], sampled_at=d)
                )

    train_output = CosinePipelineOutput(
        metadata=[train_metadata],
        embeddings=train_z,
        inputs=train_x,
        reconstructions=train_rec,
        errors=train_mse,
        model=ae_model if return_model else None,
    )

    valid_output = CosinePipelineOutput(
        metadata=[valid_metadata],
        embeddings=valid_z,
        inputs=valid_x,
        reconstructions=valid_rec,
        errors=valid_mse,
    )

    test_output = None
    if data_test is not None:
        test_output = CosinePipelineOutput(
            metadata=[test_metadata],
            embeddings=test_z,
            inputs=test_x,
            reconstructions=test_rec,
            errors=test_mse,
        )

    return train_output, valid_output, test_output

    # -- Get embedding features --
    # train_embeddings = []
    # for z, t, d in zip(train_z, train_t_tensor, train_dates):
    #     for cable in [0, 1, 2, 3]:
    #         train_embeddings.append(
    #             Embedding(
    #                 cable=cable,
    #                 sampled_at=d,
    #                 tension=t[cable],
    #                 values=z[cable].tolist(),
    #             )
    #         )

    # valid_embeddings = []
    # for z, t, d in zip(valid_z, valid_t_tensor, valid_dates):
    #     for cable in [0, 1, 2, 3]:
    #         valid_embeddings.append(
    #             Embedding(
    #                 cable=cable,
    #                 sampled_at=d,
    #                 tension=t[cable],
    #                 values=z[cable].tolist(),
    #             )
    #         )

    # if data_test is not None:
    #     test_embeddings = []
    #     for z, t, d in zip(test_z, test_t_tensor, test_dates):
    #         for cable in [0, 1, 2, 3]:
    #             test_embeddings.append(
    #                 Embedding(
    #                     cable=cable,
    #                     sampled_at=d,
    #                     tension=t[cable],
    #                     values=z[cable].tolist(),
    #                 )
    #             )

    # train_df = pd.DataFrame([f.model_dump() for f in train_embeddings])
    # valid_df = pd.DataFrame([f.model_dump() for f in valid_embeddings])
    # if data_test is not None:
    #     test_df = pd.DataFrame([f.model_dump() for f in test_embeddings])

    # # -- Get baseline --
    # train_start = train_df.sampled_at.min()
    # train_end = train_start + timedelta(days=14)

    # valid_start = valid_df.sampled_at.min()
    # valid_end = valid_start + timedelta(days=14)

    # if data_test is not None:
    #     test_start = test_df.sampled_at.min()
    #     test_end = test_start + timedelta(days=14)

    # train_baseline_per_cable = (
    #     train_df.loc[train_df.sampled_at.isin(pd.date_range(train_start, train_end))]
    #     .groupby("cable")
    #     .agg({"values": lambda x: np.mean(np.vstack(x), axis=0).tolist()})
    #     .reset_index()
    # )
    # valid_baseline_per_cable = (
    #     valid_df.loc[valid_df.sampled_at.isin(pd.date_range(valid_start, valid_end))]
    #     .groupby("cable")
    #     .agg({"values": lambda x: np.mean(np.vstack(x), axis=0).tolist()})
    #     .reset_index()
    # )
    # if data_test is not None:
    #     test_baseline_per_cable = (
    #         test_df.loc[test_df.sampled_at.isin(pd.date_range(test_start, test_end))]
    #         .groupby("cable")
    #         .agg({"values": lambda x: np.mean(np.vstack(x), axis=0).tolist()})
    #         .reset_index()
    #     )

    # # -- Compute distance per cable --
    # centered_train_embeddings = []
    # centered_valid_embeddings = []
    # centered_test_embeddings = []

    # for train_embedding in train_embeddings:
    #     cable = train_embedding.cable
    #     baseline = train_baseline_per_cable.loc[
    #         train_baseline_per_cable["cable"] == cable
    #     ]["values"].item()
    #     difference = np.array(train_embedding.values) - np.array(baseline)
    #     dist = np.linalg.norm(difference)
    #     centered_train_embeddings.append(
    #         Embedding(
    #             cable=train_embedding.cable,
    #             sampled_at=train_embedding.sampled_at,
    #             tension=train_embedding.tension,
    #             values=[dist],
    #         )
    #     )

    # for valid_embedding in valid_embeddings:
    #     cable = valid_embedding.cable
    #     baseline = valid_baseline_per_cable.loc[
    #         valid_baseline_per_cable["cable"] == cable
    #     ]["values"].item()
    #     difference = np.array(valid_embedding.values) - np.array(baseline)
    #     dist = np.linalg.norm(difference)
    #     centered_valid_embeddings.append(
    #         Embedding(
    #             cable=valid_embedding.cable,
    #             sampled_at=valid_embedding.sampled_at,
    #             tension=valid_embedding.tension,
    #             values=[dist],
    #         )
    #     )

    # if data_test is not None:
    #     for test_embedding in test_embeddings:
    #         cable = test_embedding.cable
    #         baseline = test_baseline_per_cable.loc[
    #             test_baseline_per_cable["cable"] == cable
    #         ]["values"].item()
    #         difference = np.array(test_embedding.values) - np.array(baseline)
    #         dist = np.linalg.norm(difference)
    #         centered_test_embeddings.append(
    #             Embedding(
    #                 cable=test_embedding.cable,
    #                 sampled_at=test_embedding.sampled_at,
    #                 tension=test_embedding.tension,
    #                 values=[dist],
    #             )
    #         )

    # return PipelineOutput(
    #     train_vae_results=train_embeddings,
    #     valid_vae_results=valid_embeddings,
    #     test_vae_results=test_embeddings if data_test is not None else None,
    #     train_dist_results=centered_train_embeddings,
    #     valid_dist_results=centered_valid_embeddings,
    #     test_dist_results=(centered_test_embeddings if data_test is not None else None),
    #     model=ae_model if return_model else None,
    # )
