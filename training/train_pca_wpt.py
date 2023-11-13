import sys

sys.path.append(".")

import argparse

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import get_filename

from classes.feature import WPTFeatureExtractor
from classes.loader.loader import Loader


def plot_features(x, y, xlabel=None):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()
    num_feat = y.shape[1]
    for i in range(num_feat):
        ax.plot(x, y[:, i])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return fig


def plot_2d_features(c, y):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()
    ax.scatter(y[:, 0], y[:, 1], c=c)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    return fig


PATH_TO_TRAIN_DATASET = r"./raw/sequence_2/"
PATH_TO_VALID_DATASET = r"./raw/sequence_1/"
CABLES = [1, 2, 3, 4]

# 1) Set model parameters
parser = argparse.ArgumentParser(
    description="Train PCA model with WPT feature extraction."
)
parser.add_argument("-f", "--family", type=str)
parser.add_argument("-l", "--level", type=int)
parser.add_argument("-c", "--cable", type=int)
parser.add_argument("-d", "--dimension", type=int)

args = parser.parse_args()

LEVEL = args.level
FAMILY = args.family
CABLE = args.cable
DIM = args.dimension

# 2) extract signal features
wpt_transformer = WPTFeatureExtractor(
    family=FAMILY, level=LEVEL, normalize=True, axis_to_combine=["y", "z"]
)
train_loader = Loader(PATH_TO_TRAIN_DATASET, wpt_transformer)
valid_loader = Loader(PATH_TO_VALID_DATASET, wpt_transformer)

train_dfs_per_cable = train_loader.load_from_folder()
valid_dfs_per_cable = valid_loader.load_from_folder()

# 3) select relevant arrays
cables_to_run = []
if CABLE is not None:
    cables_to_run.append(CABLE)
else:
    cables_to_run = CABLES

train_datasets = []
valid_datasets = []
train_tensions = []  # used for plotting
valid_tensions = []  # used for plotting
for c in cables_to_run:
    train_df = train_dfs_per_cable[f"c{c}"].sort_values("sampled_at")
    valid_df = valid_dfs_per_cable[f"c{c}"].sort_values("sampled_at")
    train_datasets.append(np.vstack(train_df["features"].to_list()).squeeze())
    valid_datasets.append(np.vstack(valid_df["features"].to_list()).squeeze())
    train_tensions.append(train_df["tension"].to_list())
    valid_tensions.append(valid_df["tension"].to_list())

# 4) train PCA model for the selected cables
for i, (train_dataset, valid_dataset) in enumerate(zip(train_datasets, valid_datasets)):
    c = cables_to_run[i]
    with mlflow.start_run() as run:
        params = {"family": FAMILY, "level": LEVEL, "dim": DIM}
        filename = get_filename("pcawpt", cable=str(c), **params)

        # fit PCA
        pca_model = PCA(DIM)
        pca_model.fit(train_dataset)

        y_train = pca_model.transform(train_dataset)
        y_valid = pca_model.transform(valid_dataset)

        # evaluate with linear regression
        true_train_tensions = np.array(train_tensions[i]).reshape(-1, 1)
        true_valid_tensions = np.array(valid_tensions[i]).reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(y_train, true_train_tensions)
        t_train = linear_model.predict(y_train)
        t_valid = linear_model.predict(y_valid)

        mse_train = mean_squared_error(true_train_tensions, t_train)
        r2_train = r2_score(true_train_tensions, t_train)

        mse_valid = mean_squared_error(true_valid_tensions, t_valid)
        r2_valid = r2_score(true_valid_tensions, t_valid)

        # evaluate with PCA reconstruction
        rec_train = pca_model.inverse_transform(y_train)
        rec_valid = pca_model.inverse_transform(y_valid)
        mae_rec_train = (abs(train_dataset - rec_train)).sum(axis=1).mean()
        mae_rec_valid = (abs(valid_dataset - rec_valid)).sum(axis=1).mean()
        mae_ratio = mae_rec_valid / mae_rec_train

        # figures
        fig_train = plot_features(np.arange(len(y_train)), y_train)
        fig_valid = plot_features(np.arange(len(y_valid)), y_valid)

        fig_train_tension = plot_features(
            true_train_tensions.squeeze(), y_train, xlabel="Tension (N)"
        )
        fig_valid_tension = plot_features(
            true_valid_tensions.squeeze(), y_valid, xlabel="Tension (N)"
        )
        if DIM == 2:
            fig_train_2d_features = plot_2d_features(
                true_train_tensions.squeeze(), y_train
            )
            fig_valid_2d_features = plot_2d_features(
                true_valid_tensions.squeeze(), y_valid
            )

        # mlflow
        signature = infer_signature(valid_dataset, y_valid)
        mlflow.log_metrics(
            {
                "mse_train": mse_train,
                "r2_train": r2_train,
                "mae_rec_train": mae_rec_train,
                "mse_valid": mse_valid,
                "r2_valid": r2_valid,
                "mae_rec_valid": mae_rec_valid,
                "mae_rec_ratio": mae_ratio,
            }
        )
        mlflow.log_params(params)
        mlflow.log_param("cable", c)
        mlflow.log_figure(fig_train, "train_features.png")
        mlflow.log_figure(fig_valid, "valid_features.png")
        mlflow.log_figure(fig_train_tension, "train_features_tension.png")
        mlflow.log_figure(fig_valid_tension, "valid_features_tension.png")
        if DIM == 2:
            mlflow.log_figure(fig_train_2d_features, "train_2d_features.png")
            mlflow.log_figure(fig_valid_2d_features, "valid_2d_features.png")
        plt.close("all")

        mlflow.sklearn.log_model(
            sk_model=pca_model,
            artifact_path="pcawpt-model",
            signature=signature,
            registered_model_name=filename,
        )
