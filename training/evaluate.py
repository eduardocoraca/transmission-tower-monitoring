import torch
import numpy as np
from training.models.ae_regularized import (
    VAERegularized,
    AECosineRegularized,
)
from torch.utils.data import DataLoader

def evaluate_vae(dataloader: DataLoader, model: VAERegularized):
    """
    Evaluates a VAE model with cosine regularization on a given dataloader.
    Args:
        dataloader (DataLoader): PyTorch DataLoader providing input data batches.
        model (VAERegularized): VAE model with cosine regularization to be evaluated.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Stacked input data (x) as a NumPy array.
            - Stacked latent representations (z) as a NumPy array.
            - Stacked reconstructed data (x_rec) as a NumPy array.
            - Stacked mean squared error (mse) per sample as a NumPy array.
    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - The mean squared error is computed along the last axis (axis=2) for each batch.
    """

    z_list = []
    mse_list = []
    model.eval()
    x_rec_list = []
    x_list = []
    for x in dataloader:
        x = x.float()
        with torch.no_grad():
            x_rec, mu, z, logvar = model.forward(x)
        x_list.append(x)
        z_list.append(mu)
        x_rec_list.append(x_rec)
        mse = (x_rec - x).pow(2).mean(axis=2)
        mse_list.append(mse)

    return (
        np.vstack(x_list),
        np.vstack(z_list),
        np.vstack(mse_list),
        np.vstack(x_rec_list),
    )


def evaluate_ae_cosine(dataloader: DataLoader, model: AECosineRegularized):
    """
    Evaluates an autoencoder model with cosine regularization on a given dataloader.
    Args:
        dataloader (DataLoader): PyTorch DataLoader providing input data batches.
        model (AECosineRegularized): Autoencoder model with cosine regularization to be evaluated.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Stacked input data (x) as a NumPy array.
            - Stacked latent representations (z) as a NumPy array.
            - Stacked reconstructed data (x_rec) as a NumPy array.
            - Stacked mean squared error (mse) per sample as a NumPy array.
    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - The mean squared error is computed along the last axis (axis=2) for each batch.
    """

    z_list = []
    mse_list = []
    model.eval()
    x_rec_list = []
    x_list = []
    for x in dataloader:
        x = x.float()
        with torch.no_grad():
            x_rec, z = model.forward(x)
        x_list.append(x)
        z_list.append(z)
        x_rec_list.append(x_rec)
        mse = (x_rec - x).pow(2).mean(axis=2)
        mse_list.append(mse)

    return (
        np.vstack(x_list),
        np.vstack(z_list),
        np.vstack(mse_list),
        np.vstack(x_rec_list),
    )
