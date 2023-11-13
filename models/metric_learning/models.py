from datetime import datetime

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from .model_classes import DataLoader, TripletDataset


class MLPFeatureReducer(torch.nn.Module):
    def __init__(self, N: int, d: int, hidden: int):
        super().__init__()
        self.N = N
        self.d = d
        self.hidden = hidden
        self.model = torch.nn.Sequential(
            torch.nn.Linear(N, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, d),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CNNFeatureReducer(torch.nn.Module):
    def __init__(self, N: int, d: int):
        super().__init__()

        self.N = N
        self.d = d

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear((self.N // 8) * 3, d),
        )

    def __call__(self, x: torch.Tensor):
        return self.model(x.unsqueeze(1))


class FeatureReducerScikit(TransformerMixin, BaseEstimator):
    def __init__(self, reducer_model: type[torch.nn.Module], loss_fn: callable):
        super().__init__()
        self.loss_fn = loss_fn
        self.reducer_model = reducer_model

    def train(
        self,
        x: np.ndarray,
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
    ):
        dataset = TripletDataset(data=x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # feature_reducer = CNNFeatureReducer(N=x.shape[1], d=d)
        # triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2)

        optim = torch.optim.Adam(lr=lr, params=self.reducer_model.parameters())

        loss_per_epoch = []
        for epoch in tqdm(range(num_epochs)):
            running_loss = []
            for anchor, pos, neg in dataloader:
                optim.zero_grad()
                fa = self.reducer_model(anchor)
                fp = self.reducer_model(pos)
                fn = self.reducer_model(neg)
                loss = self.loss_fn(fa, fp, fn)
                loss.backward()
                optim.step()
                running_loss.append(loss.item())
            loss_per_epoch.append(np.mean(running_loss))
            dataloader.dataset.create_index()

        self.reducer_model.eval()
        self.metadata = {
            "loss_per_epoch": loss_per_epoch,
            "trained_at": str(datetime.now()),
        }

    def fit(
        self,
        x=None,
        y=None,
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
    ):
        self.train(x, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
        return self

    def predict(self, x=None):
        return None

    def transform(self, x: np.ndarray):
        with torch.no_grad():
            y = self.reducer_model(torch.Tensor(x))
        return y.numpy()
