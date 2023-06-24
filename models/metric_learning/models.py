from datetime import datetime

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from .model_classes import DataLoader, TripletDataset


class FeatureReducer(torch.nn.Module):
    def __init__(self, N: int, d: int):
        super().__init__()

        self.N = N
        self.d = d

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear((self.N // 8) * 3, d),
        )

    def __call__(self, x: torch.Tensor):
        return self.model(x.unsqueeze(1))


class FeatureReducerScikit(TransformerMixin, BaseEstimator):
    def __init__(self, pt_model: FeatureReducer = None):
        super().__init__()
        self.pt_model = pt_model
        self.pretrained = False if self.pt_model is None else True

    def train(
        self,
        x: np.ndarray,
        d: int = 2,
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
    ):
        dataset = TripletDataset(data=x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        feature_reducer = FeatureReducer(N=x.shape[1], d=d)
        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2)

        optim = torch.optim.Adam(lr=lr, params=feature_reducer.parameters())

        loss_per_epoch = []
        for epoch in tqdm(range(num_epochs)):
            running_loss = []
            for anchor, pos, neg in dataloader:
                optim.zero_grad()
                fa = feature_reducer(anchor)
                fp = feature_reducer(pos)
                fn = feature_reducer(neg)
                loss = triplet_loss(fa, fp, fn)
                loss.backward()
                optim.step()
                running_loss.append(loss.item())
            loss_per_epoch.append(np.mean(running_loss))
            dataloader.dataset.create_index()

        self.pt_model = feature_reducer.eval()
        self.metadata = {
            "loss_per_epoch": loss_per_epoch,
            "trained_at": str(datetime.now()),
        }

    def fit(
        self,
        x=None,
        y=None,
        d: int = 2,
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
    ):
        if not self.pretrained:
            self.train(x, d=d, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
        return self

    def predict(self, x=None):
        return None

    def transform(self, x: np.ndarray):
        with torch.no_grad():
            y = self.pt_model(torch.Tensor(x))
        return y.numpy()
