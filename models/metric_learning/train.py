import numpy as np
import torch
from tqdm import tqdm

from .model_classes import DataLoader, TripletDataset
from .models import FeatureReducer


def train(
    x: np.ndarray,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 100,
):
    dataset = TripletDataset(data=x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    feature_reducer = FeatureReducer(N=64, d=3)
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

    feature_reducer = feature_reducer.eval()
    return feature_reducer, loss_per_epoch
