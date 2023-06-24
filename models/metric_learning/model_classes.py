from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model_utils import create_triplets_idx, get_proba_per_sample


class TripletDataset(Dataset):
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        pos_t: float = 250,
        neg_t: float = 750,
    ):
        super().__init__()
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        self.data = data  # dim. (N,D)
        self.N = self.data.shape[0]
        self.pos_t = pos_t
        self.neg_t = neg_t
        self.create_index()

    def create_index(self):
        positive_proba, negative_proba = get_proba_per_sample(
            N=len(self), pos_t=self.pos_t, neg_t=self.neg_t
        )
        self.triplets_idx = create_triplets_idx(
            positive_proba=positive_proba, negative_proba=negative_proba
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_pos = self.triplets_idx[idx][1]
        idx_neg = self.triplets_idx[idx][2]

        data_anchor = self.data[idx]
        data_pos = self.data[idx_pos]
        data_neg = self.data[idx_neg]

        return data_anchor, data_pos, data_neg
