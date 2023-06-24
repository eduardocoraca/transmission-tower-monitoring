from typing import Tuple

import numpy as np
import torch


def get_proba_per_sample(
    N: int, pos_t: int = 100, neg_t: int = 300
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the positive and negative probability arrays for a sequence.
    Args:
        N (int): size of the ordered sequence
        pos_t (float): softmax temperature hyperparam for the positive distributions
        neg_t (float): softmax temperature hyperparam for the negative distributions
    Returns:
        tuple(torch.Tensor, torch.Tensor): positive and negative probability arrays, dim. (N,N)
    """
    time = np.arange(N)
    time_without_anchors = np.repeat(time.reshape(1, -1), N, axis=0)
    time_distances = np.float32(np.abs(time.reshape(-1, 1) - time_without_anchors))
    positive_logits = N - time_distances
    negative_logits = time_distances
    np.fill_diagonal(positive_logits, 0)
    np.fill_diagonal(negative_logits, 0)

    positive_proba = torch.softmax(torch.Tensor(positive_logits / pos_t), axis=1)
    negative_proba = torch.softmax(torch.Tensor(negative_logits / neg_t), axis=1)

    positive_proba = positive_proba.fill_diagonal_(0)
    negative_proba = negative_proba.fill_diagonal_(0)

    positive_proba = positive_proba / positive_proba.sum(axis=1)
    negative_proba = negative_proba / negative_proba.sum(axis=1)

    return positive_proba, negative_proba


def create_triplets_idx(
    positive_proba: torch.Tensor, negative_proba: torch.Tensor
) -> torch.Tensor:
    """Creates triplets index, represented by a (N,3) tensor where N is the number of samples.
    Args:
        positive_proba (torch.Tensor): (N,N) array, where row [i,:] is the positive prob. array of each index for sample i
        negative_proba (torch.Tensor): (N,N) array, where row [i,:] is the negative prob. array of each index for sample i
    Returns:
        triplets (torch.Tensor): array of [anchor_idx, positive_idx, negative_idx]
    """

    N = positive_proba.shape[0]
    assert negative_proba.shape[0] == N
    anchor_idx = np.arange(N)
    positive_idx = np.hstack(
        [
            np.random.choice(anchor_idx, size=1, p=p_i).squeeze()
            for p_i in positive_proba.numpy()
        ]
    )
    negative_idx = np.hstack(
        [
            np.random.choice(anchor_idx, size=1, p=p_i).squeeze()
            for p_i in negative_proba.numpy()
        ]
    )
    out_array = np.stack((anchor_idx, positive_idx, negative_idx))
    return torch.Tensor(out_array.T).type(torch.int16)
