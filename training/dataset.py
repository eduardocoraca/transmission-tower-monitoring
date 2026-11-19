import numpy as np
from classes.sample.sample_classes import SampleWaveletPacket
from training.objects import WptCableDataset, NewSpectralFeatures, ScalerParams
from training.constants import CABLES
from lib.wpt_utils import build_index_per_level, combine_wpts
import torch
from torch.utils.data import TensorDataset


def create_pseudo_labels(size: int, num_bins: int) -> list[int]:
    time = np.arange(size)
    bins = np.histogram(time, bins=num_bins)[1]
    label = np.digitize(time, bins[:-1])
    return list(label)


def build_wpt_dataset_from_features(
    data: list[NewSpectralFeatures], num_bins: int
) -> list[WptCableDataset]:
    wpt_dataset = []
    for cable in CABLES:
        wpt_list = []
        tensions_list = []
        sampled_at_list = []
        for d in data:
            if d.cable == cable:
                wpt_list.append(d.values)
                tensions_list.append(d.tension)
                sampled_at_list.append(d.sampled_at)

        wpt_arr = np.array(wpt_list)
        tension_arr = np.array(tensions_list)

        # Create pseudo-labels
        label = create_pseudo_labels(len(wpt_arr), num_bins)

        # Store
        if len(wpt_arr) == 0:
            continue
        wpt_dataset.append(
            WptCableDataset(
                wpt=wpt_arr,
                tension=tension_arr,
                label=np.array(label),
                cable=cable,
                sampled_at=sampled_at_list,
            )
        )
    return wpt_dataset


def new_build_tensor_stacked_dataset(
    data_list: list[WptCableDataset],
) -> TensorDataset:
    wpt_stacked = np.vstack([v.wpt for v in data_list])
    label_stacked = np.hstack([v.label for v in data_list]) - 1
    tension_stacked = np.hstack([v.tension for v in data_list])

    return TensorDataset(
        torch.Tensor(wpt_stacked),
        torch.Tensor(label_stacked),
        torch.Tensor(tension_stacked),
    )




def normalize_tensor(x: torch.Tensor, scaler_params: ScalerParams) -> torch.Tensor:
    return (x - torch.Tensor(scaler_params.mu).reshape(1, -1)) / torch.Tensor(
        scaler_params.sigma
    ).reshape(1, -1)
