import os
from datetime import datetime

import numpy as np
import pandas as pd
import ujson as json
import yaml
from tqdm import tqdm
from yaml import CLoader as Loader

from lib.signal_classes import (Sample, SignalDCT, SignalSpectrum, SignalTime,
                                SingleSample)

PATH_TO_JSON_MAPPINGS = "/workspaces/project/lib/json_mappings.yml"

with open(PATH_TO_JSON_MAPPINGS, "r") as f:
    JSON_MAPPINGS = yaml.load(f, Loader)


def get_sample_from_json(path: str, sample_id: str, dct: bool = True) -> Sample:
    """Gets a sample from the JSON file and returns a Sample object according
    to the nomenclature in JSON_MAPPINGS.
    """

    with open(path, "r") as f:
        json_dict = json.load(f)

    sample_dict = {"metadata": {}}
    for key in json_dict.keys():
        if key in JSON_MAPPINGS.keys():
            field = JSON_MAPPINGS[key]
            sample_dict[field] = json_dict[key]
        else:
            sample_dict["metadata"][key] = json_dict[key]

    sample_per_cable = {}
    for cable in [1, 2, 3, 4]:
        cable_sample_per_channel = {}
        for channel in ["x", "y", "z"]:
            if dct:
                cable_sample_per_channel[channel] = SignalDCT(
                    coefficients=sample_dict[f"cable{cable}_{channel}"],
                    indexes=sample_dict[f"index_cable{cable}_{channel}"],
                    original_length=1024 * 16,
                    ts=sample_dict["metadata"]["dt"],
                )
            else:
                cable_sample_per_channel[channel] = SignalTime(
                    signal=sample_dict[f"cable{cable}_{channel}"],
                    ts=sample_dict["metadata"]["dt"],
                )
        sample_per_cable[f"c{cable}"] = SingleSample(
            x=cable_sample_per_channel["x"],
            y=cable_sample_per_channel["y"],
            z=cable_sample_per_channel["z"],
            tension=sample_dict[f"cable{cable}_tension"],
        )

    return Sample(
        sample_id=sample_id,
        sample_c1=sample_per_cable["c1"],
        sample_c2=sample_per_cable["c2"],
        sample_c3=sample_per_cable["c3"],
        sample_c4=sample_per_cable["c4"],
        metadata=sample_dict["metadata"],
    )


def get_samples_from_folder(path: str, min_valid_date: datetime) -> pd.DataFrame:
    sample_list = []
    date_list = []
    for file_path in tqdm(os.listdir(path)):
        sample_id = file_path.split(".json")[0]
        date = datetime.strptime(sample_id, "%Y_%m_%d_%H_%M")
        if date >= min_valid_date:
            sample = get_sample_from_json(path=path + file_path, sample_id=sample_id)
            sample_list.append(sample)
            date_list.append(date)

    dataset_per_cable = {}
    for k in [1, 2, 3, 4]:
        dataset_per_cable[f"c{k}"] = pd.DataFrame(
            {
                "x": [sample[k - 1].x for sample in sample_list],
                "y": [sample[k - 1].y for sample in sample_list],
                "z": [sample[k - 1].z for sample in sample_list],
                "tension": [sample[k - 1].tension for sample in sample_list],
            },
            index=date_list,
        )

    return dataset_per_cable


def get_features_from_folder(
    path: str, band_pairs: list, min_valid_date: datetime
) -> dict:
    bandwidths = np.array([pair[1] - pair[0] for pair in band_pairs])
    bandwidths = np.expand_dims(bandwidths, 0)

    valid_date = min_valid_date

    band_matrices_per_cable = {f"c{cable}": [] for cable in range(4)}
    tensions_per_cable = {f"c{cable}": [] for cable in range(4)}
    dates = []
    for file_path in tqdm(os.listdir(path)):
        sample_id = file_path.split(".json")[0]
        date = datetime.strptime(sample_id, "%Y_%m_%d_%H_%M")
        if date >= valid_date:
            dates.append(date)
            sample = get_sample_from_json(path=path + file_path, sample_id=sample_id)

            for cable in range(4):
                band_matrix = np.zeros(len(band_pairs))
                for j, pair in enumerate(band_pairs):
                    band_matrix[j] = sample[cable]["y"].get_energy(
                        [pair[0], pair[1]]
                    ) + sample[cable]["z"].get_energy([pair[0], pair[1]])
                normalized_band_matrix = (np.array(band_matrix) / bandwidths).squeeze()
                band_matrices_per_cable[f"c{cable}"].append(normalized_band_matrix)
                tensions_per_cable[f"c{cable}"].append(sample[cable].tension)

    for cable in range(4):
        band_matrices_per_cable[f"c{cable}"] = np.array(
            band_matrices_per_cable[f"c{cable}"]
        )

    columns = [f"band_{n}" for n in range(len(band_pairs))]
    dataset_per_cable = {}
    for cable in range(4):
        df = pd.DataFrame(band_matrices_per_cable[f"c{cable}"])
        df.columns = columns
        df["sampled_at"] = dates
        df["tension"] = tensions_per_cable[f"c{cable}"]
        df.set_index("sampled_at", inplace=True)
        dataset_per_cable[f"c{cable}"] = df
    return dataset_per_cable


def get_frequency_center_from_folder(
    path: str, band_pairs: list, min_valid_date: datetime
) -> dict:
    bandwidths = np.array([pair[1] - pair[0] for pair in band_pairs])
    bandwidths = np.expand_dims(bandwidths, 0)

    valid_date = min_valid_date

    band_matrices_per_cable = {f"c{cable}": [] for cable in range(4)}
    tensions_per_cable = {f"c{cable}": [] for cable in range(4)}
    dates = []
    for file_path in tqdm(os.listdir(path)):
        sample_id = file_path.split(".json")[0]
        date = datetime.strptime(sample_id, "%Y_%m_%d_%H_%M")
        if date >= valid_date:
            dates.append(date)
            sample = get_sample_from_json(path=path + file_path, sample_id=sample_id)

            for cable in range(4):
                band_matrix = np.zeros(len(band_pairs))
                for j, pair in enumerate(band_pairs):
                    combined = SignalDCT.combine(sample[cable]["y"], sample[cable]["z"])
                    band_matrix[j] = combined.get_frequency_center([pair[0], pair[1]])
                band_matrices_per_cable[f"c{cable}"].append(band_matrix)
                tensions_per_cable[f"c{cable}"].append(sample[cable].tension)

    for cable in range(4):
        band_matrices_per_cable[f"c{cable}"] = np.array(
            band_matrices_per_cable[f"c{cable}"]
        )

    columns = [f"band_{n}" for n in range(len(band_pairs))]
    dataset_per_cable = {}
    for cable in range(4):
        df = pd.DataFrame(band_matrices_per_cable[f"c{cable}"])
        df.columns = columns
        df["sampled_at"] = dates
        df["tension"] = tensions_per_cable[f"c{cable}"]
        df.set_index("sampled_at", inplace=True)
        dataset_per_cable[f"c{cable}"] = df
    return dataset_per_cable
