import sys

sys.path.append("..")

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from classes.common import Axes, Cables, Id
from classes.feature import FeatureExtractor, PassThroughFeatureExtractor
from classes.sample import Sample, SingleSample
from classes.sample.sample_types import SampleTypes
from classes.signal import SignalDCT

from .config import JSON_MAPPING


class Loader:
    def __init__(
        self,
        folder_path: str,
        feature_extractor: FeatureExtractor = PassThroughFeatureExtractor(),
    ):
        self.mapping = JSON_MAPPING
        self.feature_extractor = feature_extractor
        self.folder_path = folder_path

    def load_one(self, filename: str) -> Sample:
        with open(f"{self.folder_path}\{filename}", "r") as handle:
            json_dict = json.load(handle)

        sample_dict: dict[str, Any] = {"metadata": {}}
        for key in json_dict.keys():
            if key in self.mapping.keys():
                field = self.mapping[key]
                sample_dict[field] = json_dict[key]
            else:
                sample_dict["metadata"][key] = json_dict[key]

        sample_per_cable: dict[Cables, SingleSample] = {}
        for cable in Cables:
            cable_sample_per_channel = {}
            for axis in Axes:
                cable_sample_per_channel[axis] = SignalDCT(
                    coefficients=sample_dict[f"cable{cable.value}_{axis.value}"],
                    indexes=sample_dict[f"index_cable{cable.value}_{axis.value}"],
                    original_length=1024 * 16,
                    ts=sample_dict["metadata"]["dt"],
                    id=Id(cable, axis),
                )
            sample_per_cable[cable] = SingleSample(
                x=cable_sample_per_channel[Axes.X],
                y=cable_sample_per_channel[Axes.Y],
                z=cable_sample_per_channel[Axes.Z],
                tension=sample_dict[f"cable{cable.value}_tension"],
                id=Id(cable=cable),
                type=SampleTypes.DCT,
            )

        return Sample(
            sample_id=filename,
            sample_c1=sample_per_cable[Cables.C1],
            sample_c2=sample_per_cable[Cables.C2],
            sample_c3=sample_per_cable[Cables.C3],
            sample_c4=sample_per_cable[Cables.C4],
            metadata=sample_dict.get("metadata"),
            type=SampleTypes.DCT,
        )

    def load_from_folder(self) -> dict[Cables, pd.DataFrame]:
        dates = []
        features_per_cable: dict[Cables, list[Any]] = {cable: [] for cable in Cables}
        tensions_per_cable: dict[Cables, list[Any]] = {cable: [] for cable in Cables}
        df_per_cable = {}
        filenames = os.listdir(self.folder_path)
        for filename in tqdm(filenames):
            dates.append(
                datetime.strptime(filename.split(".json")[0], "%Y_%m_%d_%H_%M")
            )
            sample = self.load_one(filename)
            for cable in Cables:
                single_sample: SingleSample = sample[cable.value - 1]
                features: np.ndarray = self.feature_extractor.transform([single_sample])
                features_per_cable[cable].append(features)
                tensions_per_cable[cable].append(single_sample.tension)

        for cable in Cables:
            df_per_cable[cable] = pd.DataFrame(
                {
                    "tension": tensions_per_cable[cable],
                    "sampled_at": dates,
                    "features": features_per_cable[cable],
                }
            ).sort_values("sampled_at")

        return df_per_cable
