import sys

sys.path.append("..")

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from classes.common import Axes, Cables, Id
from classes.feature import FeatureExtractor, PassThroughFeatureExtractor
from classes.sample import Sample, SingleSample
from classes.signal import SignalDCT

JSON_MAPPING = {
    "S1_T2[N]": "cable1_tension",
    "S2_T2[N]": "cable2_tension",
    "S3_T2[N]": "cable3_tension",
    "S4_T4[N]": "cable4_tension",
    "c1t2_x:+X[g]": "cable1_x",
    "c1t2_y:+Y[g]": "cable1_y",
    "c1t2_z:+Z[g]": "cable1_z",
    "c2t2_x:+X[g]": "cable2_x",
    "c2t2_y:+Y[g]": "cable2_y",
    "c2t2_z:+Z[g]": "cable2_z",
    "c3t2_x:+X[g]": "cable3_x",
    "c3t2_y:+Y[g]": "cable3_y",
    "c3t2_z:+Z[g]": "cable3_z",
    "c4t2_x:+X[g]": "cable4_x",
    "c4t2_y:+Y[g]": "cable4_y",
    "c4t2_z:+Z[g]": "cable4_z",
    "index_c1t2_x:+X[g]": "index_cable1_x",
    "index_c1t2_y:+Y[g]": "index_cable1_y",
    "index_c1t2_z:+Z[g]": "index_cable1_z",
    "index_c2t2_x:+X[g]": "index_cable2_x",
    "index_c2t2_y:+Y[g]": "index_cable2_y",
    "index_c2t2_z:+Z[g]": "index_cable2_z",
    "index_c3t2_x:+X[g]": "index_cable3_x",
    "index_c3t2_y:+Y[g]": "index_cable3_y",
    "index_c3t2_z:+Z[g]": "index_cable3_z",
    "index_c4t2_x:+X[g]": "index_cable4_x",
    "index_c4t2_y:+Y[g]": "index_cable4_y",
    "index_c4t2_z:+Z[g]": "index_cable4_z",
}


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

        sample_dict = {"metadata": {}}
        for key in json_dict.keys():
            if key in self.mapping.keys():
                field = self.mapping[key]
                sample_dict[field] = json_dict[key]
            else:
                sample_dict["metadata"][key] = json_dict[key]

        sample_per_cable = {}
        for cable in [1, 2, 3, 4]:
            cable_sample_per_channel = {}
            for channel in ["x", "y", "z"]:
                cable_sample_per_channel[channel] = SignalDCT(
                    coefficients=sample_dict[f"cable{cable}_{channel}"],
                    indexes=sample_dict[f"index_cable{cable}_{channel}"],
                    original_length=1024 * 16,
                    ts=sample_dict["metadata"]["dt"],
                    id=Id(Cables(cable), Axes(channel)),
                )
            sample_per_cable[f"c{cable}"] = SingleSample(
                x=cable_sample_per_channel["x"],
                y=cable_sample_per_channel["y"],
                z=cable_sample_per_channel["z"],
                tension=sample_dict[f"cable{cable}_tension"],
                id=Id(Cables(cable), None),
            )

        return Sample(
            sample_id=filename,
            sample_c1=sample_per_cable["c1"],
            sample_c2=sample_per_cable["c2"],
            sample_c3=sample_per_cable["c3"],
            sample_c4=sample_per_cable["c4"],
            metadata=sample_dict["metadata"],
        )

    def load_from_folder(self):
        possible_cables = [1, 2, 3, 4]
        dates = []
        features_per_cable = {"c1": [], "c2": [], "c3": [], "c4": []}
        tensions_per_cable = {"c1": [], "c2": [], "c3": [], "c4": []}
        df_per_cable = {}
        filenames = os.listdir(self.folder_path)

        for filename in tqdm(filenames):
            dates.append(
                datetime.strptime(filename.split(".json")[0], "%Y_%m_%d_%H_%M")
            )
            sample = self.load_one(filename)
            for cable in possible_cables:
                single_sample = sample[cable - 1]
                f = self.feature_extractor.transform([single_sample])
                features_per_cable[f"c{cable}"].append(f)
                tensions_per_cable[f"c{cable}"].append(single_sample.tension)

        for cable in possible_cables:
            df_per_cable[f"c{cable}"] = pd.DataFrame(
                {
                    "tension": tensions_per_cable[f"c{cable}"],
                    "sampled_at": dates,
                    "features": features_per_cable[f"c{cable}"],
                }
            ).sort_values("sampled_at")

        return df_per_cable
