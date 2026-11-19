import sys

sys.path.append("..")
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Add the parent directory to the system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from classes.sample.raw_data import LMSFile
from classes.sample.sample_classes import (
    SampleStatisticalFeatures,
    SingleStatisticalFeaturesSample,
)
from classes.feature.feature_classes import StatisticalFeatures
import json
from typing import Tuple
from tqdm import tqdm


FOLDERS_TO_AVOID = [
    "Teste_fc_20221116",
    "Teste_fc_20221118",
    "Teste_fc_20221121",
    "Teste_fc_20221123",
    "Teste_fc_20221125",
    "Teste_fc_20221128",
    "Teste_fc_20221130",
    "Teste_fc_20221201",
    "Teste_fc_20221202",
    "Teste_fc_20221205",
    "Teste_fc_20221207",
    "Teste_fc_20221209",
    "Teste_fc_20221212",
    "Teste_fc_20221215",
    "Teste_fc_20221216",
    "Teste_fc_20221219",
    "Teste_fc_20221221",
    "Teste_fc_20221223",
]


def get_stat_features_from_lms(lms_file: LMSFile) -> SampleStatisticalFeatures:
    """
    Compute the frequency response functions (FRFs) H1 and H2 from LMS data.

    Args:
        lms_file (LMSFile): An instance of LMSFile containing the input force and response data.
        nperseg (int): Length of each segment for the Welch method.

    Returns:
        Tuple[SampleFRF, SampleFRF]: Two SampleFRF instances containing the H1 and H2 FRFs for each cable and axis.
    """
    features_per_cable = {}
    for cable in [1, 2, 3, 4]:
        features_per_axis = {}
        tension = float(np.mean(getattr(lms_file, f"s{cable}")))

        for axis in ["x", "y", "z"]:
            x = getattr(lms_file, f"c{cable}_{axis}")
            features_per_axis[axis] = StatisticalFeatures.from_signal(x)

        features_per_cable[cable] = SingleStatisticalFeaturesSample(
            x=features_per_axis["x"],
            y=features_per_axis["y"],
            z=features_per_axis["z"],
            tension=tension,
        )

    return SampleStatisticalFeatures(
        sample_id=lms_file.sampled_at.isoformat(),
        temperature=lms_file.temperature,
        sample_c1=features_per_cable[1],
        sample_c2=features_per_cable[2],
        sample_c3=features_per_cable[3],
        sample_c4=features_per_cable[4],
    )


def extract_features_and_save(path: Path, output_path: Path) -> None:
    """
    Extracts FRFs from LMS files in the given directory and saves them as JSON files.

    Args:
        path (Path): The directory containing the LMS files.
        output_path (Path): The directory where the JSON files will be saved.
    """
    folders = [
        folder
        for folder in os.listdir(path)
        if folder.startswith("Teste") and not folder.endswith(".npy")
    ]

    # Check if output directories exist, if not, create them
    output_path.mkdir(parents=True, exist_ok=True)

    processed_docs = set()
    for folder in tqdm(folders, desc="Processing folders"):
        if folder in FOLDERS_TO_AVOID:
            continue
        if folder not in ["Teste_T3_20230922", "Teste_T3_20231004"]:
            continue
        files = os.listdir(path / folder)
        prefixes = np.unique([f[:-8] for f in files if f.endswith(".txt")])

        for prefix in prefixes:
            if prefix in processed_docs:
                continue
            processed_docs.add(prefix)
            lms_path = path / folder / f"{prefix}_LMS.txt"
            try:
                lms_file = LMSFile.from_txt(lms_path)
                features = get_stat_features_from_lms(lms_file=lms_file)
                filename = features.sample_id.replace(":", "-") + ".json"
                with open(output_path / filename, "w") as h:
                    json.dump(features.model_dump(), h)

            except Exception as e:
                print(f"Failed to process {lms_path}: {e}")


if __name__ == "__main__":
    path = Path(os.environ.get("RAW_DATA_PATH", "data/raw"))
    output_path = Path(r"data\\statistical_features\\sequence_3\\")
    extract_features_and_save(path, output_path)
