import sys

sys.path.append("..")
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to the system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import json
from typing import Tuple

from scipy.signal import csd, welch
from tqdm import tqdm

from classes.sample.raw_data import LMSFile
from classes.sample.sample_classes import SampleFRF, SingleFRFSample
from classes.signal.signal_classes import SignalFRF


def get_frfs_from_lms(
    lms_file: LMSFile, nperseg: int
) -> Tuple[SampleFRF, SampleFRF, SampleFRF]:
    """
    Compute the frequency response functions (FRFs) H1, H2 and Coherence from LMS data.

    Args:
        lms_file (LMSFile): An instance of LMSFile containing the input force and response data.
        nperseg (int): Length of each segment for the Welch method.

    Returns:
        Tuple[SampleFRF, SampleFRF]: Two SampleFRF instances containing the H1, H2 and Coherence FRFs for each cable and axis.
    """
    f_input = lms_file.input_force
    dt = lms_file.dt
    fs = 1 / dt
    h1_per_cable = {}
    h2_per_cable = {}
    coh_per_cable = {}
    for cable in [1, 2, 3, 4]:
        h1_per_axis = {}
        h2_per_axis = {}
        coh_per_axis = {}
        tension = float(np.mean(getattr(lms_file, f"s{cable}")))
        for axis in ["x", "y", "z"]:
            x = getattr(lms_file, f"c{cable}_{axis}")
            f, xx = welch(x=f_input, fs=fs, window="hann", nperseg=nperseg)
            _, yy = welch(x=x, fs=fs, window="hann", nperseg=nperseg)
            _, xy = csd(x=x, y=f_input, fs=fs, nperseg=nperseg, window="hann")
            _, yx = csd(x=f_input, y=x, fs=fs, nperseg=nperseg, window="hann")

            h1 = xy / xx
            h2 = yy / yx
            coh = (xy**2) / (xx * yy)

            h1_per_axis[axis] = SignalFRF(
                frequencies=f.tolist(),
                abs=np.abs(h1).tolist(),
                angle=np.angle(h1).tolist(),
            )
            h2_per_axis[axis] = SignalFRF(
                frequencies=f.tolist(),
                abs=np.abs(h2).tolist(),
                angle=np.angle(h2).tolist(),
            )
            coh_per_axis[axis] = SignalFRF(
                frequencies=f.tolist(),
                abs=np.abs(coh).tolist(),
                angle=np.zeros_like(f).tolist(),
            )
        h1_per_cable[cable] = SingleFRFSample(
            x=h1_per_axis["x"],
            y=h1_per_axis["y"],
            z=h1_per_axis["z"],
            tension=tension,
        )
        h2_per_cable[cable] = SingleFRFSample(
            x=h2_per_axis["x"],
            y=h2_per_axis["y"],
            z=h2_per_axis["z"],
            tension=tension,
        )
        coh_per_cable[cable] = SingleFRFSample(
            x=coh_per_axis["x"],
            y=coh_per_axis["y"],
            z=coh_per_axis["z"],
            tension=tension,
        )
    return (
        SampleFRF(
            sample_id=lms_file.sampled_at.isoformat(),
            sample_c1=h1_per_cable[1],
            sample_c2=h1_per_cable[2],
            sample_c3=h1_per_cable[3],
            sample_c4=h1_per_cable[4],
        ),
        SampleFRF(
            sample_id=lms_file.sampled_at.isoformat(),
            sample_c1=h2_per_cable[1],
            sample_c2=h2_per_cable[2],
            sample_c3=h2_per_cable[3],
            sample_c4=h2_per_cable[4],
        ),
        SampleFRF(
            sample_id=lms_file.sampled_at.isoformat(),
            sample_c1=coh_per_cable[1],
            sample_c2=coh_per_cable[2],
            sample_c3=coh_per_cable[3],
            sample_c4=coh_per_cable[4],
        ),
    )


def extract_frfs_and_save(path: Path, output_path: Path) -> None:
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
    output_path_h1 = output_path / "h1"
    output_path_h2 = output_path / "h2"
    output_path_coh = output_path / "coh"

    # Check if output directories exist, if not, create them
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_h1.mkdir(parents=True, exist_ok=True)
    output_path_h2.mkdir(parents=True, exist_ok=True)
    output_path_coh.mkdir(parents=True, exist_ok=True)

    processed_docs = set()
    for folder in tqdm(folders, desc="Processing folders"):
        files = os.listdir(path / folder)
        prefixes = np.unique([f[:-8] for f in files if f.endswith(".txt")])

        for prefix in prefixes:
            if prefix in processed_docs:
                continue
            processed_docs.add(prefix)
            lms_path = path / folder / f"{prefix}_LMS.txt"
            try:
                lms_file = LMSFile.from_txt(lms_path)
                sample_h1, sample_h2, sample_coh = get_frfs_from_lms(
                    lms_file=lms_file, nperseg=2048
                )

                filename = sample_h1.sample_id.replace(":", "-") + ".json"
                with open(output_path_h1 / filename, "w") as h:
                    json.dump(sample_h1.model_dump(), h)
                with open(output_path_h2 / filename, "w") as h:
                    json.dump(sample_h2.model_dump(), h)
                with open(output_path_coh / filename, "w") as h:
                    json.dump(sample_coh.model_dump(), h)
            except Exception as e:
                print(f"Failed to process {lms_path}: {e}")


if __name__ == "__main__":
    path = Path(os.environ.get("RAW_DATA_PATH", "data/raw"))
    output_path = Path(r"data\\frfs\\sequence_3\\")
    extract_frfs_and_save(path, output_path)
