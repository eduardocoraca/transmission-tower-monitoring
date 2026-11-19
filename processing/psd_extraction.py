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
from classes.signal.signal_classes import SignalSpectrum
from classes.sample.sample_classes import SinglePSDSample, SamplePSD
import json
from scipy.signal import welch, csd
from typing import Tuple
from tqdm import tqdm


def get_psds_from_lms(lms_file: LMSFile, nperseg: int) -> SamplePSD:
    """
    Compute the power spectraum densities (PSDs) from LMS data.

    Args:
        lms_file (LMSFile): An instance of LMSFile containing the input force and response data.
        nperseg (int): Length of each segment for the Welch method.

    Returns:
        SamplePSD: An instance of SamplePSD containing the PSDs for each cable and axis.
    """
    dt = lms_file.dt
    fs = 1 / dt
    psd_per_cable = {}
    for cable in [1, 2, 3, 4]:
        tension = float(np.mean(getattr(lms_file, f"s{cable}")))
        psd_per_axis = {}
        for axis in ["x", "y", "z"]:
            x = getattr(lms_file, f"c{cable}_{axis}")
            f, yy = welch(x=x, fs=fs, window="hann", nperseg=nperseg)
            psd_per_axis[axis] = SignalSpectrum(
                frequencies=f.tolist(),
                magnitudes=np.abs(yy).tolist(),
            )

        psd_per_cable[cable] = SinglePSDSample(
            x=psd_per_axis["x"],
            y=psd_per_axis["y"],
            z=psd_per_axis["z"],
            tension=tension,
        )

    return SamplePSD(
        sample_id=lms_file.sampled_at.isoformat(),
        sample_c1=psd_per_cable[1],
        sample_c2=psd_per_cable[2],
        sample_c3=psd_per_cable[3],
        sample_c4=psd_per_cable[4],
    )


def extract_psds_and_save(path: Path, output_path: Path) -> None:
    """
    Extracts PSDs from LMS files in the given directory and saves them as JSON files.

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
        files = os.listdir(path / folder)
        prefixes = np.unique([f[:-8] for f in files if f.endswith(".txt")])

        for prefix in prefixes:
            if prefix in processed_docs:
                continue
            processed_docs.add(prefix)
            lms_path = path / folder / f"{prefix}_LMS.txt"
            try:
                lms_file = LMSFile.from_txt(lms_path)
                sample_psd = get_psds_from_lms(lms_file=lms_file, nperseg=2048)
                filename = sample_psd.sample_id.replace(":", "-") + ".json"
                with open(output_path / filename, "w") as h:
                    json.dump(sample_psd.model_dump(), h)
            except Exception as e:
                print(f"Failed to process {lms_path}: {e}")


if __name__ == "__main__":
    path = Path(os.environ.get("RAW_DATA_PATH", "data/raw"))
    output_path = Path(r"data\\psds\\sequence_3\\")
    extract_psds_and_save(path, output_path)
