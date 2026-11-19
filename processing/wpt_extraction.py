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
from classes.signal.signal_classes import WaveletPacket
from classes.sample.sample_classes import SingleWaveletPacketSample, SampleWaveletPacket
import json
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
import pywt


def wavelet_packet_vectors(
    signal: np.ndarray,
    wavelet: str = "db1",
    maxlevel: int = 3,
    mode: str = "symmetric",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose the 1D signal up to 'maxlevel' with wavelet packets and compute the
    energy (sum of squares of coefficients) for each node in level order (0...maxlevel).
    Returns a 1D array of energies in BFS order:
        - The first element corresponds to level=0 (just one node),
        - The next 2 elements correspond to level=1,
        - The next 4 elements to level=2, etc.

    Args:
        signal (np.ndarray): 1D array representing the input signal.
        wavelet (str): Name of the wavelet (default="db1").
        maxlevel (int): Maximum decomposition level (default=3).
        mode (str): Signal extension mode (default="symmetric").

    Returns:
        np.ndarray: 1D array of size (2^(maxlevel+1) - 1) containing energy values
                    in BFS (level) order.
    """
    # Create WaveletPacket object
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode=mode, maxlevel=maxlevel)

    # We will collect energies level by level
    energy_list = []
    entropy_list = []
    total_energy = np.sum(np.array(signal) ** 2)
    for level in range(maxlevel + 1):
        if level == 0:
            nodes = wp.get_level(level, order="natural")
        else:
            nodes = wp.get_level(level, order="freq")
        for node in nodes:
            coeffs = np.array(node.data)
            energy = np.sum(coeffs**2)
            p = coeffs**2 / total_energy
            entropy = np.where(p > 0.0, p * np.log(p), 0)
            entropy_list.append(-np.sum(entropy))
            energy_list.append(energy)

    # Convert to a 1D NumPy array
    energy_vector = np.array(energy_list, dtype=float)
    entropy_vector = np.array(entropy_list, dtype=float)
    return energy_vector, entropy_vector


def get_wpt_from_lms(
    lms_file: LMSFile, maxlevel: int = 6, family: str = "db14"
) -> SampleWaveletPacket:
    dt = lms_file.dt
    fs = 1 / dt
    wpt_per_cable = {}
    for cable in [1, 2, 3, 4]:
        wpt_per_axis: Dict[str, Any] = {}
        wpt_per_axis["tension"] = float(np.mean(getattr(lms_file, f"s{cable}")))
        for axis in ["x", "y", "z"]:
            x = getattr(lms_file, f"c{cable}_{axis}")
            energy_map, entropy_map = wavelet_packet_vectors(
                x, maxlevel=maxlevel, wavelet=family, mode="symmetric"
            )
            wpt_per_axis[axis] = WaveletPacket(
                max_level=maxlevel,
                family=family,
                fs=fs,
                energy_map=energy_map.tolist(),
                entropy_map=entropy_map.tolist(),
            )
        wpt_per_cable[cable] = SingleWaveletPacketSample(**wpt_per_axis)

    return SampleWaveletPacket(
        sample_id=lms_file.sampled_at.isoformat(),
        sample_c1=wpt_per_cable[1],
        sample_c2=wpt_per_cable[2],
        sample_c3=wpt_per_cable[3],
        sample_c4=wpt_per_cable[4],
    )


def extract_wpts_and_save(path: Path, output_path: Path) -> None:
    """
    Extracts WPTs from LMS files in the given directory and saves them as JSON files.

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
                sample_psd = get_wpt_from_lms(
                    lms_file=lms_file, maxlevel=8, family="db35"
                )
                filename = sample_psd.sample_id.replace(":", "-") + ".json"
                with open(output_path / filename, "w") as h:
                    json.dump(sample_psd.model_dump(), h)
            except Exception as e:
                print(f"Failed to process {lms_path}: {e}")


if __name__ == "__main__":
    path = Path(os.environ.get("RAW_DATA_PATH", "data/raw"))
    output_path = Path(r"data\\wpts\\sequence_3_db35\\")
    extract_wpts_and_save(path, output_path)
