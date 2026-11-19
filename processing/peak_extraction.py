import os
import sys

sys.path.append("..")
from pathlib import Path

# Add the parent directory to the system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from typing import List, Optional, Dict, Tuple
from datetime import datetime
import json
from tqdm import tqdm
from classes.sample.sample_classes import SampleFRF, SingleFRFSample
import numpy as np
from scipy.signal import find_peaks
from classes.signal.signal_classes import SignalFRF
from classes.peak.peak_classes import PeakRecord
from dataclasses import dataclass

CABLES = ["1", "2", "3", "4"]


@dataclass
class Peak:
    frequency: float
    magnitude: float


def load_sample_frf(path: Path) -> SampleFRF:
    with open(path, "r") as f:
        data = json.load(f)
    return SampleFRF(**data)


def combine_frf(frfs: list[SignalFRF]) -> SignalFRF:
    return SignalFRF(
        frequencies=frfs[0].frequencies,
        abs=np.sqrt(np.sum([np.array(frf.abs) ** 2 for frf in frfs], axis=0)),
        angle=np.mean([frf.angle for frf in frfs], axis=0),
    )


def get_peaks(x: SignalFRF) -> list[Peak]:
    peaks = []
    mag = np.array(x.abs)
    mag_norm = mag / np.max(mag)
    freq = np.array(x.frequencies)
    valid_idx = (8 < freq) & (freq < 120)
    mag_norm = mag_norm[valid_idx]
    freq = freq[valid_idx]
    peaks_idx, _ = find_peaks(mag_norm, distance=10, threshold=0.001, width=2)
    for idx in peaks_idx:
        peaks.append(Peak(frequency=freq[idx], magnitude=mag[idx]))
    return peaks


def create_peak_record(
    sample_id: str, tension: float, axis: str, cable: int, peaks: list[Peak]
) -> PeakRecord:
    """Create a PeakRecord from the given parameters and peaks."""
    return PeakRecord(
        sample_id=sample_id,
        tension=tension,
        axis=axis,
        cable=cable,
        frequency=[p.frequency for p in peaks],
        magnitude=[p.magnitude for p in peaks],
    )


def save_peak_record(
    record: PeakRecord, output_path: Path, axis_folder: str, filename: str
) -> None:
    """Save a PeakRecord to the specified path."""
    axis_path = output_path / axis_folder
    axis_path.mkdir(parents=True, exist_ok=True)

    with open(axis_path / filename, "w") as f:
        json.dump(record.model_dump(), f)


def get_axis_data(cable_frf: SingleFRFSample) -> Dict[str, SignalFRF]:
    """Get the signal data for each axis from a cable FRF sample."""
    return {"x": cable_frf.x, "r": combine_frf([cable_frf.y, cable_frf.z])}


def process_cable_peaks(sample_frf: SampleFRF, cable: str, output_path: Path) -> None:
    """Process peaks for a single cable and save the results."""
    cable_frf: SingleFRFSample = sample_frf.__getattribute__(f"sample_c{cable}")
    axis_data = get_axis_data(cable_frf)

    # Define axis to folder mapping
    axis_folders = {"x": "axial", "r": "radial"}
    filename = sample_frf.sample_id.replace(":", "-") + "_" + cable + ".json"

    for axis, signal_data in axis_data.items():
        peaks = get_peaks(signal_data)
        record = create_peak_record(
            sample_id=sample_frf.sample_id,
            tension=cable_frf.tension,
            axis=axis,
            cable=int(cable),
            peaks=peaks,
        )
        save_peak_record(record, output_path, axis_folders[axis], filename)


if __name__ == "__main__":
    path = Path(r"data\\frfs\\sequence_3\\h2")
    files = os.listdir(path)
    output_path = Path(r"data\\frf_peaks\\sequence_3\\")

    for f in tqdm(files):
        sample_frf = load_sample_frf(path / f)
        for cable in CABLES:
            process_cable_peaks(sample_frf, cable, output_path)
