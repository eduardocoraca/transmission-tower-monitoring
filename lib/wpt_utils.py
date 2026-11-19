import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import os
import pandas as pd
from pathlib import Path
from classes.sample.raw_data import LMSFile
from classes.signal.signal_classes import WaveletPacket
from classes.sample.sample_classes import SingleWaveletPacketSample, SampleWaveletPacket
import json

# from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
from pydantic import BaseModel
import pywt
import matplotlib.pyplot as plt
from datetime import datetime


def build_index_per_level(
    min_level: int = 0, max_level: int = 8
) -> dict[int, list[int]]:
    """
    Builds a dictionary where each key is a level and the value is a list of indices for that level.

    Args:
        min_level (int): The minimum level to build indices for.
        max_level (int): The maximum level to build indices for.

    Returns:
        Dict[int, List[int]]: A dictionary with levels as keys and lists of indices as values.

    Example:
        >>> build_index_per_level(3)
        {0: [0], 1: [1, 2], 2: [3, 4, 5, 6]}
    """
    index_per_level = {}
    i0 = 0
    for level in range(min_level, max_level):
        if level == 0:
            index_per_level[level] = [0]
            continue
        i0 = i0 + 2 ** (level - 1)
        level_indexes = np.arange(i0, i0 + 2**level)
        index_per_level[level] = level_indexes.tolist()
    return index_per_level


def combine_wpts(data: list[WaveletPacket]) -> WaveletPacket:
    """
    Combine multiple WaveletPacket objects into a single WaveletPacket object.

    This function takes a list of WaveletPacket objects and combines their energy
    and entropy maps. The combined energy map is calculated as the square root of
    the sum of the squares of the individual energy maps. The combined entropy map
    is calculated as the maximum value of the individual entropy maps.

    Args:
        data (List[WaveletPacket]): A list of WaveletPacket objects to be combined.

    Returns:
        WaveletPacket: A new WaveletPacket object with combined energy and entropy maps.
    """
    energies = np.array([x.energy_map for x in data])
    entropies = np.array([x.entropy_map for x in data])
    combined_energy = np.sqrt((energies**2).sum(axis=0))
    combined_entropy = entropies.max(axis=0)
    return WaveletPacket(
        max_level=data[0].max_level,
        fs=data[0].fs,
        family=data[0].family,
        energy_map=combined_energy,
        entropy_map=combined_entropy,
    )


def get_wpt_arrays(
    signal: np.ndarray,
    wavelet: str = "db1",
    maxlevel: int = 3,
    mode: str = "symmetric",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a signal using the Wavelet Packet Transform (WPT) and compute the energy and entropy of the coefficients at each level.

    Args:
        signal (np.ndarray): Input signal to be decomposed.
        wavelet (str): Type of wavelet to use for the decomposition. Default is "db1".
        maxlevel (int): Maximum level of decomposition. Default is 3.
        mode (str): Signal extension mode to use. Default is "symmetric".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 1D NumPy arrays containing the energy and entropy of the wavelet packet coefficients at each level.
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
    """
    Extract wavelet packet transform (WPT) data from an LMS file.

    Args:
        lms_file (LMSFile): The LMS file containing the data.
        maxlevel (int, optional): The maximum level of decomposition for the wavelet packet transform. Default is 6.
        family (str, optional): The wavelet family to use for the transform. Default is "db14".

    Returns:
        SampleWaveletPacket: An object containing the WPT data for each cable and axis.
    """
    dt = lms_file.dt
    fs = 1 / dt
    wpt_per_cable = {}
    for cable in [1, 2, 3, 4]:
        wpt_per_axis: dict = {}
        wpt_per_axis["tension"] = float(np.mean(getattr(lms_file, f"s{cable}")))
        for axis in ["x", "y", "z"]:
            x = getattr(lms_file, f"c{cable}_{axis}")
            energy_map, entropy_map = get_wpt_arrays(
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


def load_wavelets(
    path: Path,
    valid_start: datetime | None = None,
    valid_end: datetime | None = None,
    max_samples: int | None = None,
    samples_to_avoid: list[datetime] | None = None,
) -> list[SampleWaveletPacket]:
    files = os.listdir(path)
    wpts: list[SampleWaveletPacket] = []
    counter = 0
    for file in tqdm(files):
        with open(path / file, "r") as f:
            data = json.load(f)
        sample_wpt = SampleWaveletPacket(**data)
        date = datetime.fromisoformat(sample_wpt.sample_id)
        if samples_to_avoid and date in samples_to_avoid:
            continue
        if valid_start and date < valid_start:
            continue
        if valid_end and date > valid_end:
            continue
        wpts.append(sample_wpt)
        counter += 1
        if max_samples and counter >= max_samples:
            break
    return wpts


def normalize_by_level(wpt: WaveletPacket) -> WaveletPacket:
    """
    Normalizes the energy and entropy maps of a WaveletPacket by level.

    This function takes a WaveletPacket object and normalizes its energy and entropy maps
    at each level. The normalization is done by dividing the values at each level by the
    maximum value at that level.

    Args:
        wpt (WaveletPacket): The input WaveletPacket object containing the energy and entropy maps.

    Returns:
        WaveletPacket: A new WaveletPacket object with normalized energy and entropy maps.

    Notes:
        - The function assumes that the WaveletPacket object has attributes `max_level`, `energy_map`,
          and `entropy_map`.
        - The `build_index_per_level` function is used to get the indices corresponding to each level.
    """
    max_level = wpt.max_level
    energy_map = np.array(wpt.energy_map)
    entropy_map = np.array(wpt.entropy_map)
    idx_per_level = build_index_per_level(max_level=max_level + 1)
    normalized_energy = []
    normalized_entropy = []
    for level in range(max_level + 1):
        idx = idx_per_level[level]
        energies = energy_map[idx]
        entropies = entropy_map[idx]
        normalized_energy.append(energies / energies.max())
        normalized_entropy.append(entropies / entropies.max())

    return WaveletPacket(
        max_level=wpt.max_level,
        family=wpt.family,
        fs=wpt.fs,
        energy_map=np.hstack(normalized_energy),
        entropy_map=np.hstack(normalized_entropy),
    )


def flatten(l: dict[int, list[int]]) -> list[int]:
    """
    Flattens a dictionary of lists into a single list.

    Args:
        l (Dict[int, List[int]]): A dictionary where each value is a list of integers.

    Returns:
        List[int]: A single list containing all integers from the lists in the dictionary.
    """
    return [item for sublist in l.values() for item in sublist]


def plot_wpt(
    wpt_list: list[WaveletPacket],
    y_values: list[float],
    mode: str = "energy",  # "energy" or "entropy"
    level: int | None = None,
    db_scale: bool = False,
) -> None:
    """
    Plots a 2D colormap of wavelet packet energy or entropy for a list of WaveletPacket objects.

    Parameters:
    -----------
    wpt_list : List[WaveletPacket]
        A list of WaveletPacket objects to be plotted.
    y_values : List[float]
        A list of y-axis values corresponding to each WaveletPacket object.
    mode : str, optional
        The mode to plot, either "energy" or "entropy". Default is "energy".
    level : int, optional
        The specific decomposition level to plot. If None, plots all levels. Default is None.
    db_scale : bool, optional
        If True, converts the values to decibel (dB) scale. Default is False.
    """
    if not wpt_list:
        print("No WaveletPacket objects to plot.")
        return

    if len(y_values) != len(wpt_list):
        raise ValueError("len(y_values) must match len(wpt_list).")

    # 1) Determine the subband dimension from the first packet
    wpt0 = wpt_list[0]

    # 2) Create a 2D array: rows = wavelet packets, columns = subband indices
    index_per_level = build_index_per_level(wpt0.max_level + 1)
    if not level:
        ind = np.hstack([list(i) for i in index_per_level.values()])
    else:
        ind = np.array(index_per_level[level])

    num_subbands = len(ind)
    data_2d = np.zeros((len(wpt_list), num_subbands), dtype=float)
    for i, wpt_obj in enumerate(wpt_list):
        if mode.lower() == "energy":
            arr = np.ravel(wpt_obj.energy_map)[ind]
        else:
            arr = np.ravel(wpt_obj.entropy_map)[ind]

        if len(arr) != num_subbands:
            raise ValueError(
                "All WaveletPacket objects must have the same subband dimension."
            )

        # Convert to dB if requested
        if db_scale:
            arr = np.maximum(arr, 1e-20)
            arr = 20.0 * np.log10(arr)

        data_2d[i, :] = arr

    # 3) Subband index array
    subband_indices = np.arange(num_subbands)

    # 4) Plot the pcolormesh
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.pcolormesh(
        subband_indices,  # X-axis: subband index
        y_values,  # Y-axis: user-provided values
        data_2d,
        shading="auto",
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(("dB " if db_scale else "") + mode)

    # ax.set_title(f"Wavelet Packet {mode.capitalize()} Map (Y from input list)")
    ax.set_xlabel("Sub-band index $i$")
    ax.set_ylabel("Y Values")

    plt.tight_layout()
    plt.show()
