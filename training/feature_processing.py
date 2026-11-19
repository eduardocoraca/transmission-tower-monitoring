from training.objects import SpectralFeatures, BandRange, NewSpectralFeatures
from classes.sample.sample_classes import SampleWaveletPacket
from lib.wpt_utils import build_index_per_level, combine_wpts
from datetime import datetime
import numpy as np
from training.constants import CABLES


def new_extract_features(
    sample_wpt: SampleWaveletPacket,
    f_min: float,
    f_max: float,
    fs: float,
    min_level: int,
    max_level: int,
    normalize: bool = False,
) -> list[NewSpectralFeatures]:
    """Extract spectral features from wavelet packet decomposition.

    Args:
        sample_wpt (SampleWaveletPacket): Wavelet packet decomposition for a sample
        f_min (float): Minimum frequency to consider (Hz)
        f_max (float): Maximum frequency to consider (Hz)
        fs (float): Sampling frequency (Hz)
        min_level (int): Minimum decomposition level
        max_level (int): Maximum decomposition level

    Returns:
        list[SpectralFeatures]: List of spectral features containing:
            - energy: Raw energy in the frequency band
            - norm_energy: Energy normalized by maximum energy in the level
            - range: Frequency range of the band
            - level: Decomposition level
            - cable: Cable identifier
    """
    sampled_at = datetime.fromisoformat(sample_wpt.sample_id)
    allowed_range = BandRange(start=f_min, end=f_max)
    index_per_level = build_index_per_level(
        min_level=min_level, max_level=max_level + 1
    )
    features = []
    for cable in CABLES:
        wpt = getattr(sample_wpt, f"sample_{cable}")
        tension = wpt.tension
        combined_wpt = combine_wpts([wpt.y, wpt.z])
        raw_energy = np.array(combined_wpt.energy_map)

        cable_features = []
        for _, indexes in index_per_level.items():
            num_indexes = len(indexes)
            bw = (fs / 2) / num_indexes
            freq_ranges = [
                BandRange(start=k * bw, end=(k + 1) * bw) for k in range(num_indexes)
            ]
            level_energy = raw_energy[indexes]

            # select relevant subbands
            max_level_energy = 0
            for band_range, raw in zip(freq_ranges, level_energy):
                if band_range.has_intersection(allowed_range):
                    if raw > max_level_energy:
                        max_level_energy = raw
                    energy = (
                        20 * np.log10(raw) if not normalize else 20 * np.log10(raw) / bw
                    )
                    # energy = raw if not normalize else raw / bw
                    cable_features.append(energy)
        features.append(
            NewSpectralFeatures(
                sampled_at=sampled_at,
                tension=tension,
                cable=cable,
                values=cable_features,
            )
        )

    return features


def build_left_right_transition_matrix(num_components: int) -> np.ndarray:
    transition_matrix = np.zeros((num_components, num_components))
    for i in range(num_components):
        if i == num_components - 1:
            transition_matrix[i, i] = 1
        elif i == 0:
            transition_matrix[i, i] = 0.9
            transition_matrix[i, i + 1] = 0.1
        else:
            transition_matrix[i, i] = 0.9
            transition_matrix[i, i + 1] = 0.1
            # transition_matrix[i, i-1] = 0.1
    return transition_matrix
