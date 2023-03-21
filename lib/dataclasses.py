from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import welch


@dataclass
class SignalSpectrum:
    frequencies: list
    magnitudes: list

    def get_spectrum(self) -> Tuple(List, List):
        return self.frequencies, self.magnitudes


@dataclass
class SignalTime:
    signal: list
    ts: float
    dct_threshold: float = None

    def to_dct(self):
        X = dct(self.signal, type=2, norm="ortho")
        idx = np.arange(len(X))
        return SignalDCT(coefficients=X, indexes=idx, length=len(self), ts=self.ts)

    def to_spectrum(self, window: str, nperseg: int, poverlap: float) -> SignalSpectrum:
        noverlap = int(poverlap * nperseg)
        zero_mean_signal = self.signal - np.mean(self.signal)
        f, X = welch(
            zero_mean_signal,
            fs=1 / self.ts,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        return SignalSpectrum(list(f), list(X))

    def __len__(self):
        return len(self.signal)

    def get_signal(self) -> Tuple(List, List):
        t = np.arange(0, len(self) * self.ts, self.ts)
        return list(t), list(self.signal)


@dataclass
class SignalDCT:
    coefficients: list
    indexes: list
    original_length: int
    ts: float
    dct_threshold: float = None

    def __post_init__(self):
        self.is_compressed = True if self.original_length > len(self) else False

    def to_time(self) -> SignalTime:
        full_coefficients = np.zeros(self.original_length)
        full_coefficients[self.indexes] = self.coefficients
        x = idct(full_coefficients, norm="ortho", type=2)
        return SignalTime(signal=list(x), ts=self.ts)

    def compute_dct_threshold(self, thp: float):
        """Computes the absolute DCT threhsold value for this signal.
        Args:
            thp (float): percentage of the maximum absolute magnitude to keep
        """
        if not self.is_compressed:
            abs_coefficients = np.abs(self.coefficients)
            self.dct_threshold = abs_coefficients >= thp * abs_coefficients.max()
        else:
            print("The DCT is already compressed.")

    def compress(self):
        if self.is_compressed:
            print("The DCT is already compressed.")
        elif isinstance(self.dct_threshold, float):
            index_to_keep = np.abs(self.coefficients) > self.dct_threshold
            idx = np.arange(len(self.coefficients))[index_to_keep]
            self.coefficients = self.coefficients[idx]
            self.indexes = idx
            self.is_compressed = True

    def __len__(self) -> int:
        return len(self.indexes)


@dataclass
class SingleSample:
    """Sample measurement containing data from a single cable"""

    signal_x_dct: SignalDCT = None
    signal_y_dct: SignalDCT = None
    signal_z_dct: SignalDCT = None
    tension: float = None


@dataclass
class Sample:
    """Sample measurements containing data from all cables"""

    sample_id: str
    sample_c1: SingleSample = None
    sample_c2: SingleSample = None
    sample_c3: SingleSample = None
    sample_c4: SingleSample = None
    metadata: dict = None

    def __post_init__(self):
        year, month, day, hour, minute = [int(x) for x in self.sample_id.split("_")]
        self.sampled_at = datetime(year, month, day, hour, minute)
