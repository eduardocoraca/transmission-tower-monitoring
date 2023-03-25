from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import welch


@dataclass
class SignalSpectrum:
    frequencies: list
    magnitudes: list

    def get_spectrum(self) -> Tuple[List, List]:
        return self.frequencies, self.magnitudes


@dataclass
class SignalTime:
    signal: list
    ts: float
    dct_threshold: float = None

    def to_dct(self):
        X = dct(self.signal, type=2, norm="ortho")
        idx = np.arange(len(X))
        return SignalDCT(
            coefficients=list(X),
            indexes=list(idx),
            original_length=len(self),
            ts=self.ts,
        )

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

    def get_signal(self) -> Tuple[List, List]:
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

    def get(self, full=True) -> Tuple[List, List]:
        if full:
            coefficients = np.zeros(self.original_length)
            coefficients[self.indexes] = self.coefficients
        else:
            coefficients = self.coefficients
        return self.indexes, coefficients

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

    x: Union[SignalTime, SignalDCT] = None
    y: Union[SignalTime, SignalDCT] = None
    z: Union[SignalTime, SignalDCT] = None
    tension: float = None
    type: str = None

    def __post_init__(self):
        if isinstance(self.x, SignalDCT):
            self.type = "dct"
        elif isinstance(self.x, SignalTime):
            self.type = "time"

    def __getitem__(self, id: str):
        return {"x": self.x, "y": self.y, "z": self.z}[id]

    def convert_to_dct(self):
        if self.type != "dct":
            self.x = self.x.to_dct()
            self.y = self.y.to_dct()
            self.z = self.z.to_dct()
            self.type = "dct"

    def convert_to_time(self):
        if self.type != "time":
            self.x = self.x.to_time()
            self.y = self.y.to_time()
            self.z = self.z.to_time()
            self.type = "time"


@dataclass
class Sample:
    """Sample measurements containing data from all cables"""

    sample_id: str
    type: str = None
    sample_c1: SingleSample = None
    sample_c2: SingleSample = None
    sample_c3: SingleSample = None
    sample_c4: SingleSample = None
    metadata: dict = None

    def __post_init__(self):
        year, month, day, hour, minute = [int(x) for x in self.sample_id.split("_")]
        self.sampled_at = datetime(year, month, day, hour, minute)
        self.type = self.sample_c1.type

    def __getitem__(self, idx: int):
        return self._get_sample_list()[idx]

    def _get_sample_list(self):
        return [self.sample_c1, self.sample_c2, self.sample_c3, self.sample_c4]

    def convert_to_dct(self):
        if self.type != "dct":
            for sample in self._get_sample_list():
                sample.convert_to_dct()
            self.type = "dct"

    def convert_to_time(self):
        if self.type != "time":
            for sample in self._get_sample_list():
                sample.convert_to_time()
            self.type = "time"

    def compress(self, thp=0.1):
        assert self.type == "dct"

        for cable_sample in self._get_sample_list():
            cable_sample.compute_dct_threshold(thp)
            cable_sample.compress()
