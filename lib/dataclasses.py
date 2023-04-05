from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, fft, idct, ifft
from scipy.signal import spectrogram, welch


@dataclass
class SignalSpectrum:
    frequencies: list
    magnitudes: list

    def get_spectrum(self) -> Tuple[List, List]:
        return self.frequencies, self.magnitudes

    def plot(self, ax, **kwargs):
        ax.plot(self.frequencies, self.magnitudes, **kwargs)


@dataclass
class SignalSpectrogram:
    frequencies: list
    time: list
    magnitudes: np.ndarray

    def get_spectrogram(self) -> Tuple[List, List]:
        return self.frequencies, self.time, self.magnitudes

    def plot(self, ax, transpose=False, **kwargs):
        if transpose:
            y = self.frequencies
            x = self.time
            z = self.magnitudes
        else:
            y = self.time
            x = self.frequencies
            z = self.magnitudes.T
        ax.pcolormesh(x, y, z, **kwargs)


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

    def get_rms(self):
        x = np.array(self.signal)
        return float(np.sqrt((x**2).mean()))

    def to_dft(self):
        X = fft(self.signal)
        idx = np.arange(len(X))
        return SignalDFT(
            coefficients=X[0 : len(X) // 2 + 1],
            indexes=idx[0 : len(X) // 2 + 1],
            original_length=len(self.signal),
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

    def to_spectrogram(
        self, window: str, nperseg: int, poverlap: float
    ) -> SignalSpectrogram:
        noverlap = int(poverlap * nperseg)
        zero_mean_signal = self.signal - np.mean(self.signal)
        f, t, X = spectrogram(
            zero_mean_signal,
            fs=1 / self.ts,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
        )
        return SignalSpectrogram(list(f), list(t), X)

    def plot(self, ax=None, **kwargs):
        time = self.ts * np.arange(len(self.signal))
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax.plot(time, self.signal, **kwargs)

    def __len__(self):
        return len(self.signal)

    def get(self) -> Tuple[List, List]:
        t = np.arange(0, len(self) * self.ts, self.ts)
        return list(t), list(self.signal)


@dataclass
class SignalDFT:
    """This class assumes the signal is Real and the coefficients are the first half."""

    coefficients: List[complex]
    indexes: list
    original_length: int
    ts: float
    dft_threshold: float = None

    def __post_init__(self):
        self.is_compressed = (
            True if self.original_length // 2 + 1 > len(self) else False
        )

    def get(self, full=True) -> Tuple[List, List]:
        if full:
            coefficients_left = np.zeros(
                self.original_length // 2 + 1, dtype=np.complex128
            )
            coefficients_left[self.indexes] = self.coefficients
            Xl = coefficients_left[0:-1]
            Xm = coefficients_left[-1]
            Xr = np.conjugate(np.flipud(Xl))[0:-1]
            X = np.hstack((Xl, Xm, Xr))
        else:
            X = self.coefficients
        return self.indexes, self.coefficients

    def to_time(self) -> SignalTime:
        _, full_coefficients = self.get(full=True)
        x = ifft(full_coefficients)
        x = np.real(x)
        return SignalTime(signal=list(x), ts=self.ts)

    def compute_dft_threshold(self, thp: float):
        """Computes the absolute DFT threhsold value for this signal.
        Args:
            thp (float): percentage of the maximum absolute magnitude to keep
        """
        if not self.is_compressed:
            abs_coefficients = np.abs(self.coefficients)
            self.dft_threshold = thp * abs_coefficients.max()
        else:
            print("The DFT is already compressed.")

    def compress(self):
        if self.is_compressed:
            print("The DFT is already compressed.")
        elif self.dft_threshold is None:
            print("No DFT threshold.")
        elif isinstance(self.dft_threshold, float):
            index_to_keep = np.abs(self.coefficients) > self.dft_threshold
            idx = np.arange(len(self.coefficients))[index_to_keep]
            self.coefficients = list(np.array(self.coefficients)[idx])
            self.indexes = idx
            self.is_compressed = True

    def plot(self, ax=None, **kwargs):
        _, coefficients = self.get(full=True)
        frequencies = np.linspace(0, 1 / self.ts / 2, len(coefficients))
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax.plot(frequencies, np.abs(coefficients), **kwargs)

    def __len__(self) -> int:
        return len(self.indexes)


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
        _, full_coefficients = self.get(full=True)
        x = idct(full_coefficients, norm="ortho", type=2)
        return SignalTime(signal=list(x), ts=self.ts)

    def get_energy(self, band: list = None, norm=False) -> float:
        """Computes the energy for a frequency band."""
        en = (np.array(self.coefficients) ** 2).sum()
        if band is None:
            return en
        else:
            f = np.linspace(0, (1 / self.ts) / 2, len(self))
            idx = (f >= band[0]) & (f <= band[1])
            en_band = (np.array(self.coefficients)[idx] ** 2).sum()
            if not norm:
                return en_band
            else:
                return en_band / en

    def compute_dct_threshold(self, thp: float):
        """Computes the absolute DCT threhsold value for this signal.
        Args:
            thp (float): percentage of the maximum absolute magnitude to keep
        """
        if not self.is_compressed:
            abs_coefficients = np.abs(self.coefficients)
            self.dct_threshold = thp * abs_coefficients.max()
        else:
            print("The DCT is already compressed.")

    def compress(self):
        if self.is_compressed:
            print("The DCT is already compressed.")
        elif self.dct_threshold is None:
            print("No DCT threshold.")
        elif isinstance(self.dct_threshold, float):
            index_to_keep = np.abs(self.coefficients) > self.dct_threshold
            idx = np.arange(len(self.coefficients))[index_to_keep]
            self.coefficients = list(np.array(self.coefficients)[idx])
            self.indexes = idx
            self.is_compressed = True

    def plot(self, ax=None, **kwargs):
        _, coefficients = self.get(full=True)
        frequencies = np.linspace(0, 1 / self.ts / 2, len(coefficients))
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax.plot(frequencies, coefficients, **kwargs)

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

    def _get_items(self):
        return [self.x, self.y, self.z]

    def compute_dct_threshold(self, thp: float):
        for x in self._get_items():
            x.compute_dct_threshold(thp)

    def compress(self):
        assert self.type == "dct"

        for x in self._get_items():
            x.compress()

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

    def _get_items(self):
        return [self.sample_c1, self.sample_c2, self.sample_c3, self.sample_c4]

    def __getitem__(self, idx: int):
        return self._get_items()[idx]

    def convert_to_dct(self):
        if self.type != "dct":
            for sample in self._get_items():
                sample.convert_to_dct()
            self.type = "dct"

    def convert_to_time(self):
        if self.type != "time":
            for sample in self._get_items():
                sample.convert_to_time()
            self.type = "time"

    def compress(self, thp=0.1):
        assert self.type == "dct"

        for cable_sample in self._get_items():
            cable_sample.compute_dct_threshold(thp)
            cable_sample.compress()
