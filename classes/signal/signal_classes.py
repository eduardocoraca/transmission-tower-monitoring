from dataclasses import dataclass
from typing import Any, Optional

import nptyping as npt
import numpy as np
from scipy.fftpack import dct, idct  # type: ignore
from scipy.signal import spectrogram, welch  # type: ignore

from classes.common import Id
from classes.sample.sample_types import SampleTypes

from .interface import Signal


@dataclass
class SignalSpectrum(Signal):
    frequencies: list[float]
    magnitudes: list[float]
    type: str = SampleTypes.PSD

    def get(self) -> tuple[list[float], list[float]]:
        return self.frequencies, self.magnitudes


@dataclass
class SignalSpectrogram(Signal):  # TODO remove
    frequencies: list[float]
    time: list[float]
    magnitudes: npt.NDArray

    def get(
        self,
    ) -> tuple[list[float], list[float], npt.NDArray,]:
        return self.frequencies, self.time, self.magnitudes


@dataclass
class SignalTime(Signal):
    signal: list[float]
    ts: float
    id: Id
    type: SampleTypes = SampleTypes.TIME

    def to_dct(self) -> "SignalDCT":
        X = dct(self.signal, type=2, norm="ortho")
        idx = np.arange(len(X))
        return SignalDCT(
            coefficients=list(X),
            indexes=list(idx),
            original_length=len(self),
            ts=self.ts,
            id=self.id,
        )

    def get_rms(self) -> float:
        x = np.array(self.signal)
        return float(np.sqrt((x**2).mean()))

    def to_spectrum(self, window: str, nperseg: int, poverlap: float) -> SignalSpectrum:
        noverlap = int(poverlap * nperseg)
        zero_mean_signal = np.array(self.signal) - np.mean(self.signal)
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

    def __len__(self):
        return len(self.signal)

    def get(self) -> tuple[list[float], list[float]]:
        """Returns a tuple of (time, signal) lists."""
        t = np.arange(0, len(self) * self.ts, self.ts)
        return list(t), list(self.signal)


@dataclass
class SignalDCT(Signal):
    coefficients: list[float]
    indexes: list[float]
    original_length: int
    ts: float
    id: Id
    dct_threshold: Optional[float] = None
    type: SampleTypes = SampleTypes.DCT

    def __post_init__(self):
        self.is_compressed = True if self.original_length > len(self) else False

    def get(self, full=True) -> tuple[list[float], list[float]]:
        if full:
            coefficients = np.zeros(self.original_length)
            coefficients[self.indexes] = self.coefficients
        else:
            coefficients = self.coefficients
        return self.indexes, coefficients

    def to_time(self) -> SignalTime:
        _, full_coefficients = self.get(full=True)
        x = idct(full_coefficients, norm="ortho", type=2)
        return SignalTime(signal=list(x), ts=self.ts, id=self.id)

    def get_energy(self, band: list = None, norm=False) -> float:
        """Computes the energy for a frequency band."""
        en = (np.array(self.coefficients) ** 2).sum()
        if band is None:
            return en
        else:
            freq = np.linspace(0, (1 / self.ts) / 2, self.original_length)
            freq = freq[self.indexes]
            idx = (freq >= band[0]) & (freq <= band[1])
            en_band = (np.array(self.coefficients)[idx] ** 2).sum()
            if not norm:
                return en_band
            else:
                return en_band / en

    def get_frequency_center(self, band: list = None) -> float:
        """Computes the frequency center for a frequency band."""
        freq = np.linspace(0, (1 / self.ts) / 2, self.original_length)[self.indexes]
        weights = np.array(self.coefficients) ** 2
        if band is None:
            fc = np.sum(weights * freq) / np.sum(weights)
            return fc
        else:
            idx = (freq >= band[0]) & (freq <= band[1])
            weights_band = weights[idx]
            if np.sum(weights_band) == 0:
                return -1
            freq_band = freq[idx]
            fc = np.sum(weights_band * freq_band) / np.sum(weights_band)
            fc = (fc - band[0]) / (band[1] - band[0])
            if fc < 0:
                fc = 0
            return fc

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

    def compress(self) -> None:
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

    @classmethod
    def combine(cls, dct_1: "SignalDCT", dct_2: "SignalDCT") -> "SignalDCT":
        idx_1, coef_1 = dct_1.get(full=True)
        idx_2, coef_2 = dct_2.get(full=True)
        combined_coef = np.sqrt(coef_1**2 + coef_2**2)
        combined_idx = np.arange(len(combined_coef))
        return cls(
            coefficients=combined_coef,
            indexes=combined_idx,
            original_length=dct_1.original_length,
            ts=dct_1.ts,
        )

    def __len__(self) -> int:
        return len(self.indexes)
