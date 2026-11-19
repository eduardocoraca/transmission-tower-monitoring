from typing import List
from pydantic import BaseModel
import numpy as np
from scipy.stats import skew, kurtosis


class StatisticalFeatures(BaseModel):
    mean: float
    std: float
    rms: float
    skewness: float
    kurtosis: float
    shape_factor: float
    crest_factor: float
    peak_to_peak: float
    entropy: float
    energy: float

    @staticmethod
    def _mean(x: List[float]) -> float:
        return float(np.mean(x))

    @staticmethod
    def _std(x: List[float]) -> float:
        return float(np.std(x))

    @staticmethod
    def _rms(x: List[float]) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    @staticmethod
    def _skewness(x: List[float]) -> float:
        return float(skew(x))

    @staticmethod
    def _kurtosis(x: List[float]) -> float:
        return float(kurtosis(x))

    @staticmethod
    def _shape_factor(x: List[float]) -> float:
        rms = StatisticalFeatures._rms(x)
        abs_mean = StatisticalFeatures._mean(np.abs(x).tolist())
        return rms / abs_mean if abs_mean != 0 else 0.0

    @staticmethod
    def _crest_factor(x: List[float]) -> float:
        peak = float(np.max(np.abs(x)))
        rms = StatisticalFeatures._rms(x)
        return peak / rms if rms != 0 else 0.0

    @staticmethod
    def _peak_to_peak(x: List[float]) -> float:
        return float(np.ptp(x))

    @staticmethod
    def _entropy(x: List[float]) -> float:
        histogram, bin_edges = np.histogram(x, bins=10, density=True)
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))

    @staticmethod
    def _energy(x: List[float]) -> float:
        return float(np.sum(np.square(x)))

    @classmethod
    def from_signal(cls, signal: List[float]) -> "StatisticalFeatures":
        return cls(
            mean=StatisticalFeatures._mean(signal),
            std=StatisticalFeatures._std(signal),
            rms=StatisticalFeatures._rms(signal),
            skewness=StatisticalFeatures._skewness(signal),
            kurtosis=StatisticalFeatures._kurtosis(signal),
            shape_factor=StatisticalFeatures._shape_factor(signal),
            crest_factor=StatisticalFeatures._crest_factor(signal),
            peak_to_peak=StatisticalFeatures._peak_to_peak(signal),
            entropy=StatisticalFeatures._entropy(signal),
            energy=StatisticalFeatures._energy(signal),
        )
