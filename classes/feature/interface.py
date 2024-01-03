from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

from classes.sample import SingleSample
from classes.signal import SignalTime


class FeatureExtractor(ABC, BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        return self

    def combine_axis(self, x: dict[str, np.ndarray]) -> np.ndarray:
        """Combines the axis results according to axis_to_combine.

        Args:
            x (dict): results per axis
        Returns:
            np.ndarray with dimensions (N, d), where
                - d = number of features
                - N = 1 if there are axis to combine, else 3
        """

        if len(self.axis_to_combine) > 0:
            data = np.array([v for k, v in x.items() if k in self.axis_to_combine])
            return np.sqrt(np.sum(data**2, axis=0, keepdims=True))
        else:
            return np.array(list(x.values()))

    def transform(self, single_samples: list[SingleSample], y=None) -> np.ndarray:
        """Transforms a batch of SingleSample objects."""

        y = []
        for single_sample in single_samples:
            feat_per_axis = {}
            signal_per_axis = single_sample.to_dict()
            for axis, signal in signal_per_axis.items():
                if not isinstance(signal, SignalTime):
                    signal: SignalTime = signal.to_time()
                features = self.apply_transform(signal)
                feat_per_axis[axis] = features

            combined = self.combine_axis(feat_per_axis)

            if self.normalize:
                combined /= combined.sum(axis=1, keepdims=True)
            y.append(combined)
        return np.array(y)

    @abstractmethod
    def apply_transform(self) -> np.ndarray:
        pass
