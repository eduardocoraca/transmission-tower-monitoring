import numpy as np
from pywt import WaveletPacket
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from lib.signal_classes import SignalTime


class WPTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        family: str = "haar",
        reduction: str = "energy",
        level: int = 4,
        normalize: bool = False,
        threshold: float = None,
    ):
        super().__init__()
        self.family = family
        self.reduction = reduction
        self.level = level
        self.normalize = normalize
        self.threshold = threshold

    def rms_reduction(self, x: np.ndarray) -> float:
        return np.sqrt(np.mean(x**2))

    def energy_reduction(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    def combine_axis(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(x**2, axis=0))

    def fit(self, x: np.ndarray = None, y: np.ndarray = None):
        return self

    def transform(self, x: np.ndarray, y=None):
        if self.reduction == "energy":
            reduction = self.energy_reduction
        elif self.reduction == "rms":
            reduction = self.rms_reduction
        else:
            raise Exception("Unrecognized reduction.")

        y = []
        for xk in x:
            yk = []
            for xj in xk:
                if not isinstance(xj, SignalTime):
                    xj = xj.to_time()
                xj = np.array(xj.get()[1])
                wp = WaveletPacket(data=xj, wavelet=self.family)
                yj = np.array(
                    [
                        reduction(np.array(node.data))
                        for node in wp.get_level(self.level, order="freq")
                    ]
                )
                yk.append(yj)
            yk = self.combine_axis(np.array(yk))
            if self.normalize:
                yk /= yk.sum()
                if isinstance(self.threshold, float):
                    sorted_idx = np.argsort(yk)
                    sorted_idx = np.flipud(sorted_idx)
                    sorted_y_cs = np.cumsum(yk[sorted_idx])
                    sorted_idx_to_zero = np.argwhere(
                        sorted_y_cs > self.threshold
                    ).flatten()
                    if len(sorted_idx_to_zero) > 0:
                        idx_to_zero = sorted_idx[sorted_idx_to_zero]
                        yk[idx_to_zero] = 0
            y.append(yk)

        y = np.array(y)
        # if self.normalize:
        #     y /= y.sum(axis=1, keepdims=True)
        # if isinstance(self.threshold, float):
        #     sorted_idx = np.argsort(y)
        #     sorted_idx = np.flipud(sorted_idx)
        #     sorted_y_cs = np.cumsum(y[sorted_idx])
        #     sorted_idx_to_zero = np.argwhere(sorted_y_cs > self.threshold).flatten()
        #     if len(sorted_idx_to_zero) > 0:
        #         idx_to_zero = sorted_idx[sorted_idx_to_zero]
        #         y[idx_to_zero] = 0
        return y

    def get_tree(self, signal: SignalTime):
        x = np.array(signal.get()[1])


class PCAModel(BaseEstimator, TransformerMixin):
    def __init__(self, cumulative: float = 0.95):
        super().__init__()
        self.cumulative = cumulative

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        self.model = PCA()
        self.model.fit(x)
        return self

    def transform(self, x: np.ndarray, y: np.ndarray = None):
        return self.model.transform(x)

    def predict(self, x: np.ndarray, y: np.ndarray = None):
        cum_sum = np.cumsum(self.model.explained_variance_ratio_)
        return np.argwhere(cum_sum <= self.cumulative).flatten()[-1]
