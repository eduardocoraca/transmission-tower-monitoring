import sys

sys.path.append("..")
sys.path.append(".")
from typing import Dict, List

import librosa
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from pywt import WaveletPacket
from scipy.signal import TransferFunction, welch

from classes.signal import SignalTime
from training.utils import get_filename

# from sklearn.base import BaseEstimator, TransformerMixin
from .interface import FeatureExtractor
from .objects import Poles

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


class PassThroughFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    # def transform(self, x: np.ndarray, y=None) -> np.ndarray:
    #     return x

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        return signal


class RMSFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        axis_to_combine: List[str] = [],
    ):
        super().__init__()
        self.normalize = False
        self.axis_to_combine = axis_to_combine

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        if not isinstance(signal, SignalTime):
            raise Exception("Signal must be of type SignalTime.")
        reduction = lambda x: np.sqrt(np.mean(x**2))
        _, x = signal.get()
        x = np.array(x)
        return np.array([reduction(x)])


class WPTFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        family: str = "haar",
        level: int = 4,
        normalize: bool = False,
        axis_to_combine: List[str] = [],
    ):
        super().__init__()
        self.family = family
        self.axis_to_combine = axis_to_combine
        self.level = level
        self.normalize = normalize

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        if not isinstance(signal, SignalTime):
            raise Exception("Signal must be of type SignalTime.")
        reduction = lambda x: np.sum(x**2)
        _, x = signal.get()
        wp = WaveletPacket(data=np.array(x), wavelet=self.family)
        return np.array(
            [
                reduction(np.array(node.data))
                for node in wp.get_level(self.level, order="freq")
            ]
        )


class PSDFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        normalize: bool = False,
        nperseg: int = 1024,
        poverlap: float = 0.5,
        axis_to_combine: List[str] = [],
        window: str = "hamming",
    ):
        super().__init__()
        self.nperseg = nperseg
        self.poverlap = poverlap
        self.window = window
        self.axis_to_combine = axis_to_combine
        self.normalize = normalize

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        if not isinstance(signal, SignalTime):
            raise Exception("Signal must be of type SignalTime.")
        spectrum = signal.to_spectrum(
            window=self.window, nperseg=self.nperseg, poverlap=self.poverlap
        )
        f, X = spectrum.get()
        return np.array(X[1:])  # avoid zero frequency


class LPCFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        m: int = 5,
        axis_to_combine: List[str] = [],
    ):
        super().__init__()
        self.axis_to_combine = []  # unable to combine
        self.m = m
        self.normalize = False

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        if not isinstance(signal, SignalTime):
            raise Exception("Signal must be of type SignalTime.")
        t, x = signal.get()
        w = librosa.lpc(np.array(x), order=self.m)
        z = np.roots(w)
        a = np.angle(z)
        f = a / (2 * np.pi * signal.ts)
        r = scipy.signal.residue([1], w)
        return Poles(z=z, w=w, f=f, r=r)


# combined


class PCAFeatureReducer:
    def __init__(self, family: str, level: int, dimension: int):
        self.family = family
        self.level = level
        self.dimension = dimension

        self.pca_per_cable = self._load_pre_trained()

    def _load_pre_trained(self) -> dict[str, any]:
        params = {"family": self.family, "level": self.level, "dim": self.dimension}
        cables = [1, 2, 3, 4]
        models_per_cable = {}
        for c in cables:
            filename = get_filename("pcawpt", c, **params)
            models_per_cable[c] = mlflow.sklearn.load_model(
                model_uri=f"models:/{filename}/latest"
            )
        return models_per_cable

    def transform(self, x: np.ndarray, cable: int) -> np.ndarray:
        pca_features = self.pca_per_cable[cable].transform(x)
        return pca_features


#################### dev


class LPCModel(torch.nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.layer = torch.nn.Conv1d(1, 1, m)

    def __call__(self, x: torch.Tensor):
        return self.layer(x)


class LPCModel2(torch.nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.layer = torch.nn.Linear(m, 1, bias=False)

    def __call__(self, x: torch.Tensor):
        return self.layer(x)


class LPCFeatureExtractorTorch(FeatureExtractor):
    """Implements the LPC feature extractor such that the weights are
    restricted to be close to a reference vector.
    """

    def __init__(
        self,
        m: int = 5,
        axis_to_combine: List[str] = [],
    ):
        super().__init__()
        self.axis_to_combine = []  # unable to combine
        self.m = m
        self.normalize = False

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, w_ref: torch.Tensor
    ):
        return (x**2 - y**2).mean() + (w**2 - w_ref**2).mean()

    def fit(self, x: np.ndarray, w_ref: np.ndarray):
        order = self.m
        x = torch.Tensor(x)
        w_ref = torch.Tensor(w_ref)
        num_epochs = 100
        self.model = LPCModel(self.m)
        x_tensor = torch.Tensor(x).unsqueeze(0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(num_epochs):
            x_pred = self.model(x_tensor)
            y = x_tensor[:, self.m - 1 :]
            w = list(self.model.parameters())[0].data.squeeze()
            loss = self.loss(x_pred, y, w, w_ref)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def get_weights(self):
        return list(self.model.parameters())[0].data.detach().numpy()

    def apply_transform(self, signal: SignalTime) -> np.ndarray:
        if not isinstance(signal, SignalTime):
            raise Exception("Signal must be of type SignalTime.")
        t, x = signal.get()
        win = np.hanning(len(x))
        w_ref = np.zeros(self.m)
        self.fit(x, w_ref)
        w = self.get_weights()
        # a = librosa.lpc(win * np.array(x), order=self.m)
        # a = np.flipud(w.squeeze())
        a = w.squeeze()
        z = np.roots(a)
        w = np.angle(z)
        f = w / (2 * np.pi * signal.ts)
        return f
