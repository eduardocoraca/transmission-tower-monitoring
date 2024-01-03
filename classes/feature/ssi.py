from dataclasses import dataclass

import numpy as np

from classes.feature.objects import Pole, Poles
from classes.signal import SignalTime

from .utils import (compute_poles, compute_svd, solve_linear_system,
                    truncate_svd)


@dataclass
class SSIMeta:
    svals: list[float]
    ad: np.ndarray
    order: int


class SSI:
    def __init__(self, max_order: int, sval_percent: float):
        self.max_order = max_order  # max order
        self.sval_percent = sval_percent

    def predict(self, signals: list[SignalTime]) -> list[Pole]:
        x_list = [signal.get()[1] for signal in signals]
        x_array = np.vstack(x_list)
        x_array = x_array - x_array.mean(axis=1, keepdims=True)
        Yhp, Yhf = self._get_block_hankel(x_array)
        O = self._get_projection_matrix(Yhf, Yhp)
        U_full, S_full, V_full = compute_svd(O)
        order = self._detect_order(S_full)
        U, S, V = truncate_svd(U_full, S_full, V_full, 2 * order)
        L = np.matmul(U, np.sqrt(S))
        # X0 = np.matmul(np.sqrt(S), U)
        ad = self._get_system_matrix(L)
        poles = compute_poles(ad, signals[0].ts)
        meta = SSIMeta(svals=list(np.diag(S_full)), ad=ad, order=order)
        return poles, meta

        # poles_list = []
        # for order in range(self.max_order):
        #     U, S, V = truncate_svd(U_full, S_full, V_full, 2 * order)
        #     L = np.matmul(U, np.sqrt(S))
        #     # X0 = np.matmul(np.sqrt(S), U)
        #     self.Ad = self._get_system_matrix(L)
        #     poles = compute_poles(self.Ad, signals[0].ts)
        #     poles_list.append(poles)

        # return poles, poles_list, S_full

    def _detect_order(self, s_array: np.ndarray) -> int:
        svals = np.diag(s_array)
        svals_cumsum = np.cumsum(svals) / svals.sum()
        return int(np.where(svals_cumsum >= self.sval_percent)[0][0])

    def _get_block_hankel(self, Y: np.ndarray) -> tuple[np.ndarray]:
        M, N = Y.shape
        s2 = 2 * self.max_order
        Yh = np.vstack([Y[:, k : N - s2 + k] for k in range(0, s2)])
        Yhp = Yh[0 : M * self.max_order]
        Yhf = Yh[M * self.max_order :]
        return Yhp, Yhf

    def _get_projection_matrix(self, Yhf: np.ndarray, Yhp: np.ndarray) -> np.ndarray:
        Yinv = np.linalg.pinv(np.matmul(Yhp, Yhp.T))
        return np.matmul(Yhf, np.matmul(Yhp.T, np.matmul(Yinv, Yhp)))

    def _get_system_matrix(self, L: np.ndarray) -> np.ndarray:
        m = L.shape[0] // self.max_order
        L2s = L[m:]
        L1s_1 = L[0 : (self.max_order - 1) * m]
        return solve_linear_system(L2s, L1s_1)
