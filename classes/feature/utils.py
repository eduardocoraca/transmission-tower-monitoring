import numpy as np

from classes.feature.objects import Pole, Poles


def truncate_svd(
    U_full: np.ndarray, S_full: np.ndarray, V_full: np.ndarray, order: int
) -> tuple[np.ndarray]:
    S = S_full[:order, :][:, :order]
    V = V_full[:order, :]
    U = U_full[:, :order]
    return U, S, V


def solve_linear_system(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ai = np.linalg.pinv(A)
    return np.matmul(Ai, B)


def compute_svd(O: np.ndarray) -> tuple[np.ndarray]:
    U, S, V = np.linalg.svd(O, full_matrices=False)
    S = np.diag(S)
    return U, S, V


def compute_poles(Ad: np.ndarray, ts: float) -> list[Pole]:
    z, a = np.linalg.eig(Ad)
    return [Pole(z[k], a[:, k], 1 / ts) for k in range(len(z))]
