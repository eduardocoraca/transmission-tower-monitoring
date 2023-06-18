import networkx as nx
import numpy as np
from hmmlearn.hmm import GMMHMM, CategoricalHMM


def make_hmm(
    n_components: int,
    n_mix: int,
    tol: float = 1e-5,
    initialize: callable(int) = None,
    covariance_type: str = "full",
    min_covar: float = 1e-1,
    n_iter: int = 100,
):
    if callable(initialize):
        hmm = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            init_params="smcw",
            covariance_type=covariance_type,
            min_covar=1e-1,
            n_iter=100,
            tol=tol,
        )
        hmm.transmat_ = initialize(n_components)
    else:
        hmm = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            init_params="stmcw",
            covariance_type=covariance_type,
            min_covar=1e-1,
            n_iter=100,
            tol=tol,
        )
    return hmm


def make_categorical_hmm(
    n_components: int, n_features: int, initialize: callable(int) = None
):
    if callable(initialize):
        hmm = CategoricalHMM(
            n_components=n_components, init_params="smcw", n_features=n_features
        )
        hmm.transmat_ = initialize(n_components)
    else:
        hmm = CategoricalHMM(
            n_components=n_components, init_params="stmcw", n_features=n_features
        )
    return hmm


def train_hmm(
    x: np.ndarray,
    n_components: int,
    n_mix: int,
    tol: float = 1e-5,
    initialize: callable(int) = None,
    covariance_type: str = "full",
):
    if callable(initialize):
        hmm = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            init_params="smcw",
            covariance_type=covariance_type,
            min_covar=1e-1,
            n_iter=100,
            tol=tol,
        )
        hmm.transmat_ = initialize(n_components)
    else:
        hmm = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            init_params="stmcw",
            covariance_type=covariance_type,
            min_covar=1e-1,
            n_iter=100,
            tol=tol,
        )
    hmm.fit(x)
    return hmm


def initialize_left_right_common(n_components):
    A = np.zeros((n_components, n_components))
    A[-1, :] = np.ones(n_components) / n_components
    for i in range(n_components - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.25
        A[i, -1] = 0.25
    A[-2, -1] = 0.5
    return A


def initialize_neighbor(n_components):
    A = np.zeros((n_components, n_components))
    A[0, 0] = 0.5
    A[0, 1] = 0.5
    A[-1, -1] = 0.5
    A[-1, -2] = 0.5
    for i in range(1, n_components - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.25
        A[i, i - 1] = 0.25

    return A


def initialize_left_right(n_components):
    A = np.zeros((n_components, n_components))
    for i in range(n_components - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5
    A[-1, -1] = 1
    return A


def initialize_left_right_extra(n_components_double):
    n_components = n_components_double // 2
    A = np.zeros((2 * n_components, 2 * n_components))
    for i in range(n_components - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.25
        A[i, i + n_components] = 0.25
        A[i + n_components, i + n_components] = 0.5
        A[i + n_components, i] = 0.5
    A[n_components - 1, n_components - 1] = 0.5
    A[n_components - 1, 2 * n_components - 1] = 0.5
    A[-1, n_components - 1] = 0.5
    A[-1, -1] = 0.5
    return A


def plot_graph(ax, adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    edges = G.edges()
    weights = np.array([G[u][v]["weight"] for u, v in edges])
    weights_mod = weights
    weights_mod[weights >= 0.9] = 1.5
    weights_mod[(weights > 0.1) & (weights < 0.9)] = 0.75
    weights_mod[(weights > 0.01) & (weights < 0.1)] = 0.5
    weights_mod[(weights < 0.01) & ((weights > 0))] = 0.1
    weights_mod[weights == 0] = 0
    nx.draw(
        G,
        ax=ax,
        node_size=500,
        with_labels=True,
        width=weights_mod,
        node_color="grey",
        edgecolors="k",
        pos=nx.circular_layout(G),
    )
