from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y=None):
        return self

    def transform(self, x: np.ndarray, y=None) -> np.ndarray:
        return np.log(x + 1e-5)


class KMeansQuantizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int, per_feature: bool = False):
        self.n_clusters = n_clusters
        self.per_feature = per_feature

    def fit(self, x: np.ndarray, y=None):
        """Fits the clusters.
        Args:
            x (np.ndarray): input features with dim. (n_samples, n_features)
            per_feature (bool): quantize features independently True
        """
        self.clusters = []
        if self.per_feature:
            self.n_features = x.shape[1]
            for xn in x.T:
                model = KMeans(self.n_clusters)
                model.fit(xn.reshape(-1, 1))
                self.clusters.append(model)
        else:
            self.clusters = KMeans(self.n_clusters).fit(x)
            self.centers = self.clusters.cluster_centers_
        return self

    def transform(
        self, x: np.ndarray, y=None, return_centers: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the corresponding label and center (if True) of each sample."""
        if self.per_feature:
            labels = []
            centers = []
            for n, xn in enumerate(x.T):
                y = self.clusters[n].predict(xn.reshape(-1, 1))
                c = self.clusters[n].cluster_centers_[y].reshape(-1)
                centers.append(c)
                labels.append(y)
            labels = np.array(labels).T
            centers = np.array(centers).T
        else:
            labels = self.clusters.predict(x)
            centers = self.centers[labels]
        if return_centers:
            return centers, labels
        else:
            return labels.reshape(-1, 1)
