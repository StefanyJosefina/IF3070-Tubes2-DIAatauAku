from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNN(BaseEstimator, ClassifierMixin):
    """Skeleton KNN estimator (from scratch).

    TODOs:
    - Store training data in `fit`.
    - Compute pairwise distances using p-norm in `kneighbors`.
    - Implement `predict` using majority / weighted vote.
    - Implement `predict_proba` that returns class probabilities.
    - Optionally implement `score` as accuracy.
    
    Hyperparameters for tuning:
    - n_neighbors: [3, 5, 7, 9, 11]
    - weights: ['uniform', 'distance']
    - p: [1, 2] (manhattan, euclidean; Minkowski distance)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: int = 2,
    ):
        # TODO: validate and store hyperparameters
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """TODO: store training features and labels.

        Should return `self` to be compatible with scikit-learn.
        """
        raise NotImplementedError("TODO: implement fit to store training data")

    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None):
        """TODO: return `(distances, indices)` of nearest neighbors.

        - `distances` shape: (n_queries, n_neighbors)
        - `indices` shape: (n_queries, n_neighbors)
        """
        raise NotImplementedError("TODO: implement kneighbors")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO: predict class labels for `X` using k-nearest neighbors."""
        raise NotImplementedError("TODO: implement predict")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """TODO: return class probability estimates for `X`."""
        raise NotImplementedError("TODO: implement predict_proba")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """TODO: compute accuracy (or delegate to sklearn.metrics)."""
        raise NotImplementedError("TODO: implement score")


__all__ = ["KNN"]
