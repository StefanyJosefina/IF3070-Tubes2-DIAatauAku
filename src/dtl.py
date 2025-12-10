from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class _Node:
    # TODO: document node fields
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[Any] = None
    right: Optional[Any] = None
    value: Optional[np.ndarray] = None  # class distribution at leaf


class DTL(BaseEstimator, ClassifierMixin):
    """Skeleton Decision Tree classifier.

    TODOs:
    - Implement `fit` to build the tree and store class mapping.
    - Implement impurity measures (`gini` and `entropy`).
    - Implement `_best_split` and `_build_tree` recursion.
    - Implement `predict` and `predict_proba` to traverse the tree.
    - Add input validation and support for hyperparameters.
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, criterion: str = "gini"):
        # TODO: validate args
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray):
        """TODO: build the decision tree from (X, y) and return self."""
        raise NotImplementedError("TODO: implement fit to build the tree")

    def _impurity(self, y: np.ndarray) -> float:
        """TODO: compute impurity (gini or entropy) for labels y."""
        raise NotImplementedError("TODO: implement impurity calculation")

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """TODO: find best feature and threshold to split the node."""
        raise NotImplementedError("TODO: implement best split search")

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """TODO: recursive tree builder returning a `_Node` structure."""
        raise NotImplementedError("TODO: implement tree building recursion")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO: predict class labels for X by traversing the tree."""
        raise NotImplementedError("TODO: implement predict")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """TODO: return class probabilities for each sample in X."""
        raise NotImplementedError("TODO: implement predict_proba")


__all__ = ["DTL"]
