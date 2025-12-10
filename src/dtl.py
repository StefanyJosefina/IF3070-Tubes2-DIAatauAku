from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[Any] = None  
    is_categorical: bool = False  
    left: Optional[Any] = None
    right: Optional[Any] = None
    value: Optional[np.ndarray] = None 


class DTL(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        splitter: str = "best",
        max_features: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.random_state = random_state
        
        self._feature_types = None 

    def fit(self, X: np.ndarray, y: np.ndarray):
        if hasattr(X, 'to_numpy'):
            X_array = X.to_numpy()
            self._feature_types = {}
            for col_idx, col in enumerate(X.columns):
                col_data = X.iloc[:, col_idx]
                is_cat = not np.issubdtype(col_data.dtype, np.number)
                self._feature_types[col_idx] = 'categorical' if is_cat else 'numeric'
        else:
            X_array = np.asarray(X)
            self._feature_types = {}
            for col_idx in range(X_array.shape[1]):
                col_data = X_array[:, col_idx]
                try:
                    col_data.astype(np.float64)
                    self._feature_types[col_idx] = 'numeric'
                except (ValueError, TypeError):
                    self._feature_types[col_idx] = 'categorical'
        
        y = np.asarray(y)
        
        if X_array.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        y_encoded = np.searchsorted(self.classes_, y)
        
        self._tree = self._build_tree(X_array, y_encoded, depth=0)
        return self

    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity (gini or entropy) for labels y."""
        if y.size == 0:
            return 0.0
        
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        probs = counts / counts.sum()
        
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:  
            ps = probs[probs > 0]
            return -np.sum(ps * np.log2(ps))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        if n_samples < 2 or len(np.unique(y)) == 1:
            return None, None, 0.0, False
        
        base_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_is_categorical = False
        
        if self.max_features is None:
            features_to_try = range(n_features)
        elif self.max_features == "sqrt":
            n_try = max(1, int(np.sqrt(n_features)))
            features_to_try = np.random.choice(n_features, size=n_try, replace=False)
        elif self.max_features == "log2":
            n_try = max(1, int(np.log2(n_features)))
            features_to_try = np.random.choice(n_features, size=n_try, replace=False)
        else:
            features_to_try = range(n_features)
        
        for feature in features_to_try:
            values = X[:, feature]
            is_categorical = self._feature_types.get(feature, 'numeric') == 'categorical'
            
            if is_categorical:
                unique_vals = np.unique(values)
                if unique_vals.size < 2:
                    continue
                
                for category in unique_vals:
                    left_mask = values == category
                    right_mask = values != category
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    if y_left.size == 0 or y_right.size == 0:
                        continue
                    
                    n_left = y_left.size
                    n_right = y_right.size
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)
                    
                    weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                    gain = base_impurity - weighted_impurity
                    
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feature = feature
                        best_threshold = category
                        best_is_categorical = True
            else:
                try:
                    numeric_vals = values.astype(np.float64)
                except (ValueError, TypeError):
                    continue
                
                unique_vals = np.unique(numeric_vals)
                
                if unique_vals.size < 2:
                    continue
                
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                
                for threshold in thresholds:
                    left_mask = numeric_vals <= threshold
                    right_mask = ~left_mask
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    if y_left.size == 0 or y_right.size == 0:
                        continue
                    
                    n_left = y_left.size
                    n_right = y_right.size
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)
                    
                    weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                    gain = base_impurity - weighted_impurity
                    
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feature = feature
                        best_threshold = threshold
                        best_is_categorical = False
        
        return best_feature, best_threshold, best_gain, best_is_categorical

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node()
        n_samples = y.size
        unique_labels = np.unique(y)
        
        stopping = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(unique_labels) == 1
        )
        
        if stopping:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        if self.splitter == "random":
            n_features = X.shape[1]
            feat = np.random.randint(0, n_features)
            values = X[:, feat]
            is_cat = self._feature_types.get(feat, 'numeric') == 'categorical'
            unique_vals = np.unique(values)
            
            if unique_vals.size < 2:
                counts = np.bincount(y, minlength=self.n_classes_).astype(float)
                node.value = counts / counts.sum()
                return node
            
            if is_cat:
                category = unique_vals[np.random.randint(0, len(unique_vals))]
                best_feature, best_threshold, best_is_categorical = feat, category, True
            else:
                unique_vals = unique_vals.astype(np.float64)
                thr = (unique_vals[np.random.randint(0, len(unique_vals) - 1)] + 
                       unique_vals[np.random.randint(1, len(unique_vals))]) / 2.0
                best_feature, best_threshold, best_is_categorical = feat, thr, False
        else:  
            best_feature, best_threshold, gain, best_is_categorical = self._best_split(X, y)
        
        if best_feature is None:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        if best_is_categorical:
            left_mask = X[:, best_feature] == best_threshold
            right_mask = X[:, best_feature] != best_threshold
        else:
            left_mask = X[:, best_feature].astype(np.float64) <= best_threshold
            right_mask = ~left_mask
        
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_categorical = best_is_categorical
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        predictions = []
        
        for sample in X:
            node = self._tree
            while node.value is None:
                feature_value = sample[node.feature]
                
                if node.is_categorical:
                    if feature_value == node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    try:
                        numeric_value = float(feature_value)
                    except (ValueError, TypeError):
                        node = node.right
                        continue
                    
                    if numeric_value <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
            
            class_idx = np.argmax(node.value)
            predictions.append(self.classes_[class_idx])
        
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        proba = []
        
        for sample in X:
            node = self._tree
            while node.value is None:
                feature_value = sample[node.feature]
                
                if node.is_categorical:
                    if feature_value == node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    try:
                        numeric_value = float(feature_value)
                    except (ValueError, TypeError):
                        node = node.right
                        continue
                    
                    if numeric_value <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
            
            proba.append(node.value)
        
        return np.vstack(proba)


__all__ = ["DTL"]
