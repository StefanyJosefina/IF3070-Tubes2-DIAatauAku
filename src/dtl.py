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
    n_samples: int = 0


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
        max_thresholds: int = 256,
        class_weight: Optional[dict] = None,
        min_impurity_decrease: float = 0.0,
    ):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.random_state = random_state
        self.max_thresholds = max_thresholds
        self.class_weight = class_weight
        self.min_impurity_decrease = min_impurity_decrease
        self._feature_types = None
        self.feature_importances_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
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
        
        # Compute class weights - FIXED VERSION
        self._compute_class_weights(y_encoded)
        
        # Compute sample weights for each sample based on its class
        self.sample_weights_ = np.ones(len(y_encoded))
        for i, class_idx in enumerate(y_encoded):
            self.sample_weights_[i] = self.class_weights_[class_idx]
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(X_array.shape[1])
        
        self._tree = self._build_tree(X_array, y_encoded, self.sample_weights_, depth=0)
        
        # Normalize feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
            
        return self

    def _compute_class_weights(self, y: np.ndarray):
        """Compute class weights for cost-sensitive learning"""
        if self.class_weight == "balanced":
            n_samples = len(y)
            counts = np.bincount(y, minlength=self.n_classes_)
            # Standard balanced formula
            self.class_weights_ = n_samples / (self.n_classes_ * counts)
        elif isinstance(self.class_weight, dict):
            self.class_weights_ = np.array([
                self.class_weight.get(self.classes_[i], 1.0) 
                for i in range(self.n_classes_)
            ])
        else:
            self.class_weights_ = np.ones(self.n_classes_)

    def _weighted_impurity(self, y: np.ndarray, sample_weights: np.ndarray) -> float:
        """Calculate weighted impurity - FIXED VERSION"""
        if y.size == 0:
            return 0.0
        
        # Count samples per class, weighted
        weighted_counts = np.zeros(self.n_classes_)
        for class_idx in range(self.n_classes_):
            mask = (y == class_idx)
            weighted_counts[class_idx] = sample_weights[mask].sum()
        
        total_weight = weighted_counts.sum()
        if total_weight == 0:
            return 0.0
        
        # Calculate probabilities
        probs = weighted_counts / total_weight
        
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:  # entropy
            # Avoid log(0)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

    def _best_split(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray):
        n_samples, n_features = X.shape
        
        if n_samples < 2 or len(np.unique(y)) == 1:
            return None, None, 0.0, False
        
        base_impurity = self._weighted_impurity(y, sample_weights)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_is_categorical = False
        
        # Determine features to try
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
        
        total_weight = sample_weights.sum()
        
        for feature in features_to_try:
            values = X[:, feature]
            is_categorical = self._feature_types.get(feature, 'numeric') == 'categorical'
            
            if is_categorical:
                unique_vals = np.unique(values)
                if unique_vals.size < 2:
                    continue
                
                for category in unique_vals:
                    left_mask = values == category
                    right_mask = ~left_mask
                    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    weights_left = sample_weights[left_mask]
                    weights_right = sample_weights[right_mask]
                    
                    weight_left = weights_left.sum()
                    weight_right = weights_right.sum()
                    
                    impurity_left = self._weighted_impurity(y_left, weights_left)
                    impurity_right = self._weighted_impurity(y_right, weights_right)
                    
                    # Weighted average impurity
                    weighted_impurity = (
                        (weight_left / total_weight) * impurity_left + 
                        (weight_right / total_weight) * impurity_right
                    )
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
                
                # Select thresholds
                if unique_vals.size > self.max_thresholds:
                    quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
                    thresholds = np.quantile(numeric_vals, quantiles)
                    thresholds = np.unique(thresholds)
                else:
                    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                
                for threshold in thresholds:
                    left_mask = numeric_vals <= threshold
                    right_mask = ~left_mask
                    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    weights_left = sample_weights[left_mask]
                    weights_right = sample_weights[right_mask]
                    
                    weight_left = weights_left.sum()
                    weight_right = weights_right.sum()
                    
                    impurity_left = self._weighted_impurity(y_left, weights_left)
                    impurity_right = self._weighted_impurity(y_right, weights_right)
                    
                    weighted_impurity = (
                        (weight_left / total_weight) * impurity_left + 
                        (weight_right / total_weight) * impurity_right
                    )
                    gain = base_impurity - weighted_impurity
                    
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feature = feature
                        best_threshold = threshold
                        best_is_categorical = False
        
        return best_feature, best_threshold, best_gain, best_is_categorical

    def _build_tree(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, depth: int) -> _Node:
        node = _Node()
        n_samples = y.size
        node.n_samples = n_samples
        unique_labels = np.unique(y)
        
        stopping = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(unique_labels) == 1
        )
        
        if stopping:
            # Weighted class distribution
            weighted_counts = np.zeros(self.n_classes_)
            for class_idx in range(self.n_classes_):
                mask = (y == class_idx)
                weighted_counts[class_idx] = sample_weights[mask].sum()
            
            total = weighted_counts.sum()
            if total > 0:
                node.value = weighted_counts / total
            else:
                node.value = np.ones(self.n_classes_) / self.n_classes_
            return node
        
        if self.splitter == "random":
            # Simplified random splitter
            n_features = X.shape[1]
            feat = np.random.randint(0, n_features)
            values = X[:, feat]
            is_cat = self._feature_types.get(feat, 'numeric') == 'categorical'
            unique_vals = np.unique(values)
            
            if unique_vals.size < 2:
                weighted_counts = np.zeros(self.n_classes_)
                for class_idx in range(self.n_classes_):
                    mask = (y == class_idx)
                    weighted_counts[class_idx] = sample_weights[mask].sum()
                node.value = weighted_counts / weighted_counts.sum()
                return node
            
            if is_cat:
                category = unique_vals[np.random.randint(0, len(unique_vals))]
                best_feature, best_threshold, best_is_categorical = feat, category, True
            else:
                unique_vals = unique_vals.astype(np.float64)
                thr = (unique_vals[np.random.randint(0, len(unique_vals) - 1)] + 
                       unique_vals[np.random.randint(1, len(unique_vals))]) / 2.0
                best_feature, best_threshold, best_is_categorical = feat, thr, False
            gain = 0.0
        else:
            best_feature, best_threshold, gain, best_is_categorical = self._best_split(X, y, sample_weights)
        
        # Check minimum impurity decrease
        if best_feature is None or gain < self.min_impurity_decrease:
            weighted_counts = np.zeros(self.n_classes_)
            for class_idx in range(self.n_classes_):
                mask = (y == class_idx)
                weighted_counts[class_idx] = sample_weights[mask].sum()
            node.value = weighted_counts / weighted_counts.sum()
            return node
        
        # Create split
        if best_is_categorical:
            left_mask = X[:, best_feature] == best_threshold
            right_mask = ~left_mask
        else:
            left_mask = X[:, best_feature].astype(np.float64) <= best_threshold
            right_mask = ~left_mask
        
        # Check min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            weighted_counts = np.zeros(self.n_classes_)
            for class_idx in range(self.n_classes_):
                mask = (y == class_idx)
                weighted_counts[class_idx] = sample_weights[mask].sum()
            node.value = weighted_counts / weighted_counts.sum()
            return node
        
        # Update feature importance
        self.feature_importances_[best_feature] += gain * sample_weights.sum()
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_categorical = best_is_categorical
        node.left = self._build_tree(X[left_mask], y[left_mask], sample_weights[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], sample_weights[right_mask], depth + 1)
        
        return node

    def _get_node_predictions(self, node: _Node, X: np.ndarray, indices: np.ndarray, results: np.ndarray, mode='class'):
        if node.value is not None:
            if mode == 'class':
                pred_class = self.classes_[np.argmax(node.value)]
                results[indices] = pred_class
            else:
                results[indices] = node.value
            return

        X_subset = X[indices, node.feature]

        if node.is_categorical:
            left_mask = X_subset == node.threshold
            right_mask = ~left_mask
        else:
            try:
                X_numeric = X_subset.astype(np.float64)
                left_mask = X_numeric <= node.threshold
            except (ValueError, TypeError):
                left_mask = np.zeros(X_subset.shape, dtype=bool)
            
            right_mask = ~left_mask

        if np.any(left_mask):
            self._get_node_predictions(node.left, X, indices[left_mask], results, mode)
        
        if np.any(right_mask):
            self._get_node_predictions(node.right, X, indices[right_mask], results, mode)

    def predict(self, X: np.ndarray, threshold: dict = None) -> np.ndarray:
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        n_samples = X.shape[0]
        proba = np.empty((n_samples, self.n_classes_), dtype=np.float64)
        
        self._get_node_predictions(self._tree, X, np.arange(n_samples), proba, mode='proba')
        
        if threshold is None:
            class_indices = np.argmax(proba, axis=1)
            return self.classes_[class_indices]
        
        final_predictions = self.classes_[np.argmax(proba, axis=1)]
        
        for cls_label, thr in threshold.items():
            if cls_label in self.classes_:
                cls_idx = np.where(self.classes_ == cls_label)[0][0]
                mask = proba[:, cls_idx] >= thr
                final_predictions[mask] = cls_label
                
        return final_predictions

    def predict_proba(self, X):
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        n_samples = X.shape[0]
        proba = np.empty((n_samples, self.n_classes_), dtype=np.float64)
        
        self._get_node_predictions(self._tree, X, np.arange(n_samples), proba, mode='proba')
        
        return proba


__all__ = ["DTL"]