from __future__ import annotations

from typing import Optional
from collections import Counter

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
        # Validate and store hyperparameters
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training features and labels.

        Should return `self` to be compatible with scikit-learn.
        """
        # KNN is a lazy learner, so we just store the training data 
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y) # Store unique classes for probability estimation
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None):
        """Return `(distances, indices)` of nearest neighbors.

        - `distances` shape: (n_queries, n_neighbors)
        - `indices` shape: (n_queries, n_neighbors)
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        X = np.array(X)
        n_queries = X.shape[0]
        
        # Initialize result arrays
        distances = np.zeros((n_queries, n_neighbors))
        indices = np.zeros((n_queries, n_neighbors), dtype=int)
        
        # Measure 'distance' of query (unseen data) to all instance (in training data) 
        for i in range(n_queries):
            # Calculate distance based on p-norm
            if self.p == 1:
                # Manhattan distance
                dists = np.sum(np.abs(self.X_train - X[i]), axis=1)
            elif self.p == 2:
                # Euclidean distance
                dists = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
            else:
                # General Minkowski distance
                dists = np.sum(np.abs(self.X_train - X[i])**self.p, axis=1)**(1/self.p)
            
            # Find k 'most similar' instances (indices of smallest distances) [cite: 174194]
            # argsort returns indices that would sort the array
            neighbor_indices = np.argsort(dists)[:n_neighbors]
            
            distances[i] = dists[neighbor_indices]
            indices[i] = neighbor_indices
            
        return distances, indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for `X` using k-nearest neighbors."""
        X = np.array(X)
        distances, indices = self.kneighbors(X)
        predictions = []
        
        for i in range(len(X)):
            # Get the labels of the k nearest neighbors
            neighbor_labels = self.y_train[indices[i]]
            
            if self.weights == 'uniform':
                # Find the majority class from k nearest neighbor [cite: 174196]
                most_common = Counter(neighbor_labels).most_common(1)
                predictions.append(most_common[0][0])
                
            elif self.weights == 'distance':
                # Weighted voting based on inverse distance
                class_weights = {cls: 0.0 for cls in self.classes_}
                
                # Check if any neighbor has distance 0 (exact match)
                if np.any(distances[i] == 0):
                    # If distance is 0, the weight would be infinite. 
                    # We classify based on the label of the 0-distance neighbor(s).
                    zero_dist_indices = np.where(distances[i] == 0)[0]
                    zero_dist_labels = neighbor_labels[zero_dist_indices]
                    most_common = Counter(zero_dist_labels).most_common(1)
                    predictions.append(most_common[0][0])
                else:
                    weights = 1.0 / distances[i]
                    for label, weight in zip(neighbor_labels, weights):
                        class_weights[label] += weight
                    
                    # Predict class with highest accumulated weight
                    predictions.append(max(class_weights, key=class_weights.get))
        
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for `X`."""
        X = np.array(X)
        distances, indices = self.kneighbors(X)
        probabilities = []
        
        for i in range(len(X)):
            neighbor_labels = self.y_train[indices[i]]
            class_probs = {cls: 0.0 for cls in self.classes_}
            
            if self.weights == 'uniform':
                # Probability is fraction of neighbors belonging to the class
                for label in neighbor_labels:
                    class_probs[label] += 1
                total = self.n_neighbors
                
            elif self.weights == 'distance':
                # Probability weighted by inverse distance
                if np.any(distances[i] == 0):
                    zero_dist_indices = np.where(distances[i] == 0)[0]
                    zero_dist_labels = neighbor_labels[zero_dist_indices]
                    
                    # Distribute probability 1.0 among zero distance neighbors
                    for label in zero_dist_labels:
                        class_probs[label] += 1
                    total = len(zero_dist_labels)
                else:
                    weights = 1.0 / distances[i]
                    for label, weight in zip(neighbor_labels, weights):
                        class_probs[label] += weight
                    total = np.sum(weights)
            
            # Normalize to get probabilities (sum to 1)
            prob_vector = [class_probs[cls] / total for cls in self.classes_]
            probabilities.append(prob_vector)
            
        return np.array(probabilities)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy (or delegate to sklearn.metrics)."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


__all__ = ["KNN"]