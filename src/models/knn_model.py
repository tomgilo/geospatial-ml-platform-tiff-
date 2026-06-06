"""K-Nearest Neighbors regressor."""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from .base import BaseModel, validate_input

class KNNModel(BaseModel):
    """KNN Regressor."""

    def __init__(self, params=None):
        super().__init__("knn", params)
        self.model_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        n_neighbors = min(self.params.get("n_neighbors", 5), len(y_clean) - 1)
        weights = self.params.get("weights", "uniform")
        if n_neighbors < 1:
            n_neighbors = 1
        if len(y_clean) < 2:
            self.model_ = None
            self.mean_ = np.mean(y_clean) if len(y_clean) > 0 else 0.0
            return self
        self.model_ = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=1)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model_ is None:
            return np.full(X.shape[0], self.mean_, dtype=np.float32)
        return self.model_.predict(X)
