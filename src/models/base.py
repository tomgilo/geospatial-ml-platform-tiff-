"""Abstract base class for all pixel-level models."""

from abc import ABC, abstractmethod
import time
import numpy as np


class BaseModel(ABC):
    """Interface that all pixel models must implement."""

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.is_fitted = False
        self.train_time = 0.0

    @abstractmethod
    def fit(self, X, y):
        """Fit model on (n_samples, n_features) X and (n_samples,) y."""

    @abstractmethod
    def predict(self, X):
        """Predict on (n_samples, n_features) X, return (n_samples,) array."""

    def fit_timed(self, X, y):
        """Fit with timing."""
        t0 = time.perf_counter()
        self.fit(X, y)
        self.train_time = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def get_params_dict(self):
        """Return model parameters as a flat dict for reporting."""
        return {"model": self.name, **self.params}


def validate_input(X, y):
    """Remove rows where X or y is NaN. Returns clean X, y and the valid mask."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    return X[valid], y[valid], valid
