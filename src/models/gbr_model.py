"""Gradient Boosting Regressor."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .base import BaseModel, validate_input

class GBRModel(BaseModel):
    """Gradient Boosting Regressor."""

    def __init__(self, params=None):
        super().__init__("gbr", params)
        self.model_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        n_estimators = self.params.get("n_estimators", 100)
        max_depth = self.params.get("max_depth", 3)
        learning_rate = self.params.get("learning_rate", 0.1)
        subsample = self.params.get("subsample", 1.0)
        min_samples_split = self.params.get("min_samples_split", 2)
        min_samples_leaf = self.params.get("min_samples_leaf", 1)
        if len(y_clean) < 3:
            self.model_ = None
            self.mean_ = np.mean(y_clean) if len(y_clean) > 0 else 0.0
            return self
        self.model_ = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model_ is None:
            return np.full(X.shape[0], self.mean_, dtype=np.float32)
        return self.model_.predict(X)

    def get_feature_importance(self):
        if self.model_ is not None and hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        return None
