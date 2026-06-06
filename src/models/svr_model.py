"""Support Vector Regression model."""
import numpy as np
from .base import BaseModel, validate_input

class SVRModel(BaseModel):
    """Support Vector Regressor."""

    def __init__(self, params=None):
        super().__init__("svr", params)
        self.model_ = None

    def fit(self, X, y):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        X_clean, y_clean, _ = validate_input(X, y)
        kernel = self.params.get("kernel", "rbf")
        C = self.params.get("C", 1.0)
        epsilon = self.params.get("epsilon", 0.1)
        gamma = self.params.get("gamma", "scale")
        if len(y_clean) < 5:
            self.model_ = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, max_iter=1000)
        else:
            self.model_ = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, max_iter=2000, cache_size=200)
        y_std = np.std(y_clean) or 1.0
        self.y_mean_ = np.mean(y_clean)
        self.y_std_ = y_std
        y_scaled = (y_clean - self.y_mean_) / self.y_std_
        self.x_scaler_ = StandardScaler()
        X_scaled = self.x_scaler_.fit_transform(X_clean)
        self.model_.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.x_scaler_.transform(X)
        y_pred_scaled = self.model_.predict(X_scaled)
        return y_pred_scaled * self.y_std_ + self.y_mean_
