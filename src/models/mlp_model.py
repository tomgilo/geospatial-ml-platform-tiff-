"""Multi-Layer Perceptron model."""
import numpy as np
from .base import BaseModel, validate_input

class MLPModel(BaseModel):
    """MLP Regressor."""

    def __init__(self, params=None):
        super().__init__("mlp", params)
        self.model_ = None

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        X_clean, y_clean, _ = validate_input(X, y)
        hidden_layers = self.params.get("hidden_layer_sizes", [64, 32])
        max_iter = self.params.get("max_iter", 500)
        activation = self.params.get("activation", "relu")
        alpha = self.params.get("alpha", 0.0001)
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        X_scaled = self.x_scaler_.fit_transform(X_clean)
        y_scaled = self.y_scaler_.fit_transform(y_clean.reshape(-1, 1)).ravel()
        self.model_ = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers), max_iter=max_iter,
            activation=activation, alpha=alpha, random_state=42,
            early_stopping=(len(y_clean) >= 20), n_iter_no_change=10
        )
        self.model_.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.x_scaler_.transform(X)
        y_pred_scaled = self.model_.predict(X_scaled)
        return self.y_scaler_.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
