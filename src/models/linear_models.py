"""Linear regression models: OLS, Ridge, Lasso, ElasticNet."""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from .base import BaseModel, validate_input


class OLSModel(BaseModel):
    """Ordinary Least Squares using numpy.linalg.lstsq."""

    def __init__(self, params=None):
        super().__init__("ols", params)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        if len(y_clean) < 2:
            self.coef_ = np.zeros(X.shape[1], dtype=np.float32)
            self.intercept_ = 0.0
            return self
        # Add intercept column
        X_aug = np.column_stack([np.ones(len(X_clean)), X_clean])
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_aug, y_clean, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:].astype(np.float32)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_

    def get_params_dict(self):
        return {"model": "ols", "n_features": len(self.coef_) if self.coef_ is not None else 0}


class RidgeModel(BaseModel):
    """Ridge regression (L2 regularization)."""

    def __init__(self, params=None):
        super().__init__("ridge", params)
        self.model_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        alpha = self.params.get("alpha", 1.0)
        if len(y_clean) < 3:
            self.model_ = None
            # fallback to OLS
            ols = OLSModel()
            ols.fit(X, y)
            self.coef_ = ols.coef_
            self.intercept_ = ols.intercept_
            return self
        self.model_ = Ridge(alpha=alpha, fit_intercept=True)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model_ is None:
            return X @ self.coef_ + self.intercept_
        return self.model_.predict(X)


class LassoModel(BaseModel):
    """Lasso regression (L1 regularization)."""

    def __init__(self, params=None):
        super().__init__("lasso", params)
        self.model_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        alpha = self.params.get("alpha", 0.01)
        if len(y_clean) < 3:
            ols = OLSModel()
            ols.fit(X, y)
            self.coef_ = ols.coef_
            self.intercept_ = ols.intercept_
            return self
        self.model_ = Lasso(alpha=alpha, fit_intercept=True, max_iter=2000)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model_ is None:
            return X @ self.coef_ + self.intercept_
        return self.model_.predict(X)


class ElasticNetModel(BaseModel):
    """ElasticNet regression (L1 + L2 regularization)."""

    def __init__(self, params=None):
        super().__init__("elasticnet", params)
        self.model_ = None

    def fit(self, X, y):
        X_clean, y_clean, _ = validate_input(X, y)
        alpha = self.params.get("alpha", 0.01)
        l1_ratio = self.params.get("l1_ratio", 0.5)
        if len(y_clean) < 3:
            ols = OLSModel()
            ols.fit(X, y)
            self.coef_ = ols.coef_
            self.intercept_ = ols.intercept_
            return self
        self.model_ = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=2000)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.model_ is None:
            return X @ self.coef_ + self.intercept_
        return self.model_.predict(X)
