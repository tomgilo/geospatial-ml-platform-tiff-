"""Tree-based ensemble models: Random Forest, XGBoost, LightGBM."""

import numpy as np
from .base import BaseModel, validate_input


class RandomForestModel(BaseModel):
    """Random Forest Regressor."""

    def __init__(self, params=None):
        super().__init__("rf", params)
        self.model_ = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        X_clean, y_clean, _ = validate_input(X, y)
        if len(y_clean) < 5:
            self.model_ = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        else:
            n_estimators = self.params.get("n_estimators", 100)
            max_depth = self.params.get("max_depth", 10)
            min_samples_split = self.params.get("min_samples_split", 2)
            min_samples_leaf = self.params.get("min_samples_leaf", 1)
            max_features = self.params.get("max_features", "sqrt")
            if isinstance(max_features, str) and max_features.replace(".", "").isdigit():
                max_features = float(max_features)
            self.model_ = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, n_jobs=1, random_state=42
            )
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.predict(X)

    def get_feature_importance(self):
        if self.model_ is not None and hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        return None


class XGBoostModel(BaseModel):
    """XGBoost Regressor."""

    def __init__(self, params=None):
        super().__init__("xgboost", params)
        self.model_ = None

    def fit(self, X, y):
        import xgboost as xgb
        X_clean, y_clean, _ = validate_input(X, y)
        n_estimators = self.params.get("n_estimators", 100)
        max_depth = self.params.get("max_depth", 6)
        learning_rate = self.params.get("learning_rate", 0.1)
        subsample = self.params.get("subsample", 1.0)
        reg_lambda = self.params.get("reg_lambda", 1.0)
        reg_alpha = self.params.get("reg_alpha", 0.0)
        self.model_ = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            n_jobs=1, verbosity=0, random_state=42
        )
        if len(y_clean) < 3:
            self.model_ = xgb.XGBRegressor(n_estimators=10, max_depth=2, n_jobs=1, verbosity=0)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.predict(X)

    def get_feature_importance(self):
        if self.model_ is not None:
            return self.model_.feature_importances_
        return None


class LightGBMModel(BaseModel):
    """LightGBM Regressor."""

    def __init__(self, params=None):
        super().__init__("lightgbm", params)
        self.model_ = None

    def fit(self, X, y):
        import lightgbm as lgb
        X_clean, y_clean, _ = validate_input(X, y)
        n_estimators = self.params.get("n_estimators", 100)
        max_depth = self.params.get("max_depth", 6)
        learning_rate = self.params.get("learning_rate", 0.1)
        num_leaves = self.params.get("num_leaves", 31)
        subsample = self.params.get("subsample", 1.0)
        reg_lambda = self.params.get("reg_lambda", 0.0)
        reg_alpha = self.params.get("reg_alpha", 0.0)
        min_child_samples = self.params.get("min_child_samples", 20)
        self.model_ = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, num_leaves=num_leaves,
            subsample=subsample, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            min_child_samples=min_child_samples,
            n_jobs=1, random_state=42, verbose=-1
        )
        if len(y_clean) < 3:
            self.model_ = lgb.LGBMRegressor(n_estimators=10, max_depth=2, verbose=-1)
        self.model_.fit(X_clean, y_clean)
        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.predict(X)

    def get_feature_importance(self):
        if self.model_ is not None:
            return self.model_.feature_importances_
        return None
