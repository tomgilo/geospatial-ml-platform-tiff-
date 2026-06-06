"""Configuration dataclass for the geospatial ML application."""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Configuration for a single ML model."""
    name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "enabled": self.enabled, "params": self.params}


@dataclass
class AppConfig:
    """Master configuration for the application."""
    # Data paths
    y_folder: str = ""
    x_folders: list = field(default_factory=list)
    output_dir: str = ""

    # Time configuration
    train_start: int = 2000
    train_end: int = 2010
    predict_start: int = 2011
    predict_end: int = 2011
    time_unit: str = "yearly"  # "yearly" or "monthly"

    # Preprocessing
    scaling: str = "standard"  # "none", "standard", "minmax"
    valid_pct_threshold: float = 0.5  # min fraction of valid pixels required
    prediction_mode: str = "simulation"  # "simulation" (needs future X) or "extrapolation" (no future X)

    # Model configurations — all disabled by default, user must manually select
    models: list = field(default_factory=lambda: [
        ModelConfig(name="ols", enabled=False),
        ModelConfig(name="ridge", enabled=False, params={"alpha": 1.0}),
        ModelConfig(name="lasso", enabled=False, params={"alpha": 0.01}),
        ModelConfig(name="elasticnet", enabled=False, params={"alpha": 0.01, "l1_ratio": 0.5}),
        ModelConfig(name="rf", enabled=False, params={"n_estimators": 100, "max_depth": 10,
                        "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}),
        ModelConfig(name="xgboost", enabled=False, params={"n_estimators": 100, "max_depth": 6,
                        "learning_rate": 0.1, "subsample": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0}),
        ModelConfig(name="lightgbm", enabled=False, params={"n_estimators": 100, "max_depth": 6,
                        "learning_rate": 0.1, "num_leaves": 31, "subsample": 1.0,
                        "reg_lambda": 0.0, "reg_alpha": 0.0, "min_child_samples": 20}),
        ModelConfig(name="extratrees", enabled=False, params={"n_estimators": 100, "max_depth": 10,
                        "min_samples_split": 2, "max_features": "sqrt"}),
        ModelConfig(name="gbr", enabled=False, params={"n_estimators": 100, "max_depth": 3,
                        "learning_rate": 0.1, "subsample": 1.0, "min_samples_split": 2,
                        "min_samples_leaf": 1}),
        ModelConfig(name="svr", enabled=False, params={"kernel": "rbf", "C": 1.0,
                        "epsilon": 0.1, "gamma": "scale"}),
        ModelConfig(name="mlp", enabled=False, params={"hidden_layer_sizes": [64, 32],
                        "max_iter": 500, "activation": "relu", "alpha": 0.0001}),
        ModelConfig(name="knn", enabled=False, params={"n_neighbors": 5, "weights": "uniform"}),
    ])

    # Training
    n_jobs: int = -1  # -1 = all CPU cores
    cv_folds: int = 5  # cross-validation folds, 0 = no CV

    def validate(self) -> list:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.y_folder or not os.path.isdir(self.y_folder):
            errors.append(f"Y folder does not exist: {self.y_folder}")
        for i, xf in enumerate(self.x_folders):
            if not os.path.isdir(xf):
                errors.append(f"X folder {i+1} does not exist: {xf}")
        if self.train_start > self.train_end:
            errors.append("train_start must be <= train_end")
        if self.predict_start > self.predict_end:
            errors.append("predict_start must be <= predict_end")
        if not self.output_dir:
            self.output_dir = os.path.abspath("./outputs")
        return errors

    def get_enabled_models(self):
        return [m for m in self.models if m.enabled]
