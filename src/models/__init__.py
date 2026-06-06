"""Model registry — maps model names to constructors."""

from .linear_models import OLSModel, RidgeModel, LassoModel, ElasticNetModel
from .tree_models import RandomForestModel, XGBoostModel, LightGBMModel
from .svr_model import SVRModel
from .mlp_model import MLPModel
from .knn_model import KNNModel
from .gbr_model import GBRModel
from .extratrees_model import ExtraTreesModel


MODEL_REGISTRY = {
    "ols": OLSModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elasticnet": ElasticNetModel,
    "rf": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "extratrees": ExtraTreesModel,
    "gbr": GBRModel,
    "svr": SVRModel,
    "mlp": MLPModel,
    "knn": KNNModel,
}


def get_model(name, params=None):
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](params)


def get_available_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())
