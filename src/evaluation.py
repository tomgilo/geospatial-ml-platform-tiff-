"""Evaluation metrics and model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics. Handles NaN values.

    Returns: dict with r2, rmse, mae, bias, n_valid.
    With a single test sample, R² is undefined (NaN) but RMSE/MAE/bias are still computed.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    n = len(yt)

    if n < 1:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "bias": np.nan, "n_valid": 0}

    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    bias = np.mean(yp - yt)

    if n < 2:
        # R² requires at least 2 samples for variance computation
        return {"r2": np.nan, "rmse": np.sqrt(mse), "mae": mae,
                "bias": bias, "n_valid": n}

    return {
        "r2": r2_score(yt, yp),
        "rmse": np.sqrt(mse),
        "mae": mae,
        "bias": bias,
        "n_valid": n,
    }


def evaluate_pixel_models(pixel_models, y_test_2d, X_test_3d):
    """
    Evaluate all trained models on all pixels.

    Args:
      pixel_models: dict of model_name -> list of per-pixel models
      y_test_2d: (n_pixels, n_test_time) true values
      X_test_3d: (n_pixels, n_test_time, n_features) test features

    Returns:
      dict of model_name -> list of per-pixel metric dicts
    """
    n_pixels = y_test_2d.shape[0]
    all_metrics = {}

    for model_name, models_list in pixel_models.items():
        pixel_metrics = []
        for i in range(n_pixels):
            model = models_list[i]
            if model is None:
                pixel_metrics.append({"r2": np.nan, "rmse": np.nan, "mae": np.nan,
                                      "bias": np.nan, "n_valid": 0})
                continue
            try:
                pred = model.predict(X_test_3d[i])  # (n_test_time,)
                m = compute_metrics(y_test_2d[i], pred)
            except Exception:
                m = {"r2": np.nan, "rmse": np.nan, "mae": np.nan,
                     "bias": np.nan, "n_valid": 0}
            pixel_metrics.append(m)
        all_metrics[model_name] = pixel_metrics

    return all_metrics


def aggregate_metrics(all_metrics):
    """
    Aggregate per-pixel metrics to summary statistics.

    Returns: pd.DataFrame with columns: model, mean_r2, median_r2, mean_rmse,
             median_rmse, mean_mae, mean_bias, success_rate

    success_rate counts pixels where RMSE is valid (works with ≥1 test sample).
    R² may be NaN when predicting only 1 year (needs ≥2 samples).
    """
    rows = []
    for model_name, pixel_metrics in all_metrics.items():
        r2s = [m["r2"] for m in pixel_metrics if not np.isnan(m["r2"])]
        rmses = [m["rmse"] for m in pixel_metrics if not np.isnan(m["rmse"])]
        maes = [m["mae"] for m in pixel_metrics if not np.isnan(m["mae"])]
        biases = [m["bias"] for m in pixel_metrics if not np.isnan(m["bias"])]
        n_total = len(pixel_metrics)
        # Use RMSE-based success count (works with 1+ test samples)
        n_success = len(rmses)

        rows.append({
            "model": model_name,
            "mean_r2": np.mean(r2s) if r2s else np.nan,
            "median_r2": np.median(r2s) if r2s else np.nan,
            "mean_rmse": np.mean(rmses) if rmses else np.nan,
            "median_rmse": np.median(rmses) if rmses else np.nan,
            "mean_mae": np.mean(maes) if maes else np.nan,
            "mean_bias": np.mean(biases) if biases else np.nan,
            "success_rate": n_success / n_total if n_total > 0 else 0,
            "n_pixels": n_success,
        })

    return pd.DataFrame(rows)


def compute_feature_importance(pixel_models, model_name, n_features):
    """Extract feature importance across all pixels for tree-based models."""
    models_list = pixel_models.get(model_name, [])
    if not models_list:
        return None

    importances = []
    for m in models_list:
        if m is not None and hasattr(m, "get_feature_importance"):
            fi = m.get_feature_importance()
            if fi is not None:
                importances.append(fi)

    if not importances:
        return None

    importances = np.array(importances)
    return {
        "mean": np.mean(importances, axis=0),
        "std": np.std(importances, axis=0),
        "n_models": len(importances),
    }


def evaluate_all(pixel_models, y_test_2d, X_test_3d):
    """
    Full evaluation pipeline.

    Returns: (summary_df, per_pixel_metrics, feature_importances)
    """
    all_metrics = evaluate_pixel_models(pixel_models, y_test_2d, X_test_3d)
    summary_df = aggregate_metrics(all_metrics)

    fi_results = {}
    n_features = X_test_3d.shape[2]
    for model_name in pixel_models:
        fi = compute_feature_importance(pixel_models, model_name, n_features)
        if fi is not None:
            fi_results[model_name] = fi

    return summary_df, all_metrics, fi_results
