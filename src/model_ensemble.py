"""Model ensemble and stacking for geospatial predictions."""

import numpy as np


def ensemble_average(predictions_dict, weights=None):
    """
    Weighted average ensemble of multiple model predictions.

    Args:
        predictions_dict: dict of model_name -> (n_pixels, n_time) predictions
        weights: dict of model_name -> weight, or None for equal weights

    Returns:
        (n_pixels, n_time) ensemble predictions
    """
    models = list(predictions_dict.keys())
    if not models:
        return None

    first_pred = predictions_dict[models[0]]
    n_pixels, n_time = first_pred.shape

    if weights is None:
        weights = {m: 1.0 / len(models) for m in models}

    ensemble = np.zeros((n_pixels, n_time), dtype=np.float32)
    total_weight = 0

    for model_name, pred in predictions_dict.items():
        w = weights.get(model_name, 0)
        if pred.shape == (n_pixels, n_time):
            # Only include valid predictions
            valid = ~np.isnan(pred)
            ensemble[valid] += pred[valid] * w
            total_weight += w

    if total_weight > 0:
        ensemble /= total_weight

    return ensemble


def ensemble_best_model(predictions_dict, metrics_dict, metric="mean_r2", maximize=True):
    """
    Select the best single model based on metrics and return its predictions.

    Args:
        predictions_dict: dict of model_name -> predictions
        metrics_dict: dict of model_name -> metric value
        metric: metric name to use for selection
        maximize: True if higher is better

    Returns:
        (best_model_name, best_predictions)
    """
    if not metrics_dict:
        return None, None

    if maximize:
        best_model = max(metrics_dict, key=lambda k: metrics_dict.get(k, float('-inf')))
    else:
        best_model = min(metrics_dict, key=lambda k: metrics_dict.get(k, float('inf')))

    return best_model, predictions_dict.get(best_model)


def compute_ensemble_uncertainty(predictions_dict):
    """
    Compute prediction uncertainty as standard deviation across models.

    Args:
        predictions_dict: dict of model_name -> (n_pixels, n_time) predictions

    Returns:
        (n_pixels, n_time) standard deviation array
    """
    models = list(predictions_dict.keys())
    if len(models) < 2:
        return None

    first_pred = predictions_dict[models[0]]
    n_pixels, n_time = first_pred.shape

    # Stack all predictions
    stacked = np.full((len(models), n_pixels, n_time), np.nan, dtype=np.float32)
    for i, model_name in enumerate(models):
        pred = predictions_dict[model_name]
        if pred.shape == (n_pixels, n_time):
            stacked[i] = pred

    return np.nanstd(stacked, axis=0)


def generate_ensemble_predictions(pixel_models, X_test_3d, valid_indices, ref_profile,
                                  output_dir, predict_labels, weights=None):
    """Generate ensemble prediction TIFFs."""
    import os
    from .predictor import predict_pixel, predictions_to_raster

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Get predictions from all models
    predictions_dict = {}
    for model_name, models_list in pixel_models.items():
        pred_2d = predict_pixel(models_list, X_test_3d)
        predictions_dict[model_name] = pred_2d

    # Compute ensemble
    ensemble_pred = ensemble_average(predictions_dict, weights)
    if ensemble_pred is None:
        return []

    # Save ensemble predictions
    for b, label in enumerate(predict_labels):
        fname = f"ensemble_mean_{label}.tif"
        fpath = os.path.join(output_dir, fname)
        predictions_to_raster(ensemble_pred[:, b], valid_indices, ref_profile, fpath)
        output_files.append(fpath)

    # Compute and save uncertainty
    uncertainty = compute_ensemble_uncertainty(predictions_dict)
    if uncertainty is not None:
        for b, label in enumerate(predict_labels):
            fname = f"ensemble_std_{label}.tif"
            fpath = os.path.join(output_dir, fname)
            predictions_to_raster(uncertainty[:, b], valid_indices, ref_profile, fpath)
            output_files.append(fpath)

    return output_files
