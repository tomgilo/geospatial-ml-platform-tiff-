"""Generate prediction TIFF files from trained pixel models."""

import os
import numpy as np
import rasterio


def predict_pixel(models_list, X_predict):
    """
    Apply fitted models to predict each pixel.

    Args:
      models_list: list of fitted model objects (one per pixel, may be None)
      X_predict: (n_pixels, n_predict_time, n_features) prediction features

    Returns: (n_pixels, n_predict_time) predictions, NaN where model is None
    """
    n_pixels = len(models_list)
    if n_pixels == 0:
        return np.array([])

    n_predict = X_predict.shape[1]
    predictions = np.full((n_pixels, n_predict), np.nan, dtype=np.float32)

    for i, model in enumerate(models_list):
        if model is None:
            continue
        try:
            pred = model.predict(X_predict[i])  # (n_predict_time,)
            predictions[i] = pred
        except Exception:
            pass

    return predictions


def predictions_to_raster(predictions_1d, valid_indices, ref_profile, output_path):
    """
    Reconstruct a 2D raster from flat pixel predictions and save as TIFF.

    Args:
      predictions_1d: (n_valid_pixels,) array for a single prediction band
      valid_indices: flat indices of valid pixels in the full grid
      ref_profile: rasterio profile from reference raster
      output_path: path to save the output TIFF
    """
    n_rows = ref_profile["height"]
    n_cols = ref_profile["width"]

    out_profile = ref_profile.copy()
    out_profile.update({
        "dtype": "float32",
        "count": 1,
        "nodata": np.nan,
        "compress": "lzw",
    })

    full_array = np.full(n_rows * n_cols, np.nan, dtype=np.float32)
    full_array[valid_indices] = predictions_1d
    out_data = full_array.reshape(n_rows, n_cols)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(out_data, 1)

    return out_data


def generate_predictions(pixel_models, X_predict_3d, valid_indices,
                         ref_profile, output_dir, predict_labels,
                         folder_name="predictions"):
    """
    Generate prediction TIFFs for all models and time steps.

    Args:
      pixel_models: dict of model_name -> list of per-pixel models
      X_predict_3d: (n_valid_pixels, n_predict_time, n_x_vars) features
      valid_indices: flat indices of valid pixels
      ref_profile: reference rasterio profile
      output_dir: base output directory
      predict_labels: list of time labels for prediction periods
      folder_name: subfolder name for output TIFFs

    Returns:
      dict: model_name -> list of output file paths
    """
    out_folder = os.path.join(output_dir, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    output_files = {}

    for model_name, models_list in pixel_models.items():
        output_files[model_name] = []
        pred_2d = predict_pixel(models_list, X_predict_3d)  # (n_pixels, n_predict_time)

        for b, label in enumerate(predict_labels):
            fname = f"{model_name}_{label}.tif"
            fpath = os.path.join(out_folder, fname)
            predictions_to_raster(
                pred_2d[:, b], valid_indices, ref_profile, fpath
            )
            output_files[model_name].append(fpath)

    return output_files
