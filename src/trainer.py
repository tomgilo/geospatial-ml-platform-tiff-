"""Training orchestrator with joblib parallelization over pixels."""

import time
import numpy as np
from joblib import Parallel, delayed
from .models import get_model


def train_single_pixel(X_train, y_train, model_name, model_params):
    """
    Train a single model on one pixel's time series data.

    Args:
      X_train: (n_samples, n_features) — each row is a time step
      y_train: (n_samples,) — target at each time step
      model_name: registered model name
      model_params: dict of hyperparameters

    Returns: (fitted_model_or_None, success_bool)
    """
    valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    if valid.sum() < 2:
        return None, False

    try:
        model = get_model(model_name, model_params)
        model.fit_timed(X_train[valid], y_train[valid])
        return model, True
    except Exception:
        return None, False


def train_all_pixels(y_train_2d, X_train_3d, model_configs, n_jobs=-1,
                     progress_callback=None):
    """
    Train all configured models for all pixels in parallel.

    Args:
      y_train_2d: (n_pixels, n_train_time) target values
      X_train_3d: (n_pixels, n_train_time, n_x_vars) feature values
      model_configs: list of ModelConfig objects
      n_jobs: CPU cores to use (-1 = all)
      progress_callback: optional callable(model_name, success, total, elapsed)

    Returns:
      dict: model_name -> list of fitted models (one per pixel, None for failures)
    """
    n_pixels = y_train_2d.shape[0]
    results = {}

    for mc in model_configs:
        if not mc.enabled:
            continue

        t0 = time.perf_counter()
        results[mc.name] = []

        trained = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(train_single_pixel)(
                X_train_3d[i], y_train_2d[i],
                mc.name, mc.params
            )
            for i in range(n_pixels)
        )

        pixel_models = []
        success_count = 0
        for model, ok in trained:
            pixel_models.append(model if ok else None)
            if ok:
                success_count += 1

        results[mc.name] = pixel_models
        elapsed = time.perf_counter() - t0

        if progress_callback:
            progress_callback(mc.name, success_count, n_pixels, elapsed)

    return results
