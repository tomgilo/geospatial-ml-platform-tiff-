"""Model persistence: save and load trained pixel models."""

import os
import time
import json
import joblib
import numpy as np


def save_model_package(pixel_models, config, y_labels, y_predict_labels,
                       valid_indices, y_profile, output_dir, x_names=None):
    """
    Save all trained models and metadata for later reuse.

    Directory structure:
      output_dir/models/
          pixel_data.joblib          # All pixel models (dict of model_name -> list)
          metadata.json              # Config, labels, profile info
    """
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Save models
    model_path = os.path.join(models_dir, "pixel_data.joblib")
    # Extract only the model objects (not the full pipeline context)
    model_package = {}
    for model_name, models_list in pixel_models.items():
        # Count non-None models
        n_trained = sum(1 for m in models_list if m is not None)
        model_package[model_name] = {
            "models": models_list,
            "n_trained": n_trained,
            "n_total": len(models_list),
        }
    joblib.dump(model_package, model_path, compress=3)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # 2. Save metadata
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_period": f"{config.train_start}-{config.train_end}",
        "predict_period": f"{config.predict_start}-{config.predict_end}",
        "time_unit": config.time_unit,
        "prediction_mode": config.prediction_mode,
        "scaling": config.scaling,
        "n_jobs": config.n_jobs,
        "y_folder": config.y_folder,
        "x_folders": config.x_folders,
        "y_labels": y_labels,
        "y_predict_labels": y_predict_labels,
        "y_profile": {
            "height": y_profile.get("height"),
            "width": y_profile.get("width"),
            "crs": str(y_profile.get("crs", "")),
            "transform": str(y_profile.get("transform", "")),
        },
        "valid_indices": valid_indices.tolist() if isinstance(valid_indices, np.ndarray)
                         else list(valid_indices),
        "x_names": x_names,
        "model_info": {},
    }
    for mn, mp in model_package.items():
        meta["model_info"][mn] = {
            "n_trained": mp["n_trained"],
            "n_total": mp["n_total"],
        }

    meta_path = os.path.join(models_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return model_path, meta


def load_model_package(output_dir):
    """Load a previously saved model package."""
    models_dir = os.path.join(output_dir, "models")
    model_path = os.path.join(models_dir, "pixel_data.joblib")
    meta_path = os.path.join(models_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    package = joblib.load(model_path)
    # Reconstruct pixel_models dict
    pixel_models = {}
    for model_name, mp in package.items():
        pixel_models[model_name] = mp["models"]

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return pixel_models, meta


def list_saved_models(base_dir=None):
    """List all saved model directories with their metadata."""
    if base_dir is None:
        base_dir = os.getcwd()
    results = []
    for root, dirs, _ in os.walk(base_dir):
        if "models" in dirs:
            models_dir = os.path.join(root, "models")
            meta_path = os.path.join(models_dir, "metadata.json")
            model_path = os.path.join(models_dir, "pixel_data.joblib")
            if os.path.exists(meta_path) and os.path.exists(model_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                results.append({
                    "dir": root,
                    "saved_at": meta.get("saved_at", "unknown"),
                    "train": meta.get("train_period", "?"),
                    "predict": meta.get("predict_period", "?"),
                    "models": list(meta.get("model_info", {}).keys()),
                    "size_mb": os.path.getsize(model_path) / (1024 * 1024),
                })
    return results
