"""Main pipeline orchestrator — ties data loading, training, and evaluation together."""

import os
import time
import json
import numpy as np
import pandas as pd
from .config import AppConfig
from .data_loader import discover_tiff_files, load_raster_stack, validate_alignment, get_valid_mask
from .preprocessing import build_design_matrix, scale_data
from .trainer import train_all_pixels
from .predictor import generate_predictions
from .evaluation import evaluate_all
from .visualization import generate_all_charts
from .report import generate_html_report


class Pipeline:
    """End-to-end geospatial ML pipeline."""

    def __init__(self, config: AppConfig, progress_callback=None):
        self.config = config
        self.progress = progress_callback or (lambda msg, pct: None)
        self.results = {}
        self.warnings = []

    def log(self, msg, pct=0):
        self.progress(msg, pct)

    def run(self):
        t0 = time.perf_counter()

        # Step 1: Discover files
        self.log("Scanning Y folder...", 5)
        y_files, y_unit = discover_tiff_files(self.config.y_folder)
        if not y_files:
            raise ValueError("No TIFF files found in Y folder.")
        self.config.time_unit = y_unit

        x_file_dicts = []
        x_names = []
        for i, xf in enumerate(self.config.x_folders):
            self.log(f"Scanning X folder {i+1}...", 10 + i * 5)
            xf_dict, xu = discover_tiff_files(xf)
            if not xf_dict:
                self.warnings.append(f"X folder {i+1} is empty, skipping.")
                continue
            x_file_dicts.append(xf_dict)
            x_names.append(os.path.basename(xf.rstrip("/\\")))

        if not x_file_dicts:
            raise ValueError("No valid X folders found.")

        # Step 2: Load Y cube (full time range)
        self.log("Loading Y rasters...", 20)
        y_labels, y_cube, y_profile = load_raster_stack(y_files)
        n_rows, n_cols = y_cube.shape[1], y_cube.shape[2]

        # Step 3: Load X cubes (full time range)
        self.log("Loading X rasters...", 30)
        x_cubes = []
        x_labels_list = []
        x_profiles = []
        for xf_dict in x_file_dicts:
            xl, xc, xp = load_raster_stack(xf_dict, expected_shape=(n_rows, n_cols))
            x_cubes.append(xc)
            x_labels_list.append(xl)
            x_profiles.append(xp)

        # Step 4: Validate alignment
        self.log("Validating alignment...", 40)
        align_warnings = validate_alignment(y_profile, x_profiles)
        self.warnings.extend(align_warnings)

        # Step 5: Compute valid pixel mask
        self.log("Computing valid pixel mask...", 45)
        mask = get_valid_mask(y_cube, *x_cubes, min_valid_ratio=self.config.valid_pct_threshold)
        n_valid = mask.sum()
        n_total = n_rows * n_cols
        self.log(f"Valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)", 48)

        if n_valid == 0:
            raise ValueError("No valid pixels found. Check your data for NaN/nodata values.")

        # Step 6: Map train/predict year ranges to indices in the full cubes
        y_train_indices, y_train_labels = self._year_to_indices(
            y_labels, self.config.train_start, self.config.train_end
        )
        y_predict_indices, y_predict_labels = self._year_to_indices(
            y_labels, self.config.predict_start, self.config.predict_end
        )

        if not y_train_indices:
            raise ValueError(f"No Y files found in training range {self.config.train_start}-{self.config.train_end}")
        if not y_predict_indices:
            raise ValueError(f"No Y files found in prediction range {self.config.predict_start}-{self.config.predict_end}")

        # For X, map to same time indices (all data shares same time coverage)
        x_train_indices = y_train_indices

        # Simulation mode: use future X data (default)
        # Extrapolation mode: no future X — use last training year's X repeated
        if self.config.prediction_mode == "extrapolation":
            x_predict_indices = [y_train_indices[-1]] * len(y_predict_indices)
            self.warnings.append(
                f"Extrapolation mode: using X from {y_labels[y_train_indices[-1]]} "
                f"to predict {self.config.predict_start}-{self.config.predict_end}"
            )
        else:
            x_predict_indices = y_predict_indices

        self.log(f"Train on {len(y_train_indices)} time steps, predict on {len(y_predict_indices)} time steps", 50)

        # Step 7: Build design matrices
        # X shape: (n_pixels, n_time, n_x_vars) — each time step is a sample
        # y shape: (n_pixels, n_time)
        self.log("Building design matrices...", 55)
        y_train_2d, X_train_3d, y_test_2d, X_test_3d, valid_indices = build_design_matrix(
            y_cube, x_cubes,
            y_train_indices, y_predict_indices,
            x_train_indices, x_predict_indices,
            mask
        )

        # Step 8: Scale data (operates on (n_pixels, n_time, n_features) 3D arrays)
        self.log(f"Scaling data ({self.config.scaling})...", 60)
        X_train_scaled, X_test_scaled, scaler = scale_data(
            X_train_3d, X_test_3d, method=self.config.scaling
        )

        # Step 9: Train models (each pixel independently, with time steps as samples)
        self.log(f"Training models on {self.config.n_jobs} CPU cores...", 65)
        enabled_models = self.config.get_enabled_models()

        def training_progress(model_name, success, total, elapsed):
            self.log(f"  {model_name}: {success}/{total} pixels trained ({elapsed:.1f}s)", 0)

        pixel_models = train_all_pixels(
            y_train_2d, X_train_scaled, enabled_models,
            n_jobs=self.config.n_jobs,
            progress_callback=training_progress
        )

        # Step 10: Generate prediction TIFFs
        self.log("Generating prediction TIFFs...", 75)
        predict_outputs = generate_predictions(
            pixel_models, X_test_scaled, valid_indices,
            y_profile, self.config.output_dir, y_predict_labels,
            folder_name="predictions"
        )

        # Step 11: Evaluate
        self.log("Evaluating models...", 85)
        summary_df, all_metrics, fi_results = evaluate_all(
            pixel_models, y_test_2d, X_test_scaled
        )

        # Diagnostic: check if every model failed
        if summary_df is not None and not summary_df.empty:
            total_success = summary_df["success_rate"].sum()
            if total_success == 0:
                self.warnings.append(
                    "All models failed for all pixels. "
                    "Common causes: (1) training time range does not match data filenames, "
                    "(2) too many NaN/nodata values in training period, "
                    "(3) not enough valid samples per pixel (need >= 2). "
                    "Check your data and time configuration."
                )
            elif "mean_r2" in summary_df.columns and summary_df["mean_r2"].isna().all():
                # Models trained successfully (RMSE available) but R² is NaN
                # This happens when predicting only 1 year — R² needs ≥2 samples
                pass  # Not a failure, just informational

        # Step 12: Generate charts
        self.log("Generating charts...", 90)
        chart_dir = os.path.join(self.config.output_dir, "charts")
        chart_paths = generate_all_charts(
            pixel_models, y_test_2d, X_test_scaled, all_metrics,
            valid_indices, y_profile, chart_dir, x_names=x_names,
            summary_df=summary_df
        )

        # Step 13: Save results tables
        self.log("Saving results...", 95)
        tables_dir = os.path.join(self.config.output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        # Metrics summary
        csv_path = os.path.join(tables_dir, "metrics_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        # Per-pixel metrics for each model
        for model_name, metrics_list in all_metrics.items():
            pm_path = os.path.join(tables_dir, f"per_pixel_metrics_{model_name}.csv")
            pd.DataFrame(metrics_list).to_csv(pm_path, index=False)

        # Feature importance CSV
        if fi_results:
            fi_export = []
            for mn, fi in fi_results.items():
                for i, (mean_v, std_v) in enumerate(zip(fi["mean"], fi["std"])):
                    feat_name = x_names[i] if x_names and i < len(x_names) else f"X{i+1}"
                    fi_export.append({"model": mn, "feature": feat_name,
                                       "importance_mean": mean_v, "importance_std": std_v})
            if fi_export:
                pd.DataFrame(fi_export).to_csv(
                    os.path.join(tables_dir, "feature_importance.csv"), index=False)

        # Config export
        config_path = os.path.join(self.config.output_dir, "config.json")
        config_export = {
            "train_period": f"{self.config.train_start}-{self.config.train_end}",
            "predict_period": f"{self.config.predict_start}-{self.config.predict_end}",
            "time_unit": self.config.time_unit,
            "prediction_mode": self.config.prediction_mode,
            "scaling": self.config.scaling,
            "n_jobs": self.config.n_jobs,
            "n_valid_pixels": int(n_valid),
            "models": [{"name": m.name, "params": m.params} for m in enabled_models],
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_export, f, indent=2, ensure_ascii=False)

        # Step 14: Generate HTML report
        total_elapsed = time.perf_counter() - t0
        reports_dir = os.path.join(self.config.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "evaluation_report.html")
        config_info = {
            "train_period": f"{self.config.train_start}-{self.config.train_end}",
            "predict_period": f"{self.config.predict_start}-{self.config.predict_end}",
            "time_unit": self.config.time_unit,
            "scaling": self.config.scaling,
            "n_jobs": self.config.n_jobs,
            "y_folder": self.config.y_folder,
            "x_folders": ", ".join(self.config.x_folders),
        }
        training_info = {
            "n_valid_pixels": int(n_valid),
            "n_total_pixels": n_total,
            "n_train_time_steps": len(y_train_indices),
            "n_predict_time_steps": len(y_predict_indices),
            "n_x_variables": len(x_cubes),
            "model_params": ", ".join(
                f"{m.name}: {m.params}" for m in enabled_models
            ),
        }
        generate_html_report(
            summary_df, all_metrics, config_info, training_info,
            predict_outputs, chart_paths, self.warnings, report_path,
            total_time=total_elapsed
        )

        # Step 15: Save models for reuse
        self.log("Saving models...", 97)
        from .model_io import save_model_package
        model_path, model_meta = save_model_package(
            pixel_models, self.config, y_labels, y_predict_labels,
            valid_indices, y_profile, self.config.output_dir,
            x_names=x_names
        )

        # Store raw data for optional later analysis (SHAP / pixel timeseries)
        # These are NOT run automatically during training — user triggers them separately
        self.log("Done!", 100)

        self.results = {
            "summary_df": summary_df,
            "all_metrics": all_metrics,
            "fi_results": fi_results,
            "pixel_models": pixel_models,
            "predict_outputs": predict_outputs,
            "chart_paths": chart_paths,
            "report_path": report_path,
            "csv_path": csv_path,
            "n_valid_pixels": int(n_valid),
            "y_labels": y_labels,
            "y_predict_labels": y_predict_labels,
            "warnings": self.warnings,
            "total_time": total_elapsed,
            "valid_indices": valid_indices,
            "y_profile": y_profile,
            "X_test_3d": X_test_scaled,
            "y_test_2d": y_test_2d,
            "x_names": x_names,
            "y_cube": y_cube,
            "x_cubes": x_cubes,
            "model_path": model_path,
            "enabled_models": [m.name for m in enabled_models],
            "mask": mask,
            "y_train_indices": y_train_indices,
            "y_predict_indices": y_predict_indices,
        }
        return self.results

    def _year_to_indices(self, labels, start_year, end_year):
        """Map year range to indices in the sorted labels list."""
        indices = []
        matched_labels = []
        for i, label in enumerate(labels):
            year = int(label.split("-")[0]) if "-" in label else int(label)
            if start_year <= year <= end_year:
                indices.append(i)
                matched_labels.append(label)
        return indices, matched_labels
