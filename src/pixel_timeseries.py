"""Pixel-level time series extraction and visualization."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def extract_pixel_timeseries(y_cube, x_cubes, valid_indices, y_labels, x_names,
                             pixel_row, pixel_col):
    """
    Extract time series for a specific pixel (row, col).

    Returns:
      dict with keys: row, col, flat_idx, y_series (values), y_labels,
                      x_series (dict of name -> values), x_labels
    """
    n_rows, n_cols = y_cube.shape[1], y_cube.shape[2]
    if pixel_row < 0 or pixel_row >= n_rows or pixel_col < 0 or pixel_col >= n_cols:
        return None

    flat_idx = pixel_row * n_cols + pixel_col

    # Extract Y time series
    y_series = y_cube[:, pixel_row, pixel_col]  # (n_time,)

    # Extract X time series for each variable
    x_series = {}
    x_labels = []
    for i, x_cube in enumerate(x_cubes):
        name = x_names[i] if x_names and i < len(x_names) else f"X{i+1}"
        x_series[name] = x_cube[:, pixel_row, pixel_col]
        x_labels = y_labels  # all data shares same time labels

    return {
        "row": pixel_row,
        "col": pixel_col,
        "flat_idx": flat_idx,
        "y_series": y_series,
        "y_labels": y_labels,
        "x_series": x_series,
        "x_labels": x_labels,
    }


def plot_pixel_timeseries(pixel_data, model_predictions=None, output_path=None):
    """
    Plot time series for a single pixel: observed Y + model predictions.

    Args:
      pixel_data: dict from extract_pixel_timeseries
      model_predictions: dict of model_name -> (truth_series, pred_series)
                         for the train/predict split
      output_path: save path
    """
    y_labels = pixel_data["y_labels"]
    y_series = pixel_data["y_series"]

    n_vars = len(pixel_data["x_series"]) + 1  # +1 for Y
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)

    # Y plot (with predictions if available)
    ax_y = axes[0] if n_vars > 1 else axes
    x_idx = np.arange(len(y_labels))

    # Plot observed
    ax_y.plot(x_idx, y_series, "o-", color="#2196F3", linewidth=1.5, markersize=5,
              label="观测值", zorder=5)

    # Plot predictions if given
    if model_predictions:
        for model_name, (truth, pred) in model_predictions.items():
            # prediction indices (after training period)
            n_predict = len(pred)
            pred_idx = np.arange(len(y_labels) - n_predict, len(y_labels))
            # Plot truth for predict period
            ax_y.plot(pred_idx, truth, "s-", color="#4CAF50", linewidth=1.5,
                      markersize=5, label=f"真实值(预测期)", alpha=0.7)
            ax_y.plot(pred_idx, pred, "s--", color="#F44336", linewidth=1.5,
                      markersize=5, label=f"{model_name} 预测", alpha=0.7)

    # Train/predict divider
    if model_predictions:
        n_predict = len(list(model_predictions.values())[0][1])
        divider_x = len(y_labels) - n_predict - 0.5
        ax_y.axvline(divider_x, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_y.text(divider_x - 0.5, ax_y.get_ylim()[1] * 0.95, "训练期",
                  ha="right", fontsize=8, color="gray")
        ax_y.text(divider_x + 0.5, ax_y.get_ylim()[1] * 0.95, "预测期",
                  ha="left", fontsize=8, color="gray")

    ax_y.set_ylabel("Y 值", fontsize=10)
    ax_y.set_title(f"像素 [{pixel_data['row']},{pixel_data['col']}] 时间序列",
                   fontsize=12, fontweight="bold")
    ax_y.legend(fontsize=8, loc="best")
    ax_y.grid(True, alpha=0.3)
    ax_y.set_xticks(x_idx)
    ax_y.set_xticklabels(y_labels, rotation=45, fontsize=7)

    # X variable plots
    for i, (name, x_vals) in enumerate(pixel_data["x_series"].items()):
        ax = axes[i + 1] if n_vars > 1 else axes
        ax.plot(x_idx, x_vals, "s-", color="#FF9800", linewidth=1.2, markersize=4)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(y_labels, rotation=45, fontsize=7)

    fig.tight_layout(pad=1.5)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path
    else:
        return fig


def plot_multi_pixel_summary(all_pixel_data, output_path, n_samples=9):
    """
    Plot a grid of randomly sampled pixels' Y time series.
    """
    import random
    random.seed(42)
    n_pixels = len(all_pixel_data)
    sample_n = min(n_pixels, n_samples)
    sample_indices = random.sample(range(n_pixels), sample_n)

    ncols = 3
    nrows = (sample_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        pd = all_pixel_data[idx]
        labels = pd["y_labels"]
        y_vals = pd["y_series"]
        x_idx = np.arange(len(labels))

        valid_mask = ~np.isnan(y_vals)
        ax.plot(np.array(x_idx)[valid_mask], y_vals[valid_mask],
                "o-", color="#2196F3", linewidth=1, markersize=3)
        ax.set_title(f"像素 [{pd['row']},{pd['col']}]", fontsize=8)
        ax.set_xticks(x_idx[::max(1, len(x_idx)//5)])
        ax.set_xticklabels([labels[j] for j in x_idx[::max(1, len(x_idx)//5)]],
                           rotation=30, fontsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(sample_n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("随机采样像素 Y 值时间序列", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_pixel_timeseries_data(y_cube, x_cubes, valid_indices, y_labels, x_names):
    """
    Build a list of pixel time series dicts for all valid pixels.
    """
    n_rows, n_cols = y_cube.shape[1], y_cube.shape[2]
    all_data = []
    for flat_idx in valid_indices:
        row = flat_idx // n_cols
        col = flat_idx % n_cols
        y_series = y_cube[:, row, col]
        x_series = {}
        for i, x_cube in enumerate(x_cubes):
            name = x_names[i] if x_names and i < len(x_names) else f"X{i+1}"
            x_series[name] = x_cube[:, row, col]
        all_data.append({
            "row": row, "col": col, "flat_idx": flat_idx,
            "y_series": y_series, "y_labels": y_labels,
            "x_series": x_series,
        })
    return all_data
