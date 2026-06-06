"""Data exploration utilities for geospatial TIFF files."""

import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_tiff_info(filepath):
    """Return comprehensive metadata for a TIFF file."""
    info = {"file": os.path.basename(filepath), "exists": os.path.exists(filepath)}
    if not info["exists"]:
        return info

    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            info.update({
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else "未定义",
                "transform": str(src.transform),
                "resolution": (src.transform.a, -src.transform.e),
                "bounds": str(src.bounds),
                "nodata": src.nodata,
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "median": float(np.nanmedian(data)),
                "valid_pct": float(np.sum(~np.isnan(data)) / data.size * 100),
                "nan_pct": float(np.sum(np.isnan(data)) / data.size * 100),
            })
    except Exception as e:
        info["error"] = str(e)
    return info


def plot_histogram(filepath, output_path, bins=50):
    """Plot histogram of raster values."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(valid), color="red", linestyle="--", label=f"Mean={np.mean(valid):.2f}")
    ax.axvline(np.median(valid), color="green", linestyle="--", label=f"Median={np.median(valid):.2f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram — {os.path.basename(filepath)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_raster_preview(filepath, output_path):
    """Plot a quick preview of the raster data."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = np.nanpercentile(data, [2, 98])
    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Value")
    ax.set_title(f"Preview — {os.path.basename(filepath)}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_correlation_matrix(file_dict1, file_dict2=None, sample_step=10):
    """
    Compute correlation between variables across time.
    file_dict1: dict of {label: filepath} for variable 1
    file_dict2: dict of {label: filepath} for variable 2 (optional, same as 1 if None)
    Returns: (correlations, labels1, labels2)
    """
    if file_dict2 is None:
        file_dict2 = file_dict1

    labels1 = sorted(file_dict1.keys())
    labels2 = sorted(file_dict2.keys())

    # Read first file to get shape
    with rasterio.open(file_dict1[labels1[0]]) as src:
        data = src.read(1)
    n_rows, n_cols = data.shape

    # Sample pixels to speed up
    sample_rows = np.arange(0, n_rows, sample_step)
    sample_cols = np.arange(0, n_cols, sample_step)

    series1 = []
    for label in labels1:
        with rasterio.open(file_dict1[label]) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        sampled = data[np.ix_(sample_rows, sample_cols)].ravel()
        series1.append(sampled)

    series2 = []
    for label in labels2:
        with rasterio.open(file_dict2[label]) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        sampled = data[np.ix_(sample_rows, sample_cols)].ravel()
        series2.append(sampled)

    # Compute correlation
    corr = np.full((len(labels1), len(labels2)), np.nan)
    for i, s1 in enumerate(series1):
        for j, s2 in enumerate(series2):
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if valid.sum() > 10:
                corr[i, j] = np.corrcoef(s1[valid], s2[valid])[0, 1]

    return corr, labels1, labels2


def plot_correlation_heatmap(corr_matrix, labels, output_path, title="Correlation Matrix"):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.5), max(5, len(labels)*0.4)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(corr_matrix[i, j]):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                              fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_time_series(file_dict, output_path, sample_count=100):
    """Plot time series of sampled pixels."""
    labels = sorted(file_dict.keys())
    if len(labels) < 2:
        return None

    # Read all data
    cubes = []
    for label in labels:
        with rasterio.open(file_dict[label]) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        cubes.append(data)

    cube = np.stack(cubes, axis=0)  # (time, rows, cols)
    n_time, n_rows, n_cols = cube.shape

    # Randomly sample valid pixels
    valid_mask = ~np.isnan(cube).any(axis=0)
    valid_indices = np.argwhere(valid_mask)
    if len(valid_indices) == 0:
        return None

    n_sample = min(sample_count, len(valid_indices))
    sample_idx = valid_indices[np.random.choice(len(valid_indices), n_sample, replace=False)]

    fig, ax = plt.subplots(figsize=(10, 5))
    years = [int(l.split("-")[0]) if "-" in l else int(l) for l in labels]
    for idx in sample_idx[:20]:  # plot up to 20 lines
        r, c = idx
        ts = cube[:, r, c]
        ax.plot(years, ts, alpha=0.3, color="steelblue", linewidth=0.8)

    # Plot mean trend
    mean_ts = np.nanmean(cube.reshape(n_time, -1), axis=1)
    ax.plot(years, mean_ts, color="red", linewidth=2, label="Mean")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Trend (Sampled Pixels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
