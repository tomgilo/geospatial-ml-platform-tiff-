"""Spatial analysis tools for geospatial ML residuals and predictions."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_morans_i(residuals_2d, valid_mask=None, n_permutations=99):
    """
    Compute Moran's I spatial autocorrelation statistic.

    Args:
        residuals_2d: 2D array of residuals (rows, cols), NaN where invalid
        valid_mask: optional boolean mask of valid pixels
        n_permutations: number of permutations for pseudo-p-value

    Returns:
        dict with 'I', 'E_I', 'z_score', 'p_value'
    """
    if valid_mask is None:
        valid_mask = ~np.isnan(residuals_2d)

    # Extract valid residuals
    residuals = residuals_2d[valid_mask]
    n = len(residuals)
    if n < 10:
        return {"I": np.nan, "E_I": -1/(n-1) if n > 1 else np.nan,
                "z_score": np.nan, "p_value": np.nan, "n": n}

    # Get coordinates of valid pixels
    rows, cols = np.where(valid_mask)

    # Build inverse distance weights (simple rook/queen adjacency)
    # Using 4-neighbor connectivity
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = abs(rows[i] - rows[j]) + abs(cols[i] - cols[j])
            if dist == 1:  # immediate neighbors
                W[i, j] = 1
                W[j, i] = 1

    W_sum = W.sum()
    if W_sum == 0:
        return {"I": np.nan, "E_I": np.nan, "z_score": np.nan, "p_value": np.nan, "n": n}

    # Standardize residuals
    z = residuals - residuals.mean()

    # Moran's I
    numerator = (z.reshape(1, -1) * W * z.reshape(-1, 1)).sum()
    denominator = (z ** 2).sum()
    I = (n / W_sum) * (numerator / denominator) if denominator > 0 else 0

    E_I = -1 / (n - 1)

    # Permutation test for p-value
    I_perm = []
    for _ in range(n_permutations):
        z_perm = np.random.permutation(z)
        num_perm = (z_perm.reshape(1, -1) * W * z_perm.reshape(-1, 1)).sum()
        I_perm.append((n / W_sum) * (num_perm / denominator) if denominator > 0 else 0)

    I_perm = np.array(I_perm)
    p_value = np.mean(np.abs(I_perm - E_I) >= np.abs(I - E_I))
    z_score = (I - E_I) / np.std(I_perm) if np.std(I_perm) > 0 else 0

    return {
        "I": float(I),
        "E_I": float(E_I),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n": n,
        "interpretation": "clustered" if I > E_I and p_value < 0.05 else (
            "dispersed" if I < E_I and p_value < 0.05 else "random"
        )
    }


def plot_spatial_autocorrelation(residuals_2d, valid_mask, output_path, title="Spatial Autocorrelation"):
    """Plot Moran's I scatter plot (z vs spatial lag of z)."""
    residuals = residuals_2d[valid_mask]
    rows, cols = np.where(valid_mask)
    n = len(residuals)

    # Compute spatial lag
    lag = np.zeros(n)
    z = residuals - residuals.mean()
    z_std = z / np.std(z) if np.std(z) > 0 else z

    for i in range(n):
        neighbors = []
        for j in range(n):
            if i != j and abs(rows[i] - rows[j]) + abs(cols[i] - cols[j]) == 1:
                neighbors.append(z_std[j])
        if neighbors:
            lag[i] = np.mean(neighbors)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_std, lag, alpha=0.3, s=10, c="steelblue", edgecolors="none")

    # Fit line
    valid_for_fit = ~(np.isnan(z_std) | np.isnan(lag))
    if valid_for_fit.sum() > 10:
        coeffs = np.polyfit(z_std[valid_for_fit], lag[valid_for_fit], 1)
        x_line = np.linspace(z_std.min(), z_std.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=2, label=f"Slope={coeffs[0]:.3f}")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Standardized Residual (z)")
    ax.set_ylabel("Spatial Lag of z")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_performance_map(all_metrics, model_name, metric_key, valid_indices, ref_profile):
    """
    Create a 2D performance map for a given metric.

    Returns:
        2D array with metric values, NaN where invalid
    """
    n_rows = ref_profile["height"]
    n_cols = ref_profile["width"]
    full = np.full(n_rows * n_cols, np.nan, dtype=np.float32)

    metrics_list = all_metrics.get(model_name, [])
    values = [m.get(metric_key, np.nan) for m in metrics_list]
    if len(values) > 0 and len(valid_indices) >= len(values):
        full[valid_indices[:len(values)]] = values

    return full.reshape(n_rows, n_cols)


def plot_performance_map(performance_2d, output_path, title="Performance Map", cmap="RdYlGn",
                         vmin=None, vmax=None):
    """Plot a 2D performance metric map."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if vmin is None:
        vmin = np.nanpercentile(performance_2d, 2)
    if vmax is None:
        vmax = np.nanpercentile(performance_2d, 98)

    im = ax.imshow(performance_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
