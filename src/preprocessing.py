"""Preprocessing: scaling, train/test splitting, pixel extraction."""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_data(train_data, test_data, method="standard"):
    """
    Fit scaler on training pixels and transform both train and test.

    Args:
      train_data: (n_pixels, n_time, n_features) — training features
      test_data: (n_pixels, n_time, n_features) — test features
      method: "standard", "minmax", or "none"

    Returns: (scaled_train, scaled_test, scaler)
    """
    if method == "none":
        return train_data, test_data, None

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    n_pixels, n_time, n_feat = train_data.shape
    train_flat = train_data.reshape(-1, n_feat)  # (n_pixels * n_time, n_feat)
    valid_rows = ~np.isnan(train_flat).any(axis=1)

    if valid_rows.sum() == 0:
        return train_data, test_data, scaler

    scaler.fit(train_flat[valid_rows])

    # Transform preserves NaN positions
    train_scaled = np.full_like(train_data, np.nan)
    test_scaled = np.full_like(test_data, np.nan)

    train_flat_scaled = np.full_like(train_flat, np.nan)
    train_flat_scaled[valid_rows] = scaler.transform(train_flat[valid_rows])
    train_scaled = train_flat_scaled.reshape(n_pixels, n_time, n_feat)

    test_n_pixels, test_n_time, _ = test_data.shape
    test_flat = test_data.reshape(-1, n_feat)
    test_valid = ~np.isnan(test_flat).any(axis=1)
    if test_valid.sum() > 0:
        test_flat_scaled = np.full_like(test_flat, np.nan)
        test_flat_scaled[test_valid] = scaler.transform(test_flat[test_valid])
        test_scaled = test_flat_scaled.reshape(test_n_pixels, test_n_time, n_feat)

    return train_scaled, test_scaled, scaler


def extract_pixel_series(cube, time_indices):
    """
    Extract pixel time series from a 3D cube for given time indices.

    Args:
      cube: (time, rows, cols) array
      time_indices: list/array of indices into the time dimension

    Returns: (n_pixels, len(time_indices)) array
    """
    selected = cube[time_indices]  # (n_time, rows, cols)
    n_time, n_rows, n_cols = selected.shape
    return selected.reshape(n_time, -1).T  # (n_pixels, n_time)


def build_design_matrix(y_cube, x_cubes, y_train_indices, y_predict_indices,
                        x_train_indices, x_predict_indices, mask):
    """
    Build training and prediction design matrices.

    For each valid pixel:
      - y_train: y values at train time indices → (n_pixels, n_train_time)
      - X_train: x values at train time indices → (n_pixels, n_train_time, n_x_vars)
      - y_test: y values at predict time indices → (n_pixels, n_test_time)
      - X_test: x values at predict time indices → (n_pixels, n_test_time, n_x_vars)

    Each time step is a separate sample. Features are the different X variables
    at that same time step.

    Returns: (y_train_2d, X_train_3d, y_test_2d, X_test_3d, valid_pixel_indices)
    """
    n_rows, n_cols = y_cube.shape[1], y_cube.shape[2]
    pixel_count = mask.sum()
    valid_indices = np.where(mask.ravel())[0]

    n_y_train = len(y_train_indices)
    n_y_test = len(y_predict_indices)
    n_x_vars = len(x_cubes)

    y_train_2d = np.full((pixel_count, n_y_train), np.nan, dtype=np.float32)
    y_test_2d = np.full((pixel_count, n_y_test), np.nan, dtype=np.float32)
    X_train_3d = np.full((pixel_count, n_y_train, n_x_vars), np.nan, dtype=np.float32)
    X_test_3d = np.full((pixel_count, n_y_test, n_x_vars), np.nan, dtype=np.float32)

    if pixel_count == 0:
        return y_train_2d, X_train_3d, y_test_2d, X_test_3d, valid_indices

    # Pre-extract all pixel series for Y cube
    y_all_pixels = y_cube.reshape(y_cube.shape[0], -1).T  # (n_pixels, n_time)

    # Fill Y
    for i, flat_idx in enumerate(valid_indices):
        y_train_2d[i] = y_all_pixels[flat_idx, y_train_indices]
        y_test_2d[i] = y_all_pixels[flat_idx, y_predict_indices]

    # Fill X — each X variable is a column, each time step is a row
    for v, x_cube in enumerate(x_cubes):
        x_all_pixels = x_cube.reshape(x_cube.shape[0], -1).T  # (n_pixels, n_time)
        for i, flat_idx in enumerate(valid_indices):
            X_train_3d[i, :, v] = x_all_pixels[flat_idx, x_train_indices]
            X_test_3d[i, :, v] = x_all_pixels[flat_idx, x_predict_indices]

    return y_train_2d, X_train_3d, y_test_2d, X_test_3d, valid_indices
