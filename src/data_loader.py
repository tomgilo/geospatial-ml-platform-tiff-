"""TIFF data loading, discovery, and alignment."""

import os
import re
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


def discover_tiff_files(folder, pattern=None):
    """
    Scan a folder for TIFF files and extract time labels.

    Returns: dict of {time_label: filepath}, detected_unit
      - Yearly: "2000.tif" → "2000"
      - Monthly (flat): "2000-01.tif" → "2000-01"
      - Monthly (nested): year_dir/01.tif → "2000-01"
    """
    if not os.path.isdir(folder):
        return {}, "yearly"

    files = {}
    yearly_re = re.compile(r"^(\d{4})\.tif[f]?$", re.IGNORECASE)
    monthly_flat_re = re.compile(r"^(\d{4})-(\d{2})\.tif[f]?$", re.IGNORECASE)
    month_only_re = re.compile(r"^(\d{2})\.tif[f]?$", re.IGNORECASE)

    detected_unit = None

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)

        # Flat monthly: 2000-01.tif
        ym = monthly_flat_re.match(fname)
        if ym:
            label = f"{ym.group(1)}-{ym.group(2)}"
            if detected_unit is None:
                detected_unit = "monthly"
            files[label] = fpath
            continue

        # Yearly: 2000.tif
        yy = yearly_re.match(fname)
        if yy:
            label = yy.group(1)
            if detected_unit is None:
                detected_unit = "yearly"
            files[label] = fpath
            continue

        # Nested monthly: 2000/ directory containing 01.tif, 02.tif, ...
        if os.path.isdir(fpath):
            year_dir_name = fname
            year_match = re.match(r"^(\d{4})$", year_dir_name)
            if year_match:
                year_str = year_match.group(1)
                if detected_unit is None:
                    detected_unit = "monthly"
                for sub_fname in sorted(os.listdir(fpath)):
                    sub_path = os.path.join(fpath, sub_fname)
                    m = month_only_re.search(sub_fname)
                    if m:
                        label = f"{year_str}-{m.group(1)}"
                        files[label] = sub_path
                    # Also support "2000-01.tif" inside year dir
                    m2 = monthly_flat_re.match(sub_fname)
                    if m2:
                        label = f"{m2.group(1)}-{m2.group(2)}"
                        files[label] = sub_path

    if not files:
        # Loose match: any .tif with 4 digits
        loose_re = re.compile(r"(\d{4})")
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".tif", ".tiff")):
                m = loose_re.search(fname)
                if m:
                    label = m.group(1)
                    if detected_unit is None:
                        detected_unit = "yearly"
                    files[label] = os.path.join(folder, fname)

    return files, detected_unit or "yearly"


def read_raster(filepath):
    """Read a single-band raster, return (data, profile). Nodata values are converted to NaN."""
    with rasterio.open(filepath) as src:
        # Use masked read — rasterio uses the file's nodata metadata to mask
        data = src.read(1, masked=True)
        profile = src.profile.copy()

        if isinstance(data, np.ma.MaskedArray):
            # For integer rasters, convert to float32 first, then apply mask
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32)
                data[data == src.nodata] = np.nan if src.nodata is not None else data
            else:
                data = data.filled(np.nan).astype(np.float32)
            # If still a masked array, fill remaining masks
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(np.nan).astype(np.float32)
        else:
            data = data.astype(np.float32)

    return data, profile


def load_raster_stack(file_dict, expected_shape=None):
    """
    Load a stack of rasters from {label: filepath} dict.

    Returns: (labels_list, data_cube, profile)
      - labels_list: sorted list of time labels
      - data_cube: 3D numpy array (time, rows, cols)
      - profile: rasterio profile from the first file
    """
    labels = sorted(file_dict.keys())
    if not labels:
        raise ValueError("No TIFF files found.")

    first_data, profile = read_raster(file_dict[labels[0]])
    n_rows, n_cols = first_data.shape

    if expected_shape is not None:
        es_rows, es_cols = expected_shape
        if n_rows != es_rows or n_cols != es_cols:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {(n_rows, n_cols)} "
                f"for label '{labels[0]}'"
            )

    cube = np.full((len(labels), n_rows, n_cols), np.nan, dtype=np.float32)
    cube[0] = first_data

    for i, label in enumerate(labels[1:], start=1):
        data, _ = read_raster(file_dict[label])
        if data.shape != (n_rows, n_cols):
            raise ValueError(
                f"Shape mismatch for '{label}': expected {(n_rows, n_cols)}, "
                f"got {data.shape}"
            )
        cube[i] = data

    return labels, cube, profile


def validate_alignment(y_profile, x_profiles):
    """
    Check that all rasters share the same CRS, resolution, and shape.
    Returns list of warning messages.
    """
    warnings = []
    y_crs = y_profile.get("crs", None)
    y_transform = y_profile.get("transform", None)

    for i, xp in enumerate(x_profiles):
        x_crs = xp.get("crs", None)
        if y_crs is not None and x_crs is not None and y_crs != x_crs:
            warnings.append(f"CRS mismatch: Y vs X[{i}] — {y_crs} vs {x_crs}")
        xt = xp.get("transform", None)
        if y_transform is not None and xt is not None and y_transform != xt:
            warnings.append(f"Transform mismatch: Y vs X[{i}]")
    return warnings


def get_valid_mask(*cubes, min_valid_ratio=0.5):
    """
    Compute a mask where ALL cubes have valid (non-NaN) data.

    Parameters:
      *cubes: 3D arrays (time, rows, cols)
      min_valid_ratio: minimum fraction of time steps that must be valid
                       for a pixel to be included (default 0.5).

    Returns: 2D boolean mask (rows, cols) where True = valid pixel.
    """
    if not cubes:
        raise ValueError("At least one cube required.")

    valid_count = np.zeros(cubes[0].shape[1:], dtype=np.int32)
    total_bands = 0

    for cube in cubes:
        n_bands = cube.shape[0]
        total_bands += n_bands
        # Count valid (non-NaN) observations per pixel per band
        for b in range(n_bands):
            valid_count += (~np.isnan(cube[b])).astype(np.int32)

    # Pixel is valid if enough observations are non-NaN
    required = int(total_bands * min_valid_ratio)
    return valid_count >= required
