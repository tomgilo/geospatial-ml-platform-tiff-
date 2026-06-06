"""
Generate synthetic TIFF files for testing the geospatial ML application.

Creates sample data with a known relationship:
  Y = 2.5*X1 + 1.5*X2 + 0.5*X3 + sin(lat) + noise

Usage: python generate_test_data.py
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Output directory
TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# Raster parameters
NROWS, NCOLS = 30, 40
TRANSFORM = from_origin(120.0, 36.0, 0.05, 0.05)  # 0.05 degree resolution
CRS = "EPSG:4326"
NODATA = -9999.0


def create_raster(data, filepath, nodata=NODATA):
    """Write a single-band GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "height": NROWS,
        "width": NCOLS,
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "transform": TRANSFORM,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def generate_synthetic_data():
    """Generate synthetic test TIFF files."""

    # Create spatial pattern (latitude gradient)
    lat = np.linspace(36.0, 37.5, NROWS).reshape(-1, 1)
    lon = np.linspace(120.0, 122.0, NCOLS).reshape(1, -1)
    spatial_pattern = np.sin(lat * np.pi / 2) * 10

    # Create folders
    y_dir = os.path.join(TEST_DIR, "Y_variable")
    x1_dir = os.path.join(TEST_DIR, "X1_temperature")
    x2_dir = os.path.join(TEST_DIR, "X2_precipitation")
    x3_dir = os.path.join(TEST_DIR, "X3_elevation")

    for d in [y_dir, x1_dir, x2_dir, x3_dir]:
        os.makedirs(d, exist_ok=True)

    # Generate data for years 2000-2015
    np.random.seed(42)
    years = list(range(2000, 2016))

    for yr in years:
        # X1: Temperature proxy, with temporal trend
        x1_base = 20 + spatial_pattern + 0.15 * (yr - 2000)
        x1 = x1_base + np.random.normal(0, 1.0, (NROWS, NCOLS))

        # X2: Precipitation proxy, with oscillation
        x2_base = 800 + spatial_pattern * 5 + 20 * np.sin((yr - 2000) * np.pi / 4)
        x2 = x2_base + np.random.normal(0, 30, (NROWS, NCOLS))

        # X3: Elevation (static + small noise)
        x3_base = 500 + (lat - 36) * 200 + lon * 10
        x3 = x3_base + np.random.normal(0, 2, (NROWS, NCOLS))

        # Y = 2.5*X1 + 1.5*X2 + 0.5*X3 + spatial_pattern + noise
        y = (2.5 * x1 + 1.5 * x2 + 0.5 * x3 +
             spatial_pattern * 3 + np.random.normal(0, 5, (NROWS, NCOLS)))

        # Add some NaN regions (simulate water/missing data) in corners
        if yr == 2010:
            x1[0:5, 0:5] = np.nan
            x2[0:5, 0:5] = np.nan
            y[0:5, 0:5] = np.nan

        create_raster(x1, os.path.join(x1_dir, f"{yr}.tif"))
        create_raster(x2, os.path.join(x2_dir, f"{yr}.tif"))
        create_raster(x3, os.path.join(x3_dir, f"{yr}.tif"))
        create_raster(y, os.path.join(y_dir, f"{yr}.tif"))

    print(f"[OK] Generated test data in: {TEST_DIR}")
    print(f"   Y: {y_dir} ({len(years)} files)")
    print(f"   X1: {x1_dir} ({len(years)} files)")
    print(f"   X2: {x2_dir} ({len(years)} files)")
    print(f"   X3: {x3_dir} ({len(years)} files)")
    print(f"   Rasters: {NROWS}x{NCOLS}, CRS: {CRS}")
    print(f"   True relationship: Y = 2.5*X1 + 1.5*X2 + 0.5*X3 + spatial")


if __name__ == "__main__":
    generate_synthetic_data()
