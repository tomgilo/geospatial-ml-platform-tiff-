"""Mann-Kendall trend test + Theil-Sen slope analysis for GeoTIFF time series.

Per-pixel MK test → 5-class classification, Z statistic, Sen's slope,
standard error of Sen's slope (SEM), and visualization.
Supports multiprocessing for large raster datasets.
"""

import os
import re
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, Dict, Any, List, Sequence

import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# ── Year auto-discovery ────────────────────────────────────────────────────

def discover_years(data_dir: str) -> List[int]:
    """
    Auto-detect years from TIFF filenames in a directory.

    Supports patterns like: 2006.tif, LAI_2006.tif, 200601.tif (monthly),
    2006_something.tif, etc. Returns sorted list of unique years found.

    Args:
        data_dir: Directory containing yearly/monthly TIFF files.

    Returns:
        Sorted list of detected years (integers).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    years: set = set()
    pattern = re.compile(r'(\d{4})')  # Match exactly 4-digit year

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue
        matches = pattern.findall(fname)
        for m in matches:
            yr = int(m)
            if 1900 <= yr <= 2100:
                years.add(yr)

    if not years:
        raise ValueError(f"未在目录 {data_dir} 中找到任何包含年份的 TIFF 文件。"
                         f"请确保文件名包含 4 位年份，如 2006.tif 或 LAI_2006.tif")

    return sorted(years)


# ── Per-pixel MK computation ────────────────────────────────────────────────

def _mk_single_pixel(
    ts: np.ndarray,
    valid_threshold: float = 0.0,
    minimum_valid_years: int = 8,
    use_tie_correction: bool = True,
) -> Tuple[float, float, float, float, int]:
    """
    Compute MK S, Z, Theil-Sen slope, SEM, and 5-class trend for one pixel.

    Returns:
        (S_statistic, Z, Sen_slope, slope_SEM, trend_class)
        trend_class ∈ {-3, -2, 0, 2, 3}
    """
    valid_mask = np.isfinite(ts) & (ts > valid_threshold)
    nv: int = int(np.sum(valid_mask))
    if nv < minimum_valid_years:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    yv: np.ndarray = ts[valid_mask]
    tv: np.ndarray = np.where(valid_mask)[0].astype(np.float64)

    # ── Compute S & pairwise slopes ──
    s_val: float = 0.0
    n_pairs: int = nv * (nv - 1) // 2
    slopes: np.ndarray = np.empty(n_pairs, dtype=np.float64)
    pos: int = 0

    for k in range(1, nv):
        yk, tk = float(yv[k]), float(tv[k])
        diff: np.ndarray = yk - yv[:k]
        sgn: np.ndarray = np.zeros(k, dtype=np.float64)
        sgn[diff > 0] = 1.0
        sgn[diff < 0] = -1.0
        s_val += np.sum(sgn)
        dt: np.ndarray = tk - tv[:k]
        with np.errstate(divide='ignore', invalid='ignore'):
            seg: np.ndarray = np.where(dt != 0, diff / dt, np.nan)
        kk: int = len(seg)
        slopes[pos:pos + kk] = seg
        pos += kk

    slopes = slopes[:pos]
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return (s_val, np.nan, np.nan, np.nan, 0)

    slopes_sorted: np.ndarray = np.sort(slopes)
    n_eff: int = len(slopes_sorted)
    s_sen: float = float(np.median(slopes_sorted))

    # ── Variance (ties optional) ──
    if use_tie_correction:
        y_round: np.ndarray = np.round(yv, 6)
        _, ic = np.unique(y_round, return_inverse=True)
        freqs: np.ndarray = np.bincount(ic)
        tie_term: float = float(np.sum(freqs * (freqs - 1) * (2 * freqs + 5)))
        var_s: float = (nv * (nv - 1) * (2 * nv + 5) - tie_term) / 18.0
    else:
        var_s = nv * (nv - 1) * (2 * nv + 5) / 18.0

    # ── Z with continuity correction ──
    if var_s <= 0:
        z: float = 0.0
    elif s_val > 0:
        z = (s_val - 1.0) / np.sqrt(var_s)
    elif s_val < 0:
        z = (s_val + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    # ── 5-class classification ──
    az: float = abs(z)
    if s_sen > 0:
        if az > 2.58:       cls = 3
        elif az > 1.96:     cls = 2
        else:               cls = 0
    elif s_sen < 0:
        if az > 2.58:       cls = -3
        elif az > 1.96:     cls = -2
        else:               cls = 0
    else:
        cls = 0

    # ── SEM of Sen's slope ──
    sem_sen: float = np.nan
    if var_s > 0 and n_eff > 1:
        z_alpha: float = 1.96
        c_alpha: float = z_alpha * np.sqrt(var_s)
        m1: float = (n_eff - c_alpha) / 2.0
        m2: float = (n_eff + c_alpha) / 2.0
        idx_low: int = max(0, min(n_eff - 1, int(np.floor(m1))))
        idx_high: int = max(0, min(n_eff - 1, int(np.ceil(m2))))
        if idx_low < idx_high:
            sem_sen = (slopes_sorted[idx_high] - slopes_sorted[idx_low]) / (2.0 * z_alpha)

    return (s_val, z, s_sen, sem_sen, cls)


# ── Parallel worker ─────────────────────────────────────────────────────────

def _process_chunk(
    args: Tuple[np.ndarray, float, int, bool]
) -> List[Tuple[float, float, float, float, int]]:
    chunk_data, valid_threshold, minimum_valid_years, use_tie_correction = args
    results: List[Tuple[float, float, float, float, int]] = []
    for i in range(chunk_data.shape[0]):
        results.append(_mk_single_pixel(
            chunk_data[i, :],
            valid_threshold=valid_threshold,
            minimum_valid_years=minimum_valid_years,
            use_tie_correction=use_tie_correction,
        ))
    return results


# ── Visualization functions ─────────────────────────────────────────────────

def _fmt_compact(v: float) -> str:
    """Format a number compactly for chart labels."""
    if not np.isfinite(v):
        return "N/A"
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1e-4:
        return f"{v:.2e}"
    if av < 0.01:
        return f"{v:.6f}"
    if av < 1:
        return f"{v:.4f}"
    return f"{v:.3f}"


def plot_trend_classification(class_map: np.ndarray, output_path: str) -> str:
    """
    Plot 5-class trend classification map with legend.
    """
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots(figsize=(10, 8))
    # Map values: -3,-2,0,2,3 → remap to 0-4 for colormap
    display_map = class_map.copy().astype(np.float32)
    display_map[class_map <= -3] = 0  # 极显著降低 → red
    display_map[class_map == -2] = 1   # 显著降低 → orange
    display_map[class_map == 0] = 2     # 无变化 → grey/yellow
    display_map[class_map == 2] = 3     # 显著增加 → light green
    display_map[class_map >= 3] = 4     # 极显著增加 → dark green

    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#c0392b', '#e67e22', '#ecf0f1', '#2ecc71', '#1e8449']
    cmap_custom = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap_custom.N)

    im = ax.imshow(display_map, cmap=cmap_custom, norm=norm, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=0.75)
    cbar.ax.set_yticklabels(['极显著降低(-3)', '显著降低(-2)', '无变化(0)',
                              '显著增加(+2)', '极显著增加(+3)'], fontsize=9)

    total = class_map.size
    ax.set_title(f"MK 趋势检验五级分类 (总像素: {total:,})", fontsize=14, fontweight='bold')
    ax.set_xlabel("列", fontsize=10)
    ax.set_ylabel("行", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def plot_slope_distribution(slope_map: np.ndarray, output_path: str) -> str:
    """
    Plot histogram of Sen's slope values.
    """
    slope_flat = slope_map.ravel()
    slope_valid = slope_flat[np.isfinite(slope_flat)]

    if len(slope_valid) == 0:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax1 = axes[0]
    # Filter extreme outliers for better visualization (1st-99th percentile)
    lo, hi = np.percentile(slope_valid, [1, 99])
    slope_trim = slope_valid[(slope_valid >= lo) & (slope_valid <= hi)]
    ax1.hist(slope_trim, bins=80, color='#3498db', edgecolor='white', alpha=0.85, linewidth=0.3)
    ax1.axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='零线')
    ax1.axvline(np.median(slope_valid), color='#2ecc71', linestyle='-', linewidth=1.5,
                label=f"中位数={_fmt_compact(np.median(slope_valid))}")
    ax1.set_xlabel("Sen's 斜率", fontsize=10)
    ax1.set_ylabel("像素数", fontsize=10)
    ax1.set_title(f"Sen's 斜率分布 (1%-99%, n={len(slope_trim):,})", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Right: boxplot-style summary
    ax2 = axes[1]
    ax2.axis('off')
    stats_text = (
        f"Sen's 斜率统计\n"
        f"{'─' * 30}\n"
        f"有效像素: {len(slope_valid):,}\n"
        f"均值:    {_fmt_compact(np.mean(slope_valid))}\n"
        f"中位数:  {_fmt_compact(np.median(slope_valid))}\n"
        f"标准差:  {_fmt_compact(np.std(slope_valid))}\n"
        f"最小值:  {_fmt_compact(np.min(slope_valid))}\n"
        f"最大值:  {_fmt_compact(np.max(slope_valid))}\n"
        f"P5:      {_fmt_compact(np.percentile(slope_valid, 5))}\n"
        f"P25:     {_fmt_compact(np.percentile(slope_valid, 25))}\n"
        f"P75:     {_fmt_compact(np.percentile(slope_valid, 75))}\n"
        f"P95:     {_fmt_compact(np.percentile(slope_valid, 95))}\n"
        f"\n斜率>0:  {(slope_valid > 0).sum():,} ({(slope_valid > 0).sum()/len(slope_valid)*100:.1f}%)\n"
        f"斜率<0:  {(slope_valid < 0).sum():,} ({(slope_valid < 0).sum()/len(slope_valid)*100:.1f}%)"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))

    fig.suptitle("Sen's 斜率分析", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def plot_z_distribution(z_map: np.ndarray, output_path: str) -> str:
    """
    Plot MK Z-score histogram with significance thresholds.
    """
    z_flat = z_map.ravel()
    z_valid = z_flat[np.isfinite(z_flat)]

    if len(z_valid) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))

    lo, hi = np.percentile(z_valid, [0.5, 99.5])
    z_trim = z_valid[(z_valid >= lo) & (z_valid <= hi)]

    # Separate positive and negative for coloring
    z_pos = z_trim[z_trim >= 0]
    z_neg = z_trim[z_trim < 0]

    ax.hist(z_neg, bins=60, color='#e74c3c', edgecolor='white', alpha=0.7,
            linewidth=0.3, label=f"Z<0 ({(z_valid < 0).sum():,} 像素)")
    ax.hist(z_pos, bins=60, color='#2ecc71', edgecolor='white', alpha=0.7,
            linewidth=0.3, label=f"Z>0 ({(z_valid > 0).sum():,} 像素)")

    # Significance thresholds
    for thresh, label, style in [
        (-2.58, 'p<0.01', '#c0392b'),
        (-1.96, 'p<0.05', '#e67e22'),
        (1.96, 'p<0.05', '#27ae60'),
        (2.58, 'p<0.01', '#1e8449'),
    ]:
        ax.axvline(thresh, color=style, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(thresh, ax.get_ylim()[1] * 0.92, label, fontsize=8, color=style,
                ha='center', fontweight='bold')

    ax.set_xlabel("MK Z 统计量", fontsize=11)
    ax.set_ylabel("像素数", fontsize=11)
    ax.set_title(f"MK Z 统计量分布 (0.5%-99.5%, n={len(z_trim):,})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def plot_class_pie(class_map: np.ndarray, output_path: str) -> str:
    """
    Plot pie/donut chart of 5-class proportions.
    """
    cls_flat = class_map.ravel()
    cls_flat = cls_flat[np.isfinite(cls_flat)]

    cls_names = {-3: "极显著降低 (-3)", -2: "显著降低 (-2)", 0: "无明显变化 (0)",
                  2: "显著增加 (+2)", 3: "极显著增加 (+3)"}
    cls_colors = {-3: '#c0392b', -2: '#e67e22', 0: '#bdc3c7', 2: '#2ecc71', 3: '#1e8449'}
    cls_order = [-3, -2, 0, 2, 3]

    labels = []
    sizes = []
    colors = []
    for v in cls_order:
        cnt = int((cls_flat == v).sum())
        if cnt > 0:
            labels.append(f"{cls_names[v]}\n{cnt:,} ({cnt/len(cls_flat)*100:.1f}%)")
            sizes.append(cnt)
            colors.append(cls_colors[v])

    if not sizes:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: donut chart
    wedges, texts = ax1.pie(sizes, labels=None, colors=colors, startangle=90,
                             wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title("五级分类占比", fontsize=13, fontweight='bold')

    # Right: horizontal bar
    bars = ax2.barh(range(len(labels)), sizes, color=colors, edgecolor='white', linewidth=1.2)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("像素数", fontsize=10)
    ax2.set_title("五级分类像素数", fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    for bar, size in zip(bars, sizes):
        ax2.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{size:,}", va='center', fontsize=9, fontweight='bold')

    # Legend
    ax1.legend(wedges, labels, title="分类", loc='center left',
               bbox_to_anchor=(1.1, 0.5), fontsize=9, title_fontsize=10)

    fig.suptitle("MK 趋势检验 — 分类统计", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def plot_sem_map(sem_map: np.ndarray, output_path: str) -> str:
    """
    Plot SEM spatial distribution.
    """
    sem_flat = sem_map.ravel()
    sem_valid = sem_flat[np.isfinite(sem_flat) & (sem_flat > 0)]

    if len(sem_valid) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))

    lo, hi = np.percentile(sem_valid, [1, 99])
    sem_trim = sem_valid[(sem_valid >= lo) & (sem_valid <= hi)]
    ax.hist(sem_trim, bins=80, color='#9b59b6', edgecolor='white', alpha=0.8, linewidth=0.3)
    ax.axvline(np.median(sem_valid), color='#e74c3c', linestyle='-', linewidth=1.5,
               label=f"中位数={_fmt_compact(np.median(sem_valid))}")
    ax.set_xlabel("Sen's 斜率标准误差 (SEM)", fontsize=10)
    ax.set_ylabel("像素数", fontsize=10)
    ax.set_title(f"SEM 分布 (1%-99%, n={len(sem_trim):,})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def generate_all_plots(
    class_map: np.ndarray,
    slope_map: np.ndarray,
    z_map: np.ndarray,
    sem_map: np.ndarray,
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate all MK analysis plots and return path dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts: Dict[str, str] = {}

    charts['classification'] = plot_trend_classification(class_map,
        os.path.join(output_dir, "mk_classification.png"))
    charts['slope'] = plot_slope_distribution(slope_map,
        os.path.join(output_dir, "mk_slope_dist.png"))
    charts['zscore'] = plot_z_distribution(z_map,
        os.path.join(output_dir, "mk_zscore_dist.png"))
    charts['pie'] = plot_class_pie(class_map,
        os.path.join(output_dir, "mk_class_pie.png"))
    charts['sem'] = plot_sem_map(sem_map,
        os.path.join(output_dir, "mk_sem_dist.png"))

    return {k: v for k, v in charts.items() if v}


# ── Main analysis entry point ───────────────────────────────────────────────

def run_mk_analysis(
    data_dir: str,
    years: Optional[Sequence[int]] = None,
    output_dir: str = "",
    valid_threshold: float = 0.0,
    minimum_valid_years: int = 8,
    use_tie_correction: bool = True,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run per-pixel MK trend test + Sen's slope analysis on GeoTIFF time series.

    If years is None, auto-detect from filenames in data_dir.

    Outputs:
      - MK.tif        : 5-class classification
      - MK_Z.tif      : Z statistic
      - MK_slope.tif  : Sen's slope
      - MK_sem.tif    : Standard error of Sen's slope
      - mk_*.png      : Visualization charts

    Args:
        data_dir: Directory with yearly TIFF files.
        years: Year sequence (auto-detected if None).
        output_dir: Output directory (default: data_dir/MK_results).
        valid_threshold: Values > threshold are valid.
        minimum_valid_years: Minimum valid years per pixel.
        use_tie_correction: Tie-group variance correction.
        n_jobs: Parallel workers (default: CPU count - 1).

    Returns:
        Dict with maps, stats, charts, and output file paths.
    """
    # ── 0. Auto-detect years ──
    if years is None:
        years = discover_years(data_dir)

    years_list: List[int] = list(years)
    cdn: int = len(years_list)
    if cdn < minimum_valid_years:
        raise ValueError(f"检测到 {cdn} 个年份，少于最少有效年份 ({minimum_valid_years})。")

    if not output_dir:
        output_dir = os.path.join(data_dir, "MK_results")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Read template ──
    tmpl_file: str = os.path.join(data_dir, f"{years_list[0]}.tif")
    # If exact year.tif not found, try glob
    if not os.path.isfile(tmpl_file):
        # Try to find any file containing the year
        found = False
        for fname in os.listdir(data_dir):
            if fname.lower().endswith(('.tif', '.tiff')) and str(years_list[0]) in fname:
                tmpl_file = os.path.join(data_dir, fname)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"找不到 {years_list[0]} 年的 TIFF 文件: {data_dir}")

    with rasterio.open(tmpl_file) as src:
        profile: Dict[str, Any] = src.profile.copy()
        a0: np.ndarray = src.read(1)
        h, w = a0.shape

    m, n = h, w

    # ── 2. Read all years ──
    n_pixels: int = m * n
    datasum: np.ndarray = np.full((n_pixels, cdn), np.nan, dtype=np.float64)

    for p, year in enumerate(years_list):
        # Find file: try year.tif first, then glob
        fn = os.path.join(data_dir, f"{year}.tif")
        if not os.path.isfile(fn):
            found = False
            for fname in os.listdir(data_dir):
                if fname.lower().endswith(('.tif', '.tiff')) and str(year) in fname:
                    fn = os.path.join(data_dir, fname)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"找不到 {year} 年的 TIFF 文件: {data_dir}")

        with rasterio.open(fn) as src:
            arr: np.ndarray = src.read(1)
            nodata_val = src.nodata
        arr = arr.astype(np.float64)
        if nodata_val is not None:
            arr[arr == nodata_val] = np.nan
        datasum[:, p] = arr.ravel()

    # ── 3. Parallel processing ──
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    chunk_size: int = max(1, n_pixels // (n_jobs * 4))
    chunks: List[np.ndarray] = []
    for start in range(0, n_pixels, chunk_size):
        end: int = min(start + chunk_size, n_pixels)
        chunks.append(datasum[start:end, :].copy())

    chunk_args: List[Tuple[np.ndarray, float, int, bool]] = [
        (chunk, valid_threshold, minimum_valid_years, use_tie_correction)
        for chunk in chunks
    ]

    with Pool(processes=n_jobs) as pool:
        chunk_results: List[List[Tuple[float, float, float, float, int]]] = pool.map(
            _process_chunk, chunk_args
        )

    # ── 4. Assemble ──
    z_flat = np.full(n_pixels, np.nan, dtype=np.float64)
    slope_flat = np.full(n_pixels, np.nan, dtype=np.float64)
    sem_flat = np.full(n_pixels, np.nan, dtype=np.float64)
    cls_flat = np.full(n_pixels, 0, dtype=np.int8)

    idx = 0
    for chunk_res in chunk_results:
        for _, z_val, slope_val, sem_val, cls_val in chunk_res:
            if np.isfinite(z_val):
                z_flat[idx] = z_val
                slope_flat[idx] = slope_val
                cls_flat[idx] = cls_val
            else:
                z_flat[idx] = np.nan
                slope_flat[idx] = np.nan
                cls_flat[idx] = 0
            sem_flat[idx] = sem_val if np.isfinite(sem_val) else np.nan
            idx += 1

    class_map = cls_flat.reshape((m, n))
    slope_map = slope_flat.reshape((m, n))
    z_map = z_flat.reshape((m, n))
    sem_map = sem_flat.reshape((m, n))

    # ── 5. Write TIFFs ──
    output_files: List[str] = []

    cls_profile = profile.copy()
    cls_profile.update({'driver': 'GTiff', 'dtype': 'int8', 'compress': 'lzw'})
    cls_profile.pop('nodata', None)
    mk_path = os.path.join(output_dir, "MK.tif")
    with rasterio.open(mk_path, 'w', **cls_profile) as dst:
        dst.write(class_map.astype(np.int8), 1)
    output_files.append(mk_path)

    float_profile = profile.copy()
    float_profile.update({'driver': 'GTiff', 'dtype': 'float32',
                          'nodata': np.nan, 'compress': 'lzw'})

    z_path = os.path.join(output_dir, "MK_Z.tif")
    with rasterio.open(z_path, 'w', **float_profile) as dst:
        dst.write(z_map.astype(np.float32), 1)
    output_files.append(z_path)

    slope_path = os.path.join(output_dir, "MK_slope.tif")
    with rasterio.open(slope_path, 'w', **float_profile) as dst:
        dst.write(slope_map.astype(np.float32), 1)
    output_files.append(slope_path)

    sem_path = os.path.join(output_dir, "MK_sem.tif")
    with rasterio.open(sem_path, 'w', **float_profile) as dst:
        dst.write(sem_map.astype(np.float32), 1)
    output_files.append(sem_path)

    # ── 6. Statistics ──
    valid_slope = slope_flat[np.isfinite(slope_flat)]
    n_valid = len(valid_slope)
    valid_sem = sem_flat[np.isfinite(sem_flat)]

    stats: Dict[str, Any] = {
        "n_pixels_total": n_pixels,
        "n_pixels_valid": n_valid,
        "n_years": cdn,
        "years": years_list,
        "slope_mean": float(np.mean(valid_slope)) if n_valid > 0 else np.nan,
        "slope_median": float(np.median(valid_slope)) if n_valid > 0 else np.nan,
        "slope_std": float(np.std(valid_slope)) if n_valid > 0 else np.nan,
        "slope_min": float(np.min(valid_slope)) if n_valid > 0 else np.nan,
        "slope_max": float(np.max(valid_slope)) if n_valid > 0 else np.nan,
        "slope_pos_pct": float((valid_slope > 0).sum() / n_valid * 100) if n_valid > 0 else 0,
        "slope_neg_pct": float((valid_slope < 0).sum() / n_valid * 100) if n_valid > 0 else 0,
        "sem_mean": float(np.mean(valid_sem)) if len(valid_sem) > 0 else np.nan,
        "sem_median": float(np.median(valid_sem)) if len(valid_sem) > 0 else np.nan,
    }

    # ── 7. Generate charts ──
    charts_dir = os.path.join(output_dir, "charts")
    charts = generate_all_plots(class_map, slope_map, z_map, sem_map, charts_dir)

    return {
        "class_map": class_map,
        "slope_map": slope_map,
        "z_map": z_map,
        "sem_map": sem_map,
        "profile": profile,
        "output_files": output_files,
        "charts": charts,
        "stats": stats,
    }
