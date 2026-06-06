"""Visualization: high-quality charts and maps for model evaluation."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

# ── 中文字体设置 ──
_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
    "DejaVu Sans"
]
_FOUND_FONT = None
for _f in _FONT_CANDIDATES:
    try:
        from matplotlib.font_manager import FontProperties
        FontProperties(family=_f)
        _FOUND_FONT = _f
        break
    except Exception:
        continue
if _FOUND_FONT:
    plt.rcParams["font.sans-serif"] = [_FOUND_FONT] + _FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"  # for $R^2$ rendering
plt.rcParams["figure.dpi"] = 100


# ── 模型中文名映射 ──
MODEL_CN = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "rf": "随机森林",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "extratrees": "Extra Trees",
    "gbr": "GBR",
    "svr": "SVR",
    "mlp": "MLP",
    "knn": "KNN",
}


def _cn(name):
    """返回模型中文名，未知则返回原名。"""
    return MODEL_CN.get(name, name)


# ═══════════════════════════════════════════════════════════════
# 散点图：预测值 vs 观测值
# ═══════════════════════════════════════════════════════════════
def plot_scatter_comparison(y_true, y_pred, model_name, output_path, x_label="观测值"):
    """散点对比图。"""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]

    if len(yt) < 2:
        return None

    from sklearn.metrics import r2_score
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    n_points = len(yt)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(yt, yp, alpha=0.25, s=8, c="#2196F3", edgecolors="none", rasterized=True)

    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    padding = (lims[1] - lims[0]) * 0.05
    lims[0] -= padding
    lims[1] += padding
    ax.plot(lims, lims, "--", color="#F44336", linewidth=1.2, alpha=0.7, label="1:1线")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # R² 用 mathtext 避免方框
    ax.text(0.03, 0.97, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nN = {n_points:,}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#ccc", alpha=0.9))

    ax.set_xlabel(f"{x_label}", fontsize=11)
    ax.set_ylabel(f"{_cn(model_name)} 预测值", fontsize=11)
    ax.set_title(f"{_cn(model_name)}: 预测值 vs 观测值", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 残差直方图
# ═══════════════════════════════════════════════════════════════
def plot_residual_histogram(y_true, y_pred, model_name, output_path):
    """残差分布直方图。"""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    residuals = y_pred[valid] - y_true[valid]
    if len(residuals) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#F44336", linestyle="--", linewidth=1.2, label="零误差线")
    mean_err = np.mean(residuals)
    ax.axvline(mean_err, color="#FF9800", linestyle="-", linewidth=1.2,
               label=f"均值误差 = {mean_err:.4f}")
    ax.set_xlabel("残差 (预测值 - 观测值)", fontsize=11)
    ax.set_ylabel("频数", fontsize=11)
    ax.set_title(f"{_cn(model_name)}: 残差分布", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 误差分布箱线图
# ═══════════════════════════════════════════════════════════════
def plot_error_distribution(metrics_list, model_names, output_path):
    """RMSE 分布箱线图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    data, labels = [], []
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(model_names), 1)))

    for i, model_name in enumerate(model_names):
        rmses = [m["rmse"] for m in metrics_list.get(model_name, []) if not np.isnan(m["rmse"])]
        if rmses:
            data.append(rmses)
            labels.append(_cn(model_name))

    if not data:
        plt.close(fig)
        return None

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     showmeans=True, meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("各模型 RMSE 分布对比", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 特征重要性
# ═══════════════════════════════════════════════════════════════
def plot_feature_importance(fi_results, x_names, output_path):
    """特征重要性柱状图。"""
    if not fi_results:
        return None

    n_models = len(fi_results)
    ncols = min(n_models, 3)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, fi) in enumerate(fi_results.items()):
        ax = axes[idx]
        mean_fi = fi["mean"]
        std_fi = fi["std"]
        x = np.arange(len(mean_fi))
        ax.bar(x, mean_fi, yerr=std_fi, color="#2196F3", capsize=4, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        labels = x_names if x_names and len(x_names) == len(mean_fi) else [f"变量{i+1}" for i in range(len(mean_fi))]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{_cn(model_name)} — 特征重要性", fontsize=11, fontweight="bold")
        ax.set_ylabel("重要性", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 残差空间分布图
# ═══════════════════════════════════════════════════════════════
def plot_residual_map(residuals_1d, valid_indices, ref_profile, output_path, title="残差"):
    """空间残差分布图。"""
    n_rows, n_cols = ref_profile["height"], ref_profile["width"]
    full = np.full(n_rows * n_cols, np.nan, dtype=np.float32)
    full[valid_indices] = residuals_1d
    residual_map = full.reshape(n_rows, n_cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanmax(np.abs(residual_map))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1
    im = ax.imshow(residual_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("残差", fontsize=10)
    ax.set_title(f"{_cn(title.split('—')[0].strip())} — 残差空间分布" if "—" in title else title,
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("列", fontsize=10)
    ax.set_ylabel("行", fontsize=10)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 性能指标柱状对比图
# ═══════════════════════════════════════════════════════════════
def plot_metric_bar(summary_df, output_path):
    """模型性能指标柱状对比图。"""
    if summary_df is None or summary_df.empty:
        return None
    df = summary_df.copy()
    valid_r2 = "mean_r2" in df.columns and df["mean_r2"].notna().any()

    fig, axes = plt.subplots(1, 2 if valid_r2 else 1, figsize=(12 if valid_r2 else 6, 5))
    if not valid_r2:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    # RMSE bar
    if "mean_rmse" in df.columns and df["mean_rmse"].notna().any():
        order = df.sort_values("mean_rmse")["model"].values
        vals = [df[df["model"] == m]["mean_rmse"].values[0] for m in order]
        labels = [_cn(m) for m in order]
        axes[0].barh(range(len(labels)), vals, color=colors[:len(labels)], alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].set_xlabel("RMSE", fontsize=10)
        axes[0].set_title("各模型 RMSE 对比", fontsize=12, fontweight="bold")
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis="x")

    # R² bar (if available)
    if valid_r2:
        order = df.sort_values("mean_r2", ascending=False)["model"].values
        vals = [df[df["model"] == m]["mean_r2"].values[0] for m in order]
        labels = [_cn(m) for m in order]
        axes[1].barh(range(len(labels)), vals, color=colors[:len(labels)], alpha=0.85, edgecolor="white")
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].set_xlabel("$R^2$", fontsize=10)
        axes[1].set_title("各模型 $R^2$ 对比", fontsize=12, fontweight="bold")
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis="x")

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 主函数：生成全部图表
# ═══════════════════════════════════════════════════════════════
def generate_all_charts(pixel_models, y_test_2d, X_test_3d, all_metrics,
                        valid_indices, ref_profile, output_dir,
                        x_names=None, summary_df=None):
    """生成全部可视化图表。"""
    os.makedirs(output_dir, exist_ok=True)
    charts = []

    # 1. 预测-观测散点图 (每模型一张)
    for model_name, models_list in pixel_models.items():
        all_preds, all_truths = [], []
        for i, model in enumerate(models_list):
            if model is None:
                continue
            try:
                pred = model.predict(X_test_3d[i])
                y_label = "观测值" if models_list is list(pixel_models.values())[0] else "观测值"
                all_preds.append(pred[0])
                all_truths.append(y_test_2d[i, 0])
            except Exception:
                pass
        if all_preds:
            fpath = os.path.join(output_dir, f"散点图_{_cn(model_name)}.png")
            plot_scatter_comparison(np.array(all_truths), np.array(all_preds), model_name, fpath)
            charts.append(fpath)

            # 残差直方图
            fpath2 = os.path.join(output_dir, f"残差分布_{_cn(model_name)}.png")
            plot_residual_histogram(np.array(all_truths), np.array(all_preds), model_name, fpath2)
            charts.append(fpath2)

    # 2. RMSE 箱线图
    fpath = os.path.join(output_dir, "误差分布对比.png")
    result = plot_error_distribution(all_metrics, list(pixel_models.keys()), fpath)
    if result:
        charts.append(fpath)

    # 3. 残差空间分布图
    for model_name, models_list in pixel_models.items():
        residuals = []
        for i, model in enumerate(models_list):
            if model is None:
                residuals.append(np.nan)
                continue
            try:
                pred = model.predict(X_test_3d[i])
                residuals.append(y_test_2d[i, 0] - pred[0])
            except Exception:
                residuals.append(np.nan)
        fpath = os.path.join(output_dir, f"残差空间_{_cn(model_name)}.png")
        plot_residual_map(np.array(residuals), valid_indices, ref_profile, fpath,
                          title=f"{_cn(model_name)} — 残差")
        charts.append(fpath)

    # 4. 特征重要性
    from .evaluation import compute_feature_importance
    n_feat = X_test_3d.shape[2]
    fi_dict = {}
    for mn in pixel_models:
        fi = compute_feature_importance(pixel_models, mn, n_feat)
        if fi is not None:
            fi_dict[mn] = fi
    if fi_dict:
        fpath = os.path.join(output_dir, "特征重要性.png")
        plot_feature_importance(fi_dict, x_names, fpath)
        charts.append(fpath)

    # 5. 模型性能柱状对比 (summary_df needed)
    if summary_df is not None:
        fpath = os.path.join(output_dir, "模型性能对比.png")
        plot_metric_bar(summary_df, fpath)
        charts.append(fpath)

    return charts
