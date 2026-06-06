"""SHAP explainability analysis for geospatial ML models."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 中文字体 ──
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# 支持 TreeExplainer 的模型类型
_TREE_MODEL_TYPES = None

def _get_tree_model_types():
    global _TREE_MODEL_TYPES
    if _TREE_MODEL_TYPES is not None:
        return _TREE_MODEL_TYPES
    types = []
    try:
        from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                       GradientBoostingRegressor)
        types += [RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor]
    except ImportError:
        pass
    try:
        import xgboost as xgb
        types.append(xgb.XGBRegressor)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        types.append(lgb.LGBMRegressor)
    except ImportError:
        pass
    _TREE_MODEL_TYPES = tuple(types)
    return _TREE_MODEL_TYPES


def _is_tree_model(obj):
    """判断是否为 TreeExplainer 支持的树模型。"""
    tree_types = _get_tree_model_types()
    if tree_types and isinstance(obj, tree_types):
        return True
    # 也通过类名兜底判断
    cls_name = type(obj).__name__
    return any(k in cls_name for k in
               ("Forest", "Tree", "Boost", "XGB", "LGBM", "GBR", "GradientBoosting"))


def compute_shap_values(pixel_models, X_sample, model_names, x_names,
                        max_samples=500, sample_indices=None,
                        X_train_sample=None):
    """
    Compute SHAP values for tree-based models using a random pixel subset.

    v3 改进：
    - **shap.Explainer 优先**：自动选择最佳解释器（比直接 TreeExplainer 更鲁棒）
    - **错误全量收集**：所有失败原因写入返回结果，供 UI 展示
    - **feature_importances_ 兜底**：SHAP 全部失败时自动回退到模型内置特征重要性
    - 支持 X_train_sample 作为背景数据（可选）

    Args:
      pixel_models   : dict of model_name -> list of per-pixel models
      X_sample       : (n_pixels, n_time, n_features)
      model_names    : list of model names
      x_names        : list of variable names
      max_samples    : max pixels to sample
      sample_indices : optional pre-selected pixel indices

    Returns:
      dict: model_name -> {
          "values": (n_success, n_features),  # SHAP 或 feature_importances
          "base_value": float,
          "feature_names": list,
          "n_used": int,
          "method": "shap_explainer" | "shap_treeexplainer" | "feature_importances",
          "errors": [str, ...],       # 失败原因（最多 5 条）
          "n_failed": int,
          "n_total": int,
      }
    """
    try:
        import shap
    except ImportError:
        return {
            "_error": "SHAP 库未安装。请在终端运行: pip install shap",
        }

    n_pixels = X_sample.shape[0]
    n_features = X_sample.shape[2] if X_sample.ndim == 3 else X_sample.shape[1]

    # 确定采样像素索引
    if sample_indices is None:
        np.random.seed(42)
        n_sample = min(n_pixels, max_samples)
        sample_indices = np.random.choice(n_pixels, n_sample, replace=False)

    results = {}

    for model_name in model_names:
        if model_name not in pixel_models:
            results[model_name] = {
                "_error": f"模型 '{model_name}' 不在训练结果中",
                "n_total": 0, "n_used": 0, "n_failed": 0,
                "values": np.zeros((0, n_features)), "base_value": 0.0,
                "feature_names": [], "method": "none", "errors": [],
            }
            continue

        models_list = pixel_models[model_name]
        n_total_models = len(models_list)
        n_fitted = sum(1 for m in models_list
                       if m is not None and hasattr(m, "model_") and m.model_ is not None)

        if n_fitted == 0:
            results[model_name] = {
                "_error": f"共 {n_total_models} 个像素模型，0 个有效拟合",
                "n_total": n_total_models, "n_used": 0, "n_failed": 0,
                "values": np.zeros((0, n_features)), "base_value": 0.0,
                "feature_names": [], "method": "none", "errors": [],
            }
            continue

        # ── 收集有效的 (feature_row, model_obj) 对 ──
        valid_pairs = []
        for i in sample_indices:
            if i >= len(models_list) or models_list[i] is None:
                continue
            wrapper = models_list[i]
            if not (hasattr(wrapper, "model_") and wrapper.model_ is not None):
                continue
            model_obj = wrapper.model_

            # 从3D数据中找一个非NaN的时间步用于SHAP计算
            if X_sample.ndim == 3:
                X_i = X_sample[i]                    # (n_time, n_features)
                valid_rows = ~np.isnan(X_i).any(axis=1)
                if not valid_rows.any():
                    continue
                x_i = X_i[valid_rows][0, :]          # 取第一个有效时间步
            else:
                x_i = X_sample[i, :]
                if np.isnan(x_i).any():
                    continue

            # 若提供训练期特征，优先使用其第一个有效时间步
            if X_train_sample is not None and X_train_sample.ndim == 3:
                X_bg = X_train_sample[i]
                valid_bg = ~np.isnan(X_bg).any(axis=1)
                if valid_bg.any():
                    x_i = X_bg[valid_bg][0, :]

            valid_pairs.append((x_i.astype(np.float64), model_obj))

        if len(valid_pairs) < 2:
            results[model_name] = {
                "_error": f"有效像素对不足 ({len(valid_pairs)} 个)，至少需要 2 个",
                "n_total": n_fitted, "n_used": 0, "n_failed": len(valid_pairs),
                "values": np.zeros((0, n_features)), "base_value": 0.0,
                "feature_names": [], "method": "none", "errors": [],
            }
            continue

        feature_names = (list(x_names) if x_names else
                         [f"X{i+1}" for i in range(n_features)])

        # ── 阶段 0：模型预测差异度诊断 ──
        # 从 valid_pairs 中采样 3 个像素，对比不同输入的预测值差异
        pred_check_pairs = valid_pairs[:min(3, len(valid_pairs))]
        all_preds = []
        for x_i, model_obj in pred_check_pairs:
            try:
                pred = float(model_obj.predict(x_i.reshape(1, -1)).ravel()[0])
                all_preds.append(pred)
            except Exception:
                pass
        if len(all_preds) >= 2:
            pred_range = max(all_preds) - min(all_preds)
            print(f"[SHAP-DIAG] {model_name}: 采样 {len(all_preds)} 个像素"
                  f" 预测值={[f'{v:.6g}' for v in all_preds]}"
                  f" 预测值范围={pred_range:.6g}",
                  flush=True)
            if pred_range < 1e-10:
                print(f"[SHAP-DIAG] ⚠️ {model_name}: 模型预测值几乎不变！"
                      f" SHAP 值必然全为零。根因：Y 变量方差过小。",
                      flush=True)

        # ── 阶段 1：逐像素 SHAP 计算 ──
        all_shap_rows, base_values, error_msgs = [], [], []
        success_tree, success_exp, fail_count = 0, 0, 0
        used_method = "shap_treeexplainer"
        n_pairs = len(valid_pairs)

        for idx, (x_i, model_obj) in enumerate(valid_pairs):
            # 每 100 个像素打印一次进度，方便诊断卡住问题
            if idx % 100 == 0 and n_pairs > 200:
                print(f"  SHAP 进度: {idx}/{n_pairs} ({idx*100//n_pairs}%)"
                      f"  [T={success_tree} E={success_exp} F={fail_count}]",
                      flush=True)

            x_2d = x_i.reshape(1, -1)
            ok = False

            # 策略 A：直接用 TreeExplainer（最快，不自动探测模型类型）
            try:
                exp = shap.TreeExplainer(
                    model_obj,
                    feature_perturbation="tree_path_dependent",
                )
                sv = exp.shap_values(x_2d, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[0]
                sv = np.asarray(sv, dtype=np.float64)
                if sv.ndim == 2:
                    sv = sv[0]
                all_shap_rows.append(sv)
                ev = exp.expected_value
                base_values.append(float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev))
                success_tree += 1
                ok = True
            except Exception as e_tree:
                # 策略 B：退回 shap.Explainer（通用，但可能慢）
                try:
                    exp2 = shap.Explainer(model_obj)
                    sv_out = exp2(x_2d, check_additivity=False)
                    sv_arr = np.asarray(sv_out.values, dtype=np.float64)
                    if sv_arr.ndim == 2:
                        sv_arr = sv_arr[0]
                    all_shap_rows.append(sv_arr)
                    bv = sv_out.base_values
                    base_values.append(float(bv[0]) if hasattr(bv, "__len__") else float(bv))
                    success_exp += 1
                    ok = True
                except Exception as e_exp:
                    fail_count += 1
                    if len(error_msgs) < 5:
                        error_msgs.append(
                            f"像素 SHAP 全失败: TreeExplainer={str(e_tree)[:120]}; "
                            f"Explainer={str(e_exp)[:120]}"
                        )

        total_success = success_tree + success_exp
        if success_tree >= success_exp and success_tree > 0:
            used_method = "shap_treeexplainer"
        elif success_exp > 0:
            used_method = "shap_explainer"

        # ── 阶段 2：SHAP 全失败 → feature_importances_ 兜底 ──
        if total_success == 0:
            # 收集所有模型的特征重要性
            fi_rows = []
            fi_count = 0
            for _x_i, model_obj in valid_pairs:
                fi = _get_feature_importances(model_obj, n_features)
                if fi is not None and len(fi) == n_features:
                    fi_rows.append(fi)
                    fi_count += 1
                elif len(error_msgs) < 5:
                    error_msgs.append("模型无 feature_importances_ 属性")

            if fi_count > 0:
                fi_matrix = np.array(fi_rows, dtype=np.float64)
                # 归一化（原始 feature_importances_ 已归一化，此处做均值）
                results[model_name] = {
                    "values": fi_matrix,
                    "base_value": 0.0,
                    "feature_names": feature_names,
                    "n_used": fi_count,
                    "n_total": n_fitted,
                    "n_failed": fail_count,
                    "method": "feature_importances",
                    "errors": error_msgs,
                }
                continue
            else:
                results[model_name] = {
                    "_error": (f"SHAP 和 feature_importances_ 全部失败。"
                               f"共 {len(valid_pairs)} 个像素，{fail_count} 次失败"),
                    "n_total": n_fitted, "n_used": 0, "n_failed": fail_count,
                    "values": np.zeros((0, n_features)), "base_value": 0.0,
                    "feature_names": feature_names, "method": "none",
                    "errors": error_msgs,
                }
                continue

        # ── 阶段 3：聚合 SHAP 结果 ──
        shap_matrix = np.array(all_shap_rows, dtype=np.float64)
        mean_base = float(np.mean(base_values)) if base_values else 0.0

        # ── 诊断：检测 SHAP 全零问题 ──
        shap_abs_max = float(np.abs(shap_matrix).max()) if shap_matrix.size > 0 else 0.0
        shap_is_all_zero = (shap_abs_max < 1e-15)
        if shap_is_all_zero:
            # 采样检查模型预测范围，帮助用户定位问题根因
            y_pred_samples = []
            for x_i, model_obj in valid_pairs[:min(5, len(valid_pairs))]:
                try:
                    pred = model_obj.predict(x_i.reshape(1, -1))
                    y_pred_samples.append(float(pred[0]) if hasattr(pred, '__len__') else float(pred))
                except Exception:
                    pass
            y_pred_str = ", ".join(f"{v:.6g}" for v in y_pred_samples) if y_pred_samples else "未知"
            print(f"[SHAP-DIAG] ⚠️ {model_name}: SHAP 值全为零！"
                  f" 预测值样本: [{y_pred_str}]"
                  f"  SHAP范围: [{shap_matrix.min():.2e}, {shap_matrix.max():.2e}]"
                  f"  base_value={mean_base:.6g}",
                  flush=True)
            # 把诊断信息注入到 error_msgs，供 UI 展示
            if len(error_msgs) < 3:
                error_msgs.append(
                    f"SHAP 值全为零 — 模型预测值几乎恒定 (样本: {y_pred_str})。"
                    f"根因通常是 Y 变量方差极小导致模型简化为常数预测。"
                    f"建议：更换方差更大的 Y 变量，或检查数据质量。"
                )

        results[model_name] = {
            "values": shap_matrix,
            "base_value": mean_base,
            "feature_names": feature_names,
            "n_used": total_success,
            "n_total": n_fitted,
            "n_failed": fail_count,
            "method": used_method,
            "errors": error_msgs,
        }

    return results


def _get_feature_importances(model_obj, n_features):
    """尝试从模型中提取特征重要性。返回 (n_features,) 数组或 None。"""
    # sklearn tree models
    if hasattr(model_obj, "feature_importances_"):
        fi = np.asarray(model_obj.feature_importances_, dtype=np.float64)
        if len(fi) == n_features:
            return fi
    # 尝试从内部 .estimators_ 聚合（仅 RF/ET 等）
    if hasattr(model_obj, "estimators_"):
        try:
            all_fi = [np.asarray(est.feature_importances_, dtype=np.float64)
                      for est in model_obj.estimators_
                      if hasattr(est, "feature_importances_")]
            if all_fi:
                fi = np.mean(all_fi, axis=0, dtype=np.float64)
                if len(fi) == n_features:
                    return fi
        except Exception:
            pass
    return None


def plot_shap_summary(shap_values_dict, output_dir):
    """Generate SHAP summary bar chart and beeswarm-like plot."""
    import shap
    charts = []

    for model_name, data in shap_values_dict.items():
        # 跳过错误条目
        if "_error" in data and data.get("n_used", 0) == 0:
            continue

        sv = data["values"]
        fnames = data["feature_names"]
        n_used = data.get("n_used", sv.shape[0])
        method = data.get("method", "shap_explainer")

        is_fi = (method == "feature_importances")

        # 1. Bar chart
        fig, ax = plt.subplots(figsize=(8, max(4, len(fnames) * 0.45 + 1.5)))

        if is_fi:
            # feature_importances_：取均值作为条形高度
            mean_vals = sv.mean(axis=0)
            label = "mean feature importance"
            title_prefix = "特征重要性（模型内置，非 SHAP）\n"
        else:
            mean_vals = np.abs(sv).mean(axis=0)
            label = "mean(|SHAP value|)"
            title_prefix = "SHAP 特征重要性\n"

        order = np.argsort(mean_vals)[::-1]
        max_v = float(mean_vals.max()) if mean_vals.max() > 0 else 1.0
        shap_is_zero = (max_v < 1e-15)

        if shap_is_zero:
            # SHAP 全零：使用固定宽度柱子，并在标题中标注
            colors = ["#cccccc"] * len(mean_vals)
            ax.barh(range(len(order)), [1.0] * len(order), color=colors, edgecolor="#999999", linewidth=0.5)
            ax.set_xlim(0, 1.5)
            ax.set_xticks([])
            ax.set_xlabel("SHAP 值全为零 (模型预测恒定)", fontsize=10, color="#e74c3c")
        else:
            colors = plt.cm.Blues(0.4 + 0.6 * mean_vals[order] / max_v)
            ax.barh(range(len(order)), mean_vals[order], color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([fnames[i] for i in order], fontsize=10)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f"{model_name} — {title_prefix}(基于 {n_used} 个采样像素)",
                     fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        path = os.path.join(output_dir, f"shap_importance_{model_name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        charts.append(path)

        # 1b. 归一化 Bar chart（解决 Y 值域极小导致柱子不显眼的问题）
        fig_norm, ax_norm = plt.subplots(figsize=(8, max(4, len(fnames) * 0.45 + 1.5)))
        total = mean_vals.sum()
        if total > 0:
            norm_vals = (mean_vals / total) * 100.0
        else:
            norm_vals = mean_vals.copy()
        order_n = np.argsort(norm_vals)[::-1]
        max_n = float(norm_vals.max()) if norm_vals.max() > 0 else 1.0

        if shap_is_zero and total <= 0:
            # 全零时的归一化图
            colors_n = ["#cccccc"] * len(norm_vals)
            ax_norm.barh(range(len(order_n)), [1.0] * len(order_n), color=colors_n, edgecolor="#999999", linewidth=0.5)
            ax_norm.set_xlim(0, 1.5)
            ax_norm.set_xticks([])
            ax_norm.set_xlabel("所有特征贡献均为 0", fontsize=10, color="#e74c3c")
        else:
            colors_n = plt.cm.Blues(0.4 + 0.6 * norm_vals[order_n] / max_n)
            ax_norm.barh(range(len(order_n)), norm_vals[order_n], color=colors_n)
        ax_norm.set_yticks(range(len(order_n)))
        ax_norm.set_yticklabels([fnames[i] for i in order_n], fontsize=10)
        ax_norm.set_xlabel("相对贡献 (%)", fontsize=11)
        ax_norm.set_title(f"{model_name} — {title_prefix}（归一化）\n基于 {n_used} 个采样像素",
                          fontsize=12, fontweight="bold")
        ax_norm.invert_yaxis()
        ax_norm.grid(True, alpha=0.3, axis="x")
        # 在柱子末端标注百分比数值，增强可读性
        for j, idx in enumerate(order_n):
            val = norm_vals[idx]
            ax_norm.text(val + max_n * 0.01, j, f"{val:.1f}%",
                         va="center", ha="left", fontsize=9, color="#333")
        fig_norm.tight_layout()
        path_norm = os.path.join(output_dir, f"shap_importance_{model_name}_norm.png")
        fig_norm.savefig(path_norm, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_norm)
        charts.append(path_norm)

        # 2. Beeswarm — 仅非 feature_importances_ 模式
        if is_fi or sv.shape[1] < 2:
            continue

        n_top = min(sv.shape[1], 10)
        top_idx = order[:n_top]
        sv_top = sv[:, top_idx]
        fnames_top = [fnames[i] for i in top_idx]

        fig2, ax2 = plt.subplots(figsize=(8, max(4, n_top * 0.55 + 1.5)))
        try:
            shap.summary_plot(sv_top, feature_names=fnames_top, show=False,
                              plot_type="dot", max_display=n_top)
            fig2 = plt.gcf()
            fig2.set_size_inches(8, max(4, n_top * 0.55 + 1.5))
            fig2.suptitle(f"{model_name} — SHAP Beeswarm (Top {n_top} features)",
                          fontsize=11, fontweight="bold", y=1.01)
        except Exception:
            ax2.cla()
            for fi, feat in enumerate(fnames_top):
                vals = sv_top[:, fi]
                ax2.scatter(vals, [fi] * len(vals), alpha=0.4, s=15,
                            c=vals, cmap="RdBu_r")
            ax2.set_yticks(range(n_top))
            ax2.set_yticklabels(fnames_top, fontsize=9)
            ax2.set_xlabel("SHAP value", fontsize=10)
            ax2.set_title(f"{model_name} — SHAP 分布", fontsize=11, fontweight="bold")
            ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax2.grid(True, alpha=0.3, axis="x")

        path2 = os.path.join(output_dir, f"shap_beeswarm_{model_name}.png")
        fig2.savefig(path2, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        charts.append(path2)

    return charts


def compute_shap_spatial(pixel_models, X_spatial, model_name, valid_indices,
                         y_profile, output_path, x_names=None, max_pixels=2000):
    """
    Compute per-pixel SHAP values and save as a spatial map.

    Returns the SHAP map (2D array per feature) and plot paths.
    """
    try:
        import shap
    except ImportError:
        return None, None

    n_pixels = X_spatial.shape[0]
    n_features = X_spatial.shape[2] if X_spatial.ndim == 3 else X_spatial.shape[1]

    if model_name not in pixel_models:
        return None, None

    models_list = pixel_models[model_name]

    # Sample pixels for efficiency
    sample_n = min(n_pixels, max_pixels)
    sample_idx = (np.random.choice(n_pixels, sample_n, replace=False)
                  if n_pixels > sample_n else np.arange(n_pixels))

    # Compute SHAP for each sampled pixel using the first time step
    shap_per_pixel = np.full((n_pixels, n_features), np.nan)
    success = 0

    for i in sample_idx:
        if models_list[i] is None:
            continue
        wrapper = models_list[i]
        if not (hasattr(wrapper, "model_") and wrapper.model_ is not None):
            continue

        if X_spatial.ndim == 3:
            x_i = X_spatial[i, 0, :]
        else:
            x_i = X_spatial[i, :]

        if np.isnan(x_i).any():
            continue

        try:
            model_obj = wrapper.model_
            explainer = shap.TreeExplainer(model_obj)
            sv = explainer.shap_values(x_i.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.array(sv)
            if sv.ndim == 2:
                sv = sv[0]
            shap_per_pixel[i] = sv
            success += 1
        except Exception:
            try:
                explainer2 = shap.Explainer(wrapper.model_, x_i.reshape(1, -1))
                sv2 = explainer2(x_i.reshape(1, -1))
                sv_arr = sv2.values
                if sv_arr.ndim == 2:
                    sv_arr = sv_arr[0]
                shap_per_pixel[i] = sv_arr.astype(np.float64)
                success += 1
            except Exception:
                pass

    print(f"[SHAP Spatial] {model_name}: 成功 {success}/{len(sample_idx)} 像素")

    if success == 0:
        return None, None

    # For each feature, save a spatial map of SHAP
    x_names_list = x_names or [f"X{i+1}" for i in range(n_features)]
    plot_paths = []
    n_rows, n_cols = y_profile["height"], y_profile["width"]

    for f in range(n_features):
        full = np.full(n_rows * n_cols, np.nan, dtype=np.float32)
        # valid_indices 对应 shap_per_pixel 的行索引
        for vi, pi in enumerate(valid_indices):
            if vi < n_pixels and not np.isnan(shap_per_pixel[vi, f]):
                full[pi] = shap_per_pixel[vi, f]
        shap_map = full.reshape(n_rows, n_cols)

        fig, ax = plt.subplots(figsize=(7, 6))
        vmax = np.nanmax(np.abs(shap_map))
        vmax = vmax if (vmax is not None and not np.isnan(vmax) and vmax > 0) else 1.0
        im = ax.imshow(shap_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8, label="SHAP value")
        fname_safe = str(x_names_list[f]).replace(" ", "_").replace("/", "_")
        ax.set_title(f"{model_name} — {x_names_list[f]} 的 SHAP 空间分布",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("列", fontsize=9)
        ax.set_ylabel("行", fontsize=9)

        out_dir = os.path.dirname(output_path) if output_path else "."
        if not out_dir:
            out_dir = "."
        fpath = os.path.join(out_dir, f"shap_spatial_{model_name}_{fname_safe}.png")
        fig.tight_layout()
        fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        plot_paths.append(fpath)

    return shap_per_pixel, plot_paths
