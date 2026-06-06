"""
SHAP 结果诊断脚本
运行方式: cd G:\claude测试\geospatial_ml_app && .venv\Scripts\python.exe diagnose_shap.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import discover_tiff_files, load_raster_stack, get_valid_mask
from src.preprocessing import build_design_matrix, scale_data
from src.trainer import train_all_pixels
from src.config import ModelConfig

# ========== 请修改为你的实际路径 ==========
SHAP_Y = r"G:\\claude测试\\geospatial_ml_app\\test_data\\Y"   # Y 文件夹
SHAP_X = [                                              # X 文件夹列表
    r"G:\\claude测试\\geospatial_ml_app\\test_data\\X\\dem",
    r"G:\\claude测试\\geospatial_ml_app\\test_data\\X\\lai",
]
MODEL_NAME = "rf"
MODEL_PARAMS = {"n_estimators": 100, "max_depth": 10}
# ==========================================

print("=" * 60)
print("SHAP 诊断报告")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载数据...")
yf, yu = discover_tiff_files(SHAP_Y)
x_cubes, x_names = [], []
for xf in SHAP_X:
    xfd, _ = discover_tiff_files(xf)
    if xfd:
        xl, xc, xp = load_raster_stack(xfd)
        x_cubes.append(xc)
        x_names.append(os.path.basename(xf))

n_time = yu.shape[0]
mask = get_valid_mask(yu, *x_cubes, min_valid_ratio=0.3)
n_valid = mask.sum()
print(f"  时间步: {n_time}, 有效像素: {n_valid}")

# 2. 构建设计矩阵
y_tr_idx = list(range(n_time))
y_pr_idx = y_tr_idx
yt2d, Xt3d, yp2d, Xp3d, vi = build_design_matrix(
    yu, x_cubes, y_tr_idx, y_pr_idx,
    y_tr_idx, [y_tr_idx[-1]]*len(y_pr_idx) if not y_pr_idx else y_pr_idx,
    mask
)
Xt_sc, Xp_sc, _ = scale_data(Xt3d, Xp3d, method="standard")

n_features = Xt3d.shape[2]
print(f"  特征数: {n_features} ({x_names})")
print(f"  X_train 形状: {Xt3d.shape}  (像素, 时间, 特征)")
print(f"  Y_train 形状: {yt2d.shape}  (像素, 时间)")

# 3. 检查 Y 值域
print("\n[2] Y 值域检查...")
print(f"  Y 最小值: {np.nanmin(yt2d):.6f}")
print(f"  Y 最大值: {np.nanmax(yt2d):.6f}")
print(f"  Y 均值:   {np.nanmean(yt2d):.6f}")
print(f"  Y 标准差: {np.nanstd(yt2d):.6f}")

# 4. 训练少量像素模型并检查 R²
print("\n[3] 模型训练质量抽查 (随机 20 个像素)...")
mc = ModelConfig(name=MODEL_NAME, params=MODEL_PARAMS, enabled=True)
np.random.seed(42)
sample_pixels = np.random.choice(yt2d.shape[0], min(20, yt2d.shape[0]), replace=False)

r2_list = []
for i in sample_pixels:
    X_i = Xt_sc[i]          # (n_time, n_features)
    y_i = yt2d[i]           # (n_time,)
    valid = ~(np.isnan(X_i).any(axis=1) | np.isnan(y_i))
    if valid.sum() < 3:
        continue
    Xv, yv = X_i[valid], y_i[valid]

    from src.models import get_model
    m = get_model(MODEL_NAME, MODEL_PARAMS)
    m.fit_timed(Xv, yv)
    pred = m.model_.predict(Xv)

    ss_res = np.sum((yv - pred) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_list.append(r2)

if r2_list:
    print(f"  R² 中位数: {np.median(r2_list):.4f}")
    print(f"  R² 均值:   {np.mean(r2_list):.4f}")
    print(f"  R² 范围:   [{min(r2_list):.4f}, {max(r2_list):.4f}]")
    if np.median(r2_list) < 0.1:
        print("  ⚠️ 警告: R² 接近 0，模型几乎没有学到模式，SHAP 值自然会很小")
    elif np.median(r2_list) > 0.8:
        print("  ✅ 模型拟合良好，SHAP 值小可能是 Y 值域本身很小")
else:
    print("  无法计算 R²")

# 5. 直接计算一个像素的 SHAP 值看看
print("\n[4] 单像素 SHAP 值抽查...")
try:
    import shap
    i = sample_pixels[0]
    X_i = Xt_sc[i]
    y_i = yt2d[i]
    valid = ~(np.isnan(X_i).any(axis=1) | np.isnan(y_i))
    Xv, yv = X_i[valid], y_i[valid]

    m = get_model(MODEL_NAME, MODEL_PARAMS)
    m.fit_timed(Xv, yv)

    x_2d = Xv[0:1, :]   # 取第一个时间步
    explainer = shap.TreeExplainer(m.model_, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(x_2d, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv).flatten()
    ev = explainer.expected_value
    base = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    print(f"  像素 {i} 的 SHAP 值: {sv}")
    print(f"  绝对值范围: [{np.abs(sv).min():.6f}, {np.abs(sv).max():.6f}]")
    print(f"  绝对值均值: {np.abs(sv).mean():.6f}")
    print(f"  base_value: {base:.6f}")
    print(f"  该像素 Y 均值: {np.mean(yv):.6f}")

    # 检查模型预测值
    pred_all = m.model_.predict(Xv)
    print(f"  该像素预测值范围: [{pred_all.min():.6f}, {pred_all.max():.6f}]")
    print(f"  预测值标准差: {np.std(pred_all):.6f}")

except Exception as e:
    print(f"  SHAP 计算出错: {e}")

print("\n" + "=" * 60)
print("诊断结束")
print("=" * 60)
