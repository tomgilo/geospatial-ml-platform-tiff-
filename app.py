"""
地理空间TIFF机器学习平台
支持年度/月度数据 | 12种模型 | 空间分析 | 模型集成 | PDF报告
"""

import os, sys, time, json, yaml
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.config import AppConfig, ModelConfig
from src.pipeline import Pipeline

st.set_page_config(
    page_title="地理空间TIFF机器学习",
    page_icon="\U0001f30d",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("\U0001f30d 地理空间TIFF机器学习平台")
st.caption("栅格数据ML建模 | 12种算法 | 年/月数据支持 | 空间分析 | 模型集成")

# ── 全局深色科技风主题 ──
st.markdown("""
<style>
/* === 主背景 === */
.stApp {
    background: linear-gradient(135deg, #070d1a 0%, #0a1220 60%, #06100d 100%) !important;
}
.main .block-container {
    padding-top: 1.5rem !important;
}

/* === 侧边栏 === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #081020 100%) !important;
    border-right: 1px solid rgba(52,152,219,0.25) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #90bedd !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4da6d9 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid rgba(52,152,219,0.2);
    padding-bottom: 5px;
    margin-top: 12px !important;
}
[data-testid="stSidebar"] .stCaption { color: rgba(77,166,217,0.45) !important; font-size: 0.68rem !important; }

/* === 标题层级 === */
h1 { color: #5dade2 !important; letter-spacing: 1px; }
h2 { color: #6ab0d4 !important; }
h3 { color: #88c4e0 !important; }
p, .stMarkdown p { color: #9bbdd6 !important; }
.stCaption { color: rgba(155,189,214,0.65) !important; }

/* === 指标卡 === */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(12,26,52,0.95) 0%, rgba(8,18,36,0.95) 100%) !important;
    border: 1px solid rgba(52,152,219,0.28) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
[data-testid="stMetricLabel"] { color: rgba(144,190,221,0.75) !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #5dade2 !important; font-weight: 700 !important; }

/* === 主按钮 === */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #155d8a 0%, #1a5276 100%) !important;
    border: 1px solid rgba(93,173,226,0.45) !important;
    border-radius: 8px !important;
    color: #e8f4fd !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(21,93,138,0.5) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1f7ab5 0%, #155d8a 100%) !important;
    box-shadow: 0 6px 22px rgba(31,122,181,0.55) !important;
    transform: translateY(-1px) !important;
}
/* 普通按钮 */
.stButton > button:not([kind="primary"]) {
    background: rgba(10,22,44,0.85) !important;
    border: 1px solid rgba(52,152,219,0.28) !important;
    border-radius: 8px !important;
    color: #6eb5d8 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(52,152,219,0.6) !important;
    color: #a0d2ef !important;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,18,36,0.7) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid rgba(52,152,219,0.18) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #6eb5d8 !important;
    border-radius: 7px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(21,93,138,0.5) !important;
    color: #a0d2ef !important;
}

/* === Expanders === */
[data-testid="stExpander"] {
    border: 1px solid rgba(52,152,219,0.2) !important;
    border-radius: 10px !important;
    background: rgba(8,18,36,0.6) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary { color: #6eb5d8 !important; }
[data-testid="stExpander"] summary:hover { color: #a0d2ef !important; }

/* === 提示框 === */
[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 3px !important; }

/* === 输入框 === */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(8,18,36,0.8) !important;
    border: 1px solid rgba(52,152,219,0.25) !important;
    border-radius: 8px !important;
    color: #a0c8e0 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: rgba(52,152,219,0.65) !important;
    box-shadow: 0 0 0 2px rgba(52,152,219,0.12) !important;
}

/* === Selectbox === */
.stSelectbox > div > div {
    background: rgba(8,18,36,0.8) !important;
    border: 1px solid rgba(52,152,219,0.25) !important;
    border-radius: 8px !important;
}

/* === Checkbox / Radio === */
.stCheckbox label, .stRadio label { color: #90bedd !important; }

/* === 数据表格 === */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(52,152,219,0.2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* === 进度条 === */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #155d8a, #2980b9, #5dade2) !important;
    border-radius: 4px !important;
}

/* === 分割线 === */
hr { border-color: rgba(52,152,219,0.18) !important; }

/* === 代码块 === */
.stCode, code { background: rgba(8,18,36,0.9) !important; border: 1px solid rgba(52,152,219,0.15) !important; border-radius: 6px !important; }

/* === Download 按钮 === */
.stDownloadButton > button {
    background: rgba(8,18,36,0.85) !important;
    border: 1px solid rgba(46,204,113,0.35) !important;
    border-radius: 8px !important;
    color: #82e0aa !important;
}
.stDownloadButton > button:hover {
    border-color: rgba(46,204,113,0.65) !important;
    box-shadow: 0 4px 14px rgba(46,204,113,0.2) !important;
}

/* === Slider === */
[data-baseweb="slider"] [role="slider"] { background: #2980b9 !important; }
[data-baseweb="slider"] [data-testid="stSlider"] { color: #6eb5d8 !important; }

/* === Spinner === */
.stSpinner > div { border-top-color: #3498db !important; }

/* === Popover === */
[data-testid="stPopover"] button {
    background: rgba(8,18,36,0.85) !important;
    border: 1px solid rgba(52,152,219,0.28) !important;
    border-radius: 8px !important;
    color: #6eb5d8 !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)

# 导航
page = st.sidebar.radio(
    "\U0001f4d1 功能导航",
    ["\U0001f3e0 首页与运行", "\U0001f50d 数据探索",
     "\U0001f4ca 结果分析", "\U0001f4e6 模型复用",
     "\U0001f52c SHAP解释", "\U0001f4ca MK趋势检验", "\u2699\ufe0f 配置历史"]
)

# ========== Session State ==========
defaults = {
    "results": None, "pipeline": None, "explore_cache": {},
    "config_history": [], "y_folder": "", "x_folders_str": "",
    "output_dir": os.path.abspath("./outputs"),
    "data_year_start": None, "data_year_end": None,
    "time_mode": "yearly",  # yearly or monthly
    "prediction_mode": "simulation",  # simulation or extrapolation
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def detect_time_range(folder, mode="yearly"):
    """自动从文件名检测时间范围。支持嵌套目录的月度数据。"""
    if not folder or not os.path.isdir(folder):
        return None, None
    from src.data_loader import discover_tiff_files
    yf, _ = discover_tiff_files(folder)
    if not yf:
        return None, None
    years = []
    for label in yf.keys():
        y = int(label.split("-")[0]) if "-" in label else int(label)
        years.append(y)
    return min(years), max(years)


# ============================================================
# 页面1: 首页与运行
# ============================================================
if page == "\U0001f3e0 首页与运行":

    # ── 首页地球动画(未训练时显示) ──
    if st.session_state.results is None:
        st.components.v1.html("""
<!DOCTYPE html>
<html>
<head>
<style>
*{box-sizing:border-box;margin:0;padding:0;}

/* ── Keyframes ── */
@keyframes contDrift{from{transform:translateX(-6px)}to{transform:translateX(6px)}}
@keyframes gridDrift{from{transform:translateX(-4px)}to{transform:translateX(4px)}}
@keyframes cld1{0%,100%{transform:translate(0,0)}33%{transform:translate(10px,-5px)}66%{transform:translate(6px,-2px)}}
@keyframes cld2{0%,100%{transform:translate(0,0)}33%{transform:translate(-8px,-4px)}66%{transform:translate(-4px,-6px)}}
@keyframes cld3{0%,100%{transform:translate(0,0)}33%{transform:translate(6px,-3px)}66%{transform:translate(9px,2px)}}
@keyframes cld4{0%,100%{transform:translate(0,0)}33%{transform:translate(-7px,-2px)}66%{transform:translate(-10px,-4px)}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes atmosPulse{0%,100%{opacity:.55;transform:scale(1)}50%{opacity:.85;transform:scale(1.025)}}
@keyframes twinkle{0%,100%{opacity:.15}50%{opacity:1}}
@keyframes scanAnim{from{top:-3px}to{top:100%}}
@keyframes orbit1{from{transform:rotate(0deg) translateX(130px) rotate(0deg)}to{transform:rotate(360deg) translateX(130px) rotate(-360deg)}}
@keyframes orbit2{from{transform:rotate(115deg) translateX(150px) rotate(-115deg)}to{transform:rotate(475deg) translateX(150px) rotate(-475deg)}}
@keyframes orbit3{from{transform:rotate(240deg) translateX(118px) rotate(-240deg)}to{transform:rotate(-120deg) translateX(118px) rotate(120deg)}}
@keyframes dataRise{0%{opacity:0;transform:translateY(0) scale(0)}15%{opacity:1;transform:translateY(-12px) scale(1)}80%{opacity:.4;transform:translateY(-55px) scale(.7)}100%{opacity:0;transform:translateY(-72px) scale(0)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── Scene ── */
.scene{
  position:relative;width:100%;height:390px;
  background:radial-gradient(ellipse at 38% 28%,#0b1d3a 0%,#07111f 45%,#030b14 100%);
  border-radius:16px;overflow:hidden;
}

/* ── Stars ── */
.star{position:absolute;background:#fff;border-radius:50%;}

/* ── HUD corners ── */
.corner{position:absolute;width:22px;height:22px;border-color:rgba(52,152,219,.45);border-style:solid;}
.tl{top:13px;left:13px;border-width:2px 0 0 2px;border-radius:2px 0 0 0;}
.tr{top:13px;right:13px;border-width:2px 2px 0 0;border-radius:0 2px 0 0;}
.bl{bottom:13px;left:13px;border-width:0 0 2px 2px;border-radius:0 0 0 2px;}
.br{bottom:13px;right:13px;border-width:0 2px 2px 0;border-radius:0 0 2px 0;}

/* ── HUD text ── */
.hud{position:absolute;font:600 9px/1 'Courier New',monospace;letter-spacing:1.5px;color:rgba(77,166,217,.6);}

/* ── Scan line ── */
.scanline{position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,rgba(52,152,219,.18),transparent);
  animation:scanAnim 5s linear infinite;pointer-events:none;}

/* ── Orbit container (anchored at scene center) ── */
.orbit-anchor{position:absolute;top:50%;left:50%;width:0;height:0;}

/* ── Orbit rings ── */
.ring{position:absolute;border-radius:50%;border-style:solid;
  transform:translate(-50%,-50%) rotateX(70deg);}

/* ── Satellite tracks ── */
.sat-track{position:absolute;top:0;left:0;width:0;height:0;}
.sat-dot{position:absolute;border-radius:50%;width:8px;height:8px;margin:-4px 0 0 -4px;}

/* ── Earth wrapper (float animation) ── */
.earth-wrap{
  position:absolute;top:38%;left:44%;
  transform:translate(-50%,-50%);
  animation:floatY 7s ease-in-out infinite;
}

/* ── Atmosphere glow ── */
.atmos{
  position:absolute;
  inset:-22px;border-radius:50%;
  background:radial-gradient(circle at center,
    transparent 42%,
    rgba(30,110,200,.07) 50%,
    rgba(52,152,219,.22) 60%,
    rgba(100,181,246,.1) 72%,
    transparent 85%);
  animation:atmosPulse 4.5s ease-in-out infinite;
}

/* ── Earth sphere ── */
.earth{
  width:200px;height:200px;border-radius:50%;
  position:relative;overflow:hidden;
  background:
    radial-gradient(circle at 33% 28%,rgba(255,255,255,.07) 0%,transparent 38%),
    radial-gradient(circle at 65% 72%,rgba(0,0,0,.22) 0%,transparent 45%),
    linear-gradient(140deg,#1a7bc4 0%,#1660a0 25%,#0d3e72 55%,#061c33 100%);
  box-shadow:
    inset -24px -18px 60px rgba(0,0,0,.65),
    inset  10px  10px 25px rgba(255,255,255,.04),
    0 0 70px rgba(26,107,196,.35),
    0 0 140px rgba(20,80,180,.12);
}

/* Ocean shimmer */
.earth::after{
  content:'';position:absolute;inset:0;border-radius:50%;
  background:
    radial-gradient(ellipse at 28% 22%,rgba(80,180,255,.14) 0%,transparent 38%),
    radial-gradient(ellipse at 68% 68%,rgba(0,40,110,.18) 0%,transparent 40%);
}

/* ── Grid lines ── */
.grid-svg{position:absolute;inset:0;width:100%;height:100%;opacity:.15;
  animation:gridDrift 12s ease-in-out infinite alternate;}

/* ── Continents SVG ── */
.cont-svg{position:absolute;inset:0;width:100%;height:100%;
  animation:contDrift 16s ease-in-out infinite alternate;}

/* ── Polar caps ── */
.polar-n{
  position:absolute;top:-4%;left:14%;width:72%;height:28%;border-radius:50%;
  background:radial-gradient(ellipse at center,rgba(220,240,255,.55) 0%,rgba(200,230,255,.2) 52%,transparent 75%);
}
.polar-s{
  position:absolute;bottom:-4%;left:20%;width:60%;height:24%;border-radius:50%;
  background:radial-gradient(ellipse at center,rgba(220,240,255,.45) 0%,rgba(200,230,255,.15) 52%,transparent 75%);
}

/* ── Specular highlight ── */
.spec{
  position:absolute;top:7%;left:13%;width:42%;height:36%;border-radius:50%;
  background:radial-gradient(ellipse at 40% 30%,rgba(255,255,255,.14) 0%,transparent 70%);
  pointer-events:none;
}

/* ── Clouds ── */
.cloud{position:absolute;background:rgba(255,255,255,.14);border-radius:50px;filter:blur(2px);}

/* ── Data particles ── */
.dp{position:absolute;width:5px;height:5px;border-radius:50%;animation:dataRise 3.5s ease-out infinite;}

/* ── Status dot ── */
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(46,204,113,.5)}50%{box-shadow:0 0 0 6px rgba(46,204,113,0)}}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  background:#2ecc71;box-shadow:0 0 8px #2ecc71;animation:pulse 2s ease-in-out infinite;
  margin-right:6px;vertical-align:middle;}

/* ── Bottom bar ── */
.info-bar{
  position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  color:rgba(133,193,233,.85);font:500 11px/1 'Microsoft YaHei','Courier New',sans-serif;
  letter-spacing:2px;white-space:nowrap;
}
</style>
</head>
<body>
<div class="scene" id="scene">
  <div class="scanline"></div>

  <!-- HUD corners -->
  <div class="corner tl"></div><div class="corner tr"></div>
  <div class="corner bl"></div><div class="corner br"></div>

  <!-- HUD labels -->
  <div class="hud" style="top:19px;left:40px;">SYS:ONLINE</div>
  <div class="hud" style="top:19px;right:40px;text-align:right;" id="lonLabel">LON:+000°</div>
  <div class="hud" style="bottom:20px;left:40px;"><span class="status-dot"></span>READY</div>
  <div class="hud" style="bottom:20px;right:40px;text-align:right;">ML ENGINE v1.0</div>

  <!-- Stars container -->
  <div id="stars"></div>

  <!-- Orbit anchor (centered) -->
  <div class="orbit-anchor">
    <!-- Orbit rings -->
    <div class="ring" style="width:300px;height:300px;margin:-150px 0 0 -150px;border:1.2px solid rgba(52,152,219,.18);"></div>
    <div class="ring" style="width:350px;height:350px;margin:-175px 0 0 -175px;border:1px dashed rgba(52,152,219,.09);transform:translate(-50%,-50%) rotateX(62deg) rotateZ(28deg);"></div>

    <!-- Satellites -->
    <div class="sat-track" style="animation:orbit1 10s linear infinite;">
      <div class="sat-dot" style="background:#f39c12;box-shadow:0 0 10px #f39c12,0 0 22px rgba(243,156,18,.35);"></div>
    </div>
    <div class="sat-track" style="animation:orbit2 15s linear infinite;">
      <div class="sat-dot" style="background:#00d4ff;box-shadow:0 0 10px #00d4ff,0 0 22px rgba(0,212,255,.35);"></div>
    </div>
    <div class="sat-track" style="animation:orbit3 11s linear infinite;">
      <div class="sat-dot" style="background:#a855f7;box-shadow:0 0 8px #a855f7,0 0 18px rgba(168,85,247,.3);"></div>
    </div>
  </div>

  <!-- Earth -->
  <div class="earth-wrap">
    <div style="position:relative;width:200px;height:200px;">
      <!-- Atmosphere -->
      <div class="atmos"></div>

      <!-- Sphere -->
      <div class="earth">
        <!-- Grid lines -->
        <svg class="grid-svg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
          <ellipse cx="100" cy="100" rx="94" ry="13" fill="none" stroke="rgba(100,200,255,.9)" stroke-width=".6"/>
          <ellipse cx="100" cy="68"  rx="84" ry="9"  fill="none" stroke="rgba(100,200,255,.6)" stroke-width=".45"/>
          <ellipse cx="100" cy="132" rx="84" ry="9"  fill="none" stroke="rgba(100,200,255,.6)" stroke-width=".45"/>
          <ellipse cx="100" cy="40"  rx="57" ry="5"  fill="none" stroke="rgba(100,200,255,.35)" stroke-width=".35"/>
          <ellipse cx="100" cy="160" rx="57" ry="5"  fill="none" stroke="rgba(100,200,255,.35)" stroke-width=".35"/>
          <line x1="100" y1="6" x2="100" y2="194" stroke="rgba(100,200,255,.4)" stroke-width=".4"/>
          <path d="M100,6 Q146,100 100,194" fill="none" stroke="rgba(100,200,255,.28)" stroke-width=".4"/>
          <path d="M100,6 Q54,100 100,194"  fill="none" stroke="rgba(100,200,255,.28)" stroke-width=".4"/>
        </svg>

        <!-- Continents (SVG paths, simplified but geographically suggestive) -->
        <svg class="cont-svg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
          <!-- Africa -->
          <path d="M98,52 C106,49 120,53 124,65 C128,77 126,92 123,104
                   C120,118 115,132 110,143 C106,152 101,157 98,155
                   C93,153 87,142 84,130 C80,115 78,98 80,84
                   C83,70 90,57 98,52 Z"
            fill="rgba(46,180,100,.6)" stroke="rgba(80,220,130,.35)" stroke-width=".7"/>
          <!-- Madagascar -->
          <path d="M128,118 C131,115 135,117 136,122 C137,127 134,133 131,134
                   C128,135 125,131 125,126 C125,121 126,120 128,118 Z"
            fill="rgba(46,180,100,.5)" stroke="rgba(80,220,130,.3)" stroke-width=".5"/>
          <!-- Europe -->
          <path d="M72,26 C80,22 94,24 98,30 C102,37 99,46 95,52
                   C91,56 84,57 79,54 C73,50 69,40 72,26 Z"
            fill="rgba(46,180,100,.55)" stroke="rgba(80,220,130,.3)" stroke-width=".6"/>
          <!-- Asia (large) -->
          <path d="M108,16 C120,13 146,16 158,26 C170,37 173,54 170,67
                   C167,79 157,88 146,92 C136,96 124,93 116,86
                   C108,78 104,64 104,50 C104,36 106,22 108,16 Z"
            fill="rgba(46,180,100,.55)" stroke="rgba(80,220,130,.3)" stroke-width=".6"/>
          <!-- Indian subcontinent bump -->
          <path d="M130,90 C135,88 142,92 143,100 C144,108 138,114 133,114
                   C128,114 124,108 125,100 C126,94 128,92 130,90 Z"
            fill="rgba(46,180,100,.52)" stroke="rgba(80,220,130,.25)" stroke-width=".5"/>
          <!-- North America -->
          <path d="M22,16 C32,12 50,15 58,24 C67,33 70,48 68,60
                   C66,71 58,80 49,84 C41,88 33,85 27,77
                   C20,68 17,53 18,40 C19,30 20,20 22,16 Z"
            fill="rgba(46,180,100,.55)" stroke="rgba(80,220,130,.3)" stroke-width=".6"/>
          <!-- South America -->
          <path d="M44,90 C52,86 62,88 65,97 C69,108 68,122 65,134
                   C62,146 56,157 51,162 C47,167 44,167 42,163
                   C38,157 36,144 36,130 C36,116 38,100 44,90 Z"
            fill="rgba(46,180,100,.55)" stroke="rgba(80,220,130,.3)" stroke-width=".6"/>
          <!-- Australia -->
          <path d="M152,118 C162,114 174,118 176,128 C178,138 171,149 162,151
                   C154,153 145,147 143,137 C141,127 144,121 152,118 Z"
            fill="rgba(46,180,100,.55)" stroke="rgba(80,220,130,.3)" stroke-width=".6"/>
          <!-- Greenland -->
          <path d="M36,10 C42,7 52,9 54,16 C56,22 51,28 45,29
                   C39,30 33,24 34,17 C34,14 35,12 36,10 Z"
            fill="rgba(200,235,255,.35)" stroke="rgba(200,235,255,.2)" stroke-width=".4"/>
        </svg>

        <!-- Polar caps -->
        <div class="polar-n"></div>
        <div class="polar-s"></div>

        <!-- Clouds -->
        <div class="cloud" style="top:21%;left:36%;width:52px;height:15px;animation:cld1 10s ease-in-out infinite 1s;"></div>
        <div class="cloud" style="top:46%;left:22%;width:40px;height:12px;animation:cld2 11s ease-in-out infinite 2s;opacity:.11;"></div>
        <div class="cloud" style="top:33%;left:58%;width:36px;height:11px;animation:cld3 9s ease-in-out infinite 3s;"></div>
        <div class="cloud" style="top:65%;left:44%;width:45px;height:13px;animation:cld4 12s ease-in-out infinite .5s;opacity:.1;"></div>

        <!-- Specular highlight -->
        <div class="spec"></div>
      </div>
    </div>
  </div>

  <!-- Data particles rising from globe -->
  <div class="dp" style="left:48%;top:36%;background:#00d4ff;box-shadow:0 0 6px #00d4ff;animation-delay:0s;"></div>
  <div class="dp" style="left:52%;top:39%;background:#2ecc71;box-shadow:0 0 6px #2ecc71;animation-delay:1.4s;animation-duration:4s;"></div>
  <div class="dp" style="left:46%;top:41%;background:#f39c12;box-shadow:0 0 6px #f39c12;animation-delay:2.7s;animation-duration:3s;"></div>
  <div class="dp" style="left:50%;top:37%;background:#a855f7;box-shadow:0 0 6px #a855f7;animation-delay:.8s;animation-duration:4.5s;"></div>

  <!-- Bottom info bar -->
  <div class="info-bar">
    <span class="status-dot"></span>请在左侧配置数据路径和模型参数后开始训练
  </div>
</div>

<script>
// Generate starfield
const scene = document.getElementById('scene');
const sc = document.getElementById('stars');
for(let i=0;i<140;i++){
  const s=document.createElement('div');
  const sz=Math.random()*2.2+0.4;
  s.className='star';
  s.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;top:${Math.random()*100}%;
    opacity:${Math.random()*.7+.1};animation:twinkle ${Math.random()*3+1.5}s ease-in-out ${Math.random()*4}s infinite;`;
  sc.appendChild(s);
}
// Animate LON display
let lon=0;
setInterval(()=>{
  lon=(lon+.8)%360;
  const v=Math.round(lon>180?lon-360:lon);
  const el=document.getElementById('lonLabel');
  if(el) el.textContent=`LON:${v>=0?'+':''}${String(Math.abs(v)).padStart(3,'0')}°`;
},80);
</script>
</body>
</html>
        """, height=415)

    with st.sidebar:

        # ── 数据模式选择 ──
        st.header("\U0001f4c5 数据模式")
        time_mode = st.selectbox(
            "数据时间单位",
            ["yearly", "monthly"],
            index=0 if st.session_state.time_mode == "yearly" else 1,
            format_func=lambda x: "\U0001f4c6 年度数据 (如 2000.tif)" if x == "yearly" 
                                   else "\U0001f4c5 月度数据 (如 2000/01.tif 或 2000-01.tif)",
            key="time_mode_sel",
        )
        st.session_state.time_mode = time_mode

        # ── 预测模式选择 ──
        prediction_mode = st.radio(
            "预测模式",
            ["simulation", "extrapolation"],
            index=0 if st.session_state.prediction_mode == "simulation" else 1,
            format_func=lambda x: "🔬 模拟模式 (预测期有X数据)" if x == "simulation"
                                   else "🔮 外推预测 (预测期无X数据，使用末年X)",
            key="pred_mode_sel",
        )
        st.session_state.prediction_mode = prediction_mode

        # 模式说明
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            with st.popover("🔬 模拟模式说明", use_container_width=True):
                st.markdown("""
### 模拟模式 (Simulation)

**适用场景：** 预测年份的自变量 X 数据已存在。

**工作原理：**
1. 使用训练年份(如2014-2022)的 Y 和 X 数据训练模型
2. 使用预测年份(如2023)的 X 数据驱动模型预测
3. 将预测值与预测年份的真实 Y 做对比评估

**数据要求：**
- Y 文件夹: 必须包含训练年和预测年所有数据
- X 文件夹: 必须包含训练年和预测年所有数据
- 例: Y有2014-2023, X也必须有2014-2023

**典型场景：**
- 你有完整的X变量观测数据覆盖到2023
- 想验证模型在"已知X、未知Y"情况下的表现
                """)
        with col_h2:
            with st.popover("🔮 外推预测说明", use_container_width=True):
                st.markdown("""
### 外推预测 (Extrapolation)

**适用场景：** 预测年份的自变量 X 数据不存在。

**工作原理：**
1. 使用训练年份的 Y 和 X 数据训练模型
2. 预测时自动使用训练末年(如2022)的 X 值
3. 即使没有未来X数据也能产出预测

**数据要求：**
- Y 文件夹: 必须包含训练年和预测年数据
- X 文件夹: 只需包含训练年数据即可
- 例: X只有2014-2022, 预测2023年仍可运行

**典型场景：**
- X变量数据只更新到2022年，需要预测2023年
- 对未来进行真正的"预报"，不依赖未来X
- 假设X变量滞后/不变(持久性假设)
                """)

        # ── 数据路径配置 ──
        st.header("\U0001f4c2 数据路径")
        y_folder = st.text_input(
            "Y 因变量文件夹路径",
            value=st.session_state.y_folder,
            placeholder="输入Y变量TIFF文件夹路径"
        )
        x_folders_str = st.text_area(
            "X 自变量文件夹 (每行一个)",
            value=st.session_state.x_folders_str,
            placeholder="每行输入一个X变量TIFF文件夹路径",
            height=80,
        )
        output_dir = st.text_input(
            "输出文件夹",
            value=st.session_state.output_dir,
            placeholder="默认: ./outputs (留空使用默认)",
            help="所有结果(预测图/图表/表格/报告)将保存到此目录。留空则自动使用当前目录下的outputs文件夹。"
        )

        # 自动检测年份范围
        if y_folder and y_folder != st.session_state.y_folder:
            st.session_state.y_folder = y_folder
            ys, ye = detect_time_range(y_folder, time_mode)
            st.session_state.data_year_start = ys
            st.session_state.data_year_end = ye

        st.divider()
        st.header("\u23f1 时间范围配置")

        ds = st.session_state.data_year_start
        de = st.session_state.data_year_end
        auto_ts = ds if ds is not None else 2000
        auto_te = (de - 1) if de is not None and ds is not None and de > ds else 2010
        auto_ps = de if de is not None else 2011
        auto_pe = de if de is not None else 2011

        if ds is not None and de is not None:
            unit = "\u4e2a\u6708" if time_mode == "monthly" else "\u5e74"
            st.caption(f"\U0001f4c5 \u81ea\u52a8\u68c0\u6d4b: {ds}-{de} (共 {de-ds+1} {unit})")

        col1, col2 = st.columns(2)
        with col1:
            label_suffix = "\u5e74" if time_mode == "yearly" else "\u5e74\u4efd"
            train_start = st.number_input(
                f"\u8bad\u7ec3\u8d77\u59cb{label_suffix}",
                value=auto_ts, min_value=1980, max_value=2100,
                help="默认: 数据第一年"
            )
            train_end = st.number_input(
                f"\u8bad\u7ec3\u7ed3\u675f{label_suffix}",
                value=auto_te, min_value=1980, max_value=2100,
                help="默认: 数据最后一年 - 1"
            )
        with col2:
            predict_start = st.number_input(
                f"\u9884\u6d4b\u8d77\u59cb{label_suffix}",
                value=auto_ps, min_value=1980, max_value=2100,
                help="默认: 数据最后一年"
            )
            predict_end = st.number_input(
                f"\u9884\u6d4b\u7ed3\u675f{label_suffix}",
                value=auto_pe, min_value=1980, max_value=2100,
                help="默认: 数据最后一年"
            )

        st.divider()
        st.header("\U0001f916 模型选择")
        st.caption("\u26a0\ufe0f 所有模型默认不勾选，请手动选择需要的模型")

        model_configs = {}
        model_param_defaults = {
            "ols": {},
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 0.01},
            "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
            "rf": {"n_estimators": 100, "max_depth": 10,
                   "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
            "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                        "subsample": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
            "lightgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                         "num_leaves": 31, "subsample": 1.0,
                         "reg_lambda": 0.0, "reg_alpha": 0.0, "min_child_samples": 20},
            "svr": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
            "mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 500,
                    "activation": "relu", "alpha": 0.0001},
            "knn": {"n_neighbors": 5, "weights": "uniform"},
            "gbr": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
                    "subsample": 1.0, "min_samples_split": 2, "min_samples_leaf": 1},
            "extratrees": {"n_estimators": 100, "max_depth": 10,
                           "min_samples_split": 2, "max_features": "sqrt"},
        }

        with st.expander("\U0001f4c8 线性模型", expanded=True):
            model_configs["ols"] = st.checkbox("OLS (普通最小二乘)", value=False)
            model_configs["ridge"] = st.checkbox("Ridge (岭回归/L2)", value=False)
            if model_configs["ridge"]:
                model_param_defaults["ridge"]["alpha"] = st.number_input("Ridge alpha", 0.001, 100.0, 1.0, 0.1, key="ridge_a")
            model_configs["lasso"] = st.checkbox("Lasso (L1正则化)", value=False)
            if model_configs["lasso"]:
                model_param_defaults["lasso"]["alpha"] = st.number_input("Lasso alpha", 0.0001, 10.0, 0.01, 0.001, key="lasso_a")
            model_configs["elasticnet"] = st.checkbox("ElasticNet (L1+L2)", value=False)
            if model_configs["elasticnet"]:
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["elasticnet"]["alpha"] = st.number_input("EN alpha", 0.0001, 10.0, 0.01, 0.001, key="en_a")
                with c2:
                    model_param_defaults["elasticnet"]["l1_ratio"] = st.number_input("EN l1_ratio", 0.0, 1.0, 0.5, 0.05, key="en_l1r")

        with st.expander("\U0001f332 树模型", expanded=True):
            model_configs["rf"] = st.checkbox("Random Forest (随机森林)", value=False)
            if model_configs["rf"]:
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["rf"]["n_estimators"] = st.number_input("RF 树数量", 10, 500, 100, 10, key="rf_n")
                with c2:
                    model_param_defaults["rf"]["max_depth"] = st.number_input("RF 最大深度", 2, 50, 10, 1, key="rf_d")
                c3, c4 = st.columns(2)
                with c3:
                    model_param_defaults["rf"]["min_samples_leaf"] = st.number_input("RF 叶节点最小样本", 1, 20, 1, 1, key="rf_msl")
                with c4:
                    model_param_defaults["rf"]["max_features"] = st.selectbox("RF max_features", ["sqrt", "log2", "0.5", "0.8", "1.0"], 0, key="rf_mf")
            model_configs["xgboost"] = st.checkbox("XGBoost", value=False)
            if model_configs["xgboost"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    model_param_defaults["xgboost"]["n_estimators"] = st.number_input("XGB 树数量", 10, 500, 100, 10, key="xgb_n")
                with c2:
                    model_param_defaults["xgboost"]["max_depth"] = st.number_input("XGB 最大深度", 2, 20, 6, 1, key="xgb_d")
                with c3:
                    model_param_defaults["xgboost"]["learning_rate"] = st.number_input("XGB 学习率", 0.001, 1.0, 0.1, 0.01, key="xgb_lr")
                c4, c5 = st.columns(2)
                with c4:
                    model_param_defaults["xgboost"]["subsample"] = st.slider("XGB 子采样比例", 0.1, 1.0, 1.0, 0.1, key="xgb_ss")
                with c5:
                    model_param_defaults["xgboost"]["reg_lambda"] = st.number_input("XGB L2正则化", 0.0, 100.0, 1.0, 0.1, key="xgb_l2")
            model_configs["lightgbm"] = st.checkbox("LightGBM", value=False)
            if model_configs["lightgbm"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    model_param_defaults["lightgbm"]["n_estimators"] = st.number_input("LGB 树数量", 10, 500, 100, 10, key="lgb_n")
                with c2:
                    model_param_defaults["lightgbm"]["max_depth"] = st.number_input("LGB 最大深度", 2, 20, 6, 1, key="lgb_d")
                with c3:
                    model_param_defaults["lightgbm"]["learning_rate"] = st.number_input("LGB 学习率", 0.001, 1.0, 0.1, 0.01, key="lgb_lr")
            model_configs["extratrees"] = st.checkbox("Extra Trees (极端随机树)", value=False)
            if model_configs["extratrees"]:
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["extratrees"]["n_estimators"] = st.number_input("ET 树数量", 10, 500, 100, 10, key="et_n")
                with c2:
                    model_param_defaults["extratrees"]["max_depth"] = st.number_input("ET 最大深度", 2, 50, 10, 1, key="et_d")
            model_configs["gbr"] = st.checkbox("Gradient Boosting (梯度提升)", value=False)
            if model_configs["gbr"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    model_param_defaults["gbr"]["n_estimators"] = st.number_input("GBR 树数量", 10, 500, 100, 10, key="gbr_n")
                with c2:
                    model_param_defaults["gbr"]["max_depth"] = st.number_input("GBR 最大深度", 2, 20, 3, 1, key="gbr_d")
                with c3:
                    model_param_defaults["gbr"]["learning_rate"] = st.number_input("GBR 学习率", 0.001, 1.0, 0.1, 0.01, key="gbr_lr")

        with st.expander("\U0001f52c 其他模型", expanded=True):
            model_configs["svr"] = st.checkbox("SVR (支持向量回归)", value=False)
            if model_configs["svr"]:
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["svr"]["kernel"] = st.selectbox("核函数", ["rbf", "linear", "poly"], 0, key="svr_k")
                with c2:
                    model_param_defaults["svr"]["C"] = st.number_input("SVR C", 0.01, 100.0, 1.0, 0.1, key="svr_c")
                c3, c4 = st.columns(2)
                with c3:
                    model_param_defaults["svr"]["epsilon"] = st.number_input("SVR epsilon", 0.001, 1.0, 0.1, 0.01, key="svr_e")
                with c4:
                    model_param_defaults["svr"]["gamma"] = st.selectbox("SVR gamma", ["scale", "auto", 0.01, 0.1, 1.0], 0, key="svr_g")
            model_configs["mlp"] = st.checkbox("MLP (神经网络)", value=False)
            if model_configs["mlp"]:
                model_param_defaults["mlp"]["hidden_layer_sizes"] = eval(st.text_input("隐藏层结构", "[64, 32]", key="mlp_h"))
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["mlp"]["max_iter"] = st.number_input("最大迭代次数", 100, 2000, 500, 50, key="mlp_i")
                with c2:
                    model_param_defaults["mlp"]["activation"] = st.selectbox("激活函数", ["relu", "tanh", "logistic"], 0, key="mlp_act")
                model_param_defaults["mlp"]["alpha"] = st.number_input("L2正则化 alpha", 0.00001, 0.1, 0.0001, 0.0001, format="%.5f", key="mlp_a")
            model_configs["knn"] = st.checkbox("KNN (K近邻)", value=False)
            if model_configs["knn"]:
                c1, c2 = st.columns(2)
                with c1:
                    model_param_defaults["knn"]["n_neighbors"] = st.number_input("K值", 1, 50, 5, 1, key="knn_k")
                with c2:
                    model_param_defaults["knn"]["weights"] = st.selectbox("权重模式", ["uniform", "distance"], 0, key="knn_w")

        st.divider()
        st.header("\u2699\ufe0f 高级参数")
        scaling = st.selectbox(
            "数据标准化方法", ["standard", "minmax", "none"],
            index=0,
            format_func=lambda x: {"standard": "StandardScaler (推荐)",
                                    "minmax": "MinMaxScaler (0-1归一化)",
                                    "none": "不标准化"}[x]
        )
        n_jobs = st.slider("CPU核心数 (-1=全部)", -1, 16, -1)
        valid_threshold = st.slider(
            "有效像素比例阈值", 0.1, 1.0, 0.5, 0.05,
            help="像素在所有时间步中至少有此比例的值为非空(NaN)才被纳入训练"
        )
        cv_folds = st.slider("交叉验证折数 (0=关闭)", 0, 10, 0)
        ensemble_enabled = st.checkbox("启用模型集成预测", value=True)

    # 解析X路径
    x_folders = [p.strip() for p in x_folders_str.split("\n") if p.strip()]

    # 构建模型配置列表
    models = []
    for mname, enabled in model_configs.items():
        models.append(ModelConfig(name=mname, enabled=enabled,
                                  params=model_param_defaults.get(mname, {})))

    config = AppConfig(
        y_folder=y_folder, x_folders=x_folders, output_dir=output_dir,
        train_start=train_start, train_end=train_end,
        predict_start=predict_start, predict_end=predict_end,
        time_unit=time_mode,
        scaling=scaling, n_jobs=n_jobs,
        valid_pct_threshold=valid_threshold,
        models=models, cv_folds=cv_folds,
    )
    config.prediction_mode = prediction_mode

    # 数据摘要
    if y_folder and os.path.isdir(y_folder):
        from src.data_loader import discover_tiff_files
        yf, yu = discover_tiff_files(y_folder)
        if yf:
            yrs = sorted(yf.keys())
            mode_label = "月度" if time_mode == "monthly" else "年度"
            st.info(f"\U0001f4c1 Y变量: {len(yf)} 个TIFF文件 ({mode_label}), {yrs[0]} \u2014 {yrs[-1]}")
        for i, xf in enumerate(x_folders):
            if os.path.isdir(xf):
                xfd, xu = discover_tiff_files(xf)
                st.info(f"\U0001f4c1 X{i+1}变量: {len(xfd)} 个TIFF文件")

    st.session_state.y_folder = y_folder
    st.session_state.x_folders_str = x_folders_str
    st.session_state.output_dir = output_dir

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        run_clicked = st.button("\U0001f680 开始训练", type="primary", use_container_width=True)
    with col_btn2:
        save_config_clicked = st.button("\U0001f4be 保存配置", use_container_width=True)

    if save_config_clicked:
        config_dict = {
            "y_folder": y_folder, "x_folders": x_folders,
            "output_dir": output_dir, "time_mode": time_mode,
            "prediction_mode": prediction_mode,
            "train_start": train_start, "train_end": train_end,
            "predict_start": predict_start, "predict_end": predict_end,
            "scaling": scaling, "n_jobs": n_jobs,
            "valid_threshold": valid_threshold,
            "models": {k: {"enabled": v, "params": model_param_defaults.get(k, {})}
                       for k, v in model_configs.items()},
        }
        st.session_state.config_history.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config_dict,
        })
        st.success("\u2705 配置已保存到历史记录")

    if run_clicked:
        errors = config.validate()
        if errors:
            for e in errors:
                st.error(f"\u274c {e}")
        elif not y_folder:
            st.error("\u274c 请输入Y变量文件夹路径")
        elif not x_folders:
            st.error("\u274c 请输入至少一个X变量文件夹路径")
        elif not config.get_enabled_models():
            st.error("\u274c 请至少勾选一个模型")
        else:
            st.session_state.results = None
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 显示 CPU 信息
            try:
                import psutil
                cpu_model = "Unknown"
                if hasattr(psutil, "cpu_freq"):
                    cpu_freq = psutil.cpu_freq()
                    freq_str = f"@{cpu_freq.max:.0f}MHz" if cpu_freq else ""
                cpu_count = psutil.cpu_count(logical=True)
                cpu_physical = psutil.cpu_count(logical=False)
                cpu_info = st.empty()
                cpu_usage_placeholder = st.empty()
            except ImportError:
                cpu_info = None
                cpu_usage_placeholder = None

            def progress_callback(msg, pct):
                status_text.text(msg)
                if pct > 0:
                    progress_bar.progress(pct / 100)
                if cpu_usage_placeholder is not None:
                    try:
                        cpu_pct = psutil.cpu_percent(interval=0.1)
                        cpu_usage_placeholder.text(f"CPU使用率: {cpu_pct:.1f}% | 核心数: {cpu_physical}核/{cpu_count}线程")
                    except Exception:
                        pass

            pipeline = Pipeline(config, progress_callback=progress_callback)
            try:
                with st.spinner("正在运行模型训练..."):
                    results = pipeline.run()
                st.session_state.results = results
                st.session_state.pipeline = pipeline
                progress_bar.progress(100)
                status_text.text("\u2705 训练完成!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
            except Exception as e:
                st.error(f"\u274c 运行失败: {e}")
                import traceback
                st.code(traceback.format_exc())

    results = st.session_state.results
    if results:
        st.divider()
        st.header("\U0001f4ca 训练结果")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "\U0001f4cb 性能总览", "\U0001f4c8 可视化图表", "\U0001f5fa 空间分析",
            "\U0001f4dd 详细指标", "\U0001f4c4 评估报告", "\U0001f4be 导出结果"
        ])

        with tab1:
            st.subheader("模型性能对比")
            if results["summary_df"] is not None:
                df = results["summary_df"].copy()
                has_rmse = "mean_rmse" in df.columns and df["mean_rmse"].notna().any()
                has_r2 = "mean_r2" in df.columns and df["mean_r2"].notna().any()
                if has_rmse:
                    fmt = {"mean_r2": "{:.4f}", "mean_rmse": "{:.4f}",
                           "mean_mae": "{:.4f}", "success_rate": "{:.2%}"} if has_r2 else {
                           "mean_rmse": "{:.4f}", "mean_mae": "{:.4f}", "success_rate": "{:.2%}"}
                    styled = df.style.format(fmt)
                    if has_r2:
                        styled = styled.highlight_max(subset=["mean_r2"], color="#d4edda")
                    styled = styled.highlight_min(subset=["mean_rmse"], color="#d4edda")
                    st.dataframe(styled, use_container_width=True)
                    best_idx = df["mean_rmse"].idxmin()
                    bn, be = df.loc[best_idx, "model"], df.loc[best_idx, "mean_rmse"]
                    if has_r2:
                        st.success(f"\U0001f3c6 最佳: **{bn}** (R\u00b2={df.loc[best_idx,'mean_r2']:.4f}, RMSE={be:.4f})")
                    else:
                        st.success(f"\U0001f3c6 最佳: **{bn}** (RMSE={be:.4f}, R\u00b2需\u22652个预测年)")
                else:
                    st.warning("\u26a0\ufe0f 所有模型均未产出有效评估指标，请检查数据和时间配置")
                    st.dataframe(df, use_container_width=True)

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("有效像素", f"{results['n_valid_pixels']:,}")
            with col_m2:
                st.metric("总耗时", f"{results['total_time']:.1f} 秒")
            with col_m3:
                st.metric("训练时段", f"{train_start}\u2014{train_end}")
            with col_m4:
                st.metric("预测时段", f"{predict_start}\u2014{predict_end}")
            with col_m5:
                st.metric("模型数", len(results.get("all_metrics", {})))

        with tab2:
            st.subheader("可视化图表")
            if results["chart_paths"]:
                # Group by chart type
                scatter_charts = [cp for cp in results["chart_paths"] if "散点图" in os.path.basename(cp)]
                residual_charts = [cp for cp in results["chart_paths"] if "残差分布" in os.path.basename(cp)]
                spatial_charts = [cp for cp in results["chart_paths"] if "残差空间" in os.path.basename(cp)]
                other_charts = [cp for cp in results["chart_paths"]
                                if cp not in scatter_charts + residual_charts + spatial_charts]

                if scatter_charts:
                    st.markdown("#### 预测值 vs 观测值散点图")
                    cols = st.columns(min(len(scatter_charts), 3))
                    for i, cp in enumerate(scatter_charts):
                        if os.path.exists(cp):
                            cols[i % 3].image(cp, caption=os.path.basename(cp)[:-4], use_container_width=True)

                if residual_charts:
                    st.markdown("#### 残差分布直方图")
                    cols = st.columns(min(len(residual_charts), 3))
                    for i, cp in enumerate(residual_charts):
                        if os.path.exists(cp):
                            cols[i % 3].image(cp, caption=os.path.basename(cp)[:-4], use_container_width=True)

                if spatial_charts:
                    st.markdown("#### 残差空间分布")
                    cols = st.columns(min(len(spatial_charts), 3))
                    for i, cp in enumerate(spatial_charts):
                        if os.path.exists(cp):
                            cols[i % 3].image(cp, caption=os.path.basename(cp)[:-4], use_container_width=True)

                if other_charts:
                    st.markdown("#### 综合对比图表")
                    for cp in other_charts:
                        if os.path.exists(cp) and cp.endswith(".png"):
                            st.image(cp, caption=os.path.basename(cp), use_container_width=True)
            else:
                st.info("暂无图表")

        with tab3:
            st.subheader("\U0001f5fa 空间分析")
            from src.spatial_analysis import plot_performance_map, compute_performance_map
            if results.get("all_metrics") and results.get("pixel_models"):
                model_sel = st.selectbox("选择模型", list(results["all_metrics"].keys()))
                metric_sel = st.selectbox("选择指标", ["r2", "rmse", "mae"], key="sp_met")
                pm = compute_performance_map(
                    results["all_metrics"], model_sel, metric_sel,
                    results.get("valid_indices", []),
                    results.get("y_profile", {"height": 100, "width": 100})
                )
                if pm is not None:
                    fig_path = os.path.join(output_dir, f"perf_{model_sel}_{metric_sel}.png")
                    cmap = "RdYlGn" if metric_sel == "r2" else "RdYlGn_r"
                    plot_performance_map(pm, fig_path, title=f"{model_sel} - {metric_sel.upper()}", cmap=cmap)
                    st.image(fig_path, use_container_width=True)
            else:
                st.info("请先完成模型训练")

        with tab4:
            st.subheader("模型详细指标")
            if results["all_metrics"]:
                model_sel = st.selectbox("选择模型查看详情", list(results["all_metrics"].keys()))
                pm = results["all_metrics"].get(model_sel, [])
                if pm:
                    metrics_df = pd.DataFrame(pm)
                    st.dataframe(metrics_df.describe(), use_container_width=True)
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                    for ax, col in zip(axes, ["r2", "rmse", "mae"]):
                        vals = [m[col] for m in pm if not np.isnan(m.get(col, np.nan))]
                        if vals:
                            ax.hist(vals, bins=30, color="steelblue", edgecolor="white")
                        ax.set_title(col.upper())
                    fig.tight_layout()
                    st.pyplot(fig)
            if results.get("fi_results"):
                st.subheader("特征重要性 (树模型)")
                for mn, fi in results["fi_results"].items():
                    st.markdown(f"**{mn}** (基于 {fi['n_models']} 个像素)")
                    fi_df = pd.DataFrame({
                        "特征": [f"X{i+1}" for i in range(len(fi["mean"]))],
                        "重要性均值": fi["mean"], "标准差": fi["std"],
                    }).sort_values("重要性均值", ascending=False)
                    st.dataframe(fi_df, use_container_width=True)

        with tab5:
            st.subheader("HTML 评估报告")
            report_path = results.get("report_path", "")
            if report_path and os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.download_button("\u2b07\ufe0f 下载 HTML 报告", f.read(),
                                       "evaluation_report.html", "text/html")
            else:
                st.info("报告尚未生成")
            if st.button("\U0001f4c4 生成 PDF 报告", key="pdf_btn"):
                with st.spinner("正在生成 PDF..."):
                    from src.pdf_report import generate_pdf_report
                    pdf_path = os.path.join(output_dir, "evaluation_report.pdf")
                    try:
                        generate_pdf_report(
                            results["summary_df"], results["all_metrics"],
                            {"train_period": f"{train_start}-{train_end}",
                             "predict_period": f"{predict_start}-{predict_end}"},
                            {"n_valid_pixels": results["n_valid_pixels"],
                             "total_time": f"{results['total_time']:.1f}s"},
                            results["chart_paths"], pdf_path
                        )
                        st.success(f"PDF 已保存: {pdf_path}")
                        with open(pdf_path, "rb") as f:
                            st.download_button("\u2b07\ufe0f 下载 PDF", f,
                                               "evaluation_report.pdf", "application/pdf")
                    except Exception as e:
                        st.error(f"PDF 生成失败: {e}")

        with tab6:
            st.subheader("导出结果")
            csv_path = results.get("csv_path", "")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    st.download_button("\u2b07\ufe0f 下载 CSV 指标表", f.read(),
                                       "metrics_summary.csv", "text/csv")
            st.markdown("#### 导出配置")
            export_config = {
                "y_folder": y_folder, "x_folders": x_folders,
                "output_dir": output_dir, "time_mode": time_mode,
                "train_period": [train_start, train_end],
                "predict_period": [predict_start, predict_end],
                "models": {k: v for k, v in model_configs.items() if v},
            }
            st.download_button(
                "\u2b07\ufe0f 导出 JSON 配置",
                json.dumps(export_config, indent=2, ensure_ascii=False),
                "config.json", "application/json"
            )

    if results and results.get("warnings"):
        with st.expander("\u26a0\ufe0f 警告信息"):
            for w in results["warnings"]:
                st.warning(w)

elif page == "\U0001f50d 数据探索":
    st.header("\U0001f50d 数据探索与质量检查")
    explore_y = st.text_input("Y 变量文件夹", value=st.session_state.y_folder, key="exp_y")
    explore_x = st.text_area("X 变量文件夹 (每行一个)", value=st.session_state.x_folders_str, key="exp_x", height=80)
    explore_x_list = [p.strip() for p in explore_x.split("\n") if p.strip()]

    if st.button("\U0001f50d 开始探索", type="primary"):
        if not explore_y or not os.path.isdir(explore_y):
            st.error("请输入有效的Y变量文件夹路径")
            st.session_state.explore_data = None
        else:
            from src.data_loader import discover_tiff_files
            from src.data_explorer import get_tiff_info, plot_histogram, plot_raster_preview, plot_time_series, compute_correlation_matrix, plot_correlation_heatmap
            with st.spinner("扫描文件中..."):
                y_files, y_unit = discover_tiff_files(explore_y)
                # 预加载全部元数据
                y_sorted = sorted(y_files.keys())
                y_info_list = [get_tiff_info(y_files[k]) for k in y_sorted]
                # X变量数据
                x_data = []
                for i, xf in enumerate(explore_x_list):
                    if os.path.isdir(xf):
                        xf_files, xf_unit = discover_tiff_files(xf)
                        x_sorted = sorted(xf_files.keys())
                        x_data.append({
                            "path": xf, "name": os.path.basename(xf),
                            "files": xf_files, "unit": xf_unit,
                            "sorted_keys": x_sorted,
                            "info_list": [get_tiff_info(xf_files[k]) for k in x_sorted]
                        })
                st.session_state.explore_data = {
                    "y_folder": explore_y,
                    "y_files": y_files,
                    "y_unit": y_unit,
                    "y_sorted": y_sorted,
                    "y_info_list": y_info_list,
                    "x_data": x_data,
                }

    # ========== 展示区（依赖 session_state，不受按钮重置影响） ==========
    if "explore_data" in st.session_state and st.session_state.explore_data:
        ed = st.session_state.explore_data
        y_files = ed["y_files"]
        y_sorted = ed["y_sorted"]
        y_info_list = ed["y_info_list"]
        explore_y = ed["y_folder"]

        from src.data_explorer import get_tiff_info, plot_histogram, plot_raster_preview, plot_time_series, compute_correlation_matrix, plot_correlation_heatmap

        st.success(f"Y变量: 发现 {len(y_files)} 个文件 ({ed['y_unit']})")

        if y_files:
            # --- 元数据全部展示 ---
            st.subheader(f"\U0001f4cb 全部文件元数据 ({len(y_sorted)} 个)")
            if y_info_list:
                st.dataframe(pd.DataFrame(y_info_list), use_container_width=True)

            # --- 年份选择器 ---
            st.divider()
            if len(y_sorted) > 1:
                selected_key = st.selectbox(
                    "\U0001f4c5 选择探索年份",
                    options=y_sorted,
                    index=0,
                    key="exp_y_year"
                )
            else:
                selected_key = y_sorted[0]

            selected_file = y_files[selected_key]
            info = get_tiff_info(selected_file)

            # --- 单年份统计 ---
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("分辨率", f"{info.get('width','?')} x {info.get('height','?')}")
            with c2:
                st.metric("最小值", f"{info.get('min',0):.4f}" if isinstance(info.get('min'), float) else "N/A")
            with c3:
                st.metric("最大值", f"{info.get('max',0):.4f}" if isinstance(info.get('max'), float) else "N/A")
                st.metric("均值", f"{info.get('mean',0):.4f}" if isinstance(info.get('mean'), float) else "N/A")
            with c4:
                st.metric("标准差", f"{info.get('std',0):.4f}" if isinstance(info.get('std'), float) else "N/A")
            st.text(f'坐标系 (CRS): {info.get("crs","N/A")}')

            # --- 预览图（带年份后缀避免覆盖） ---
            preview_path = os.path.join(explore_y, f"_preview_{selected_key}.png")
            plot_raster_preview(selected_file, preview_path)
            st.image(preview_path, caption=f"栅格预览 — {selected_key}", use_container_width=True)

            # --- 直方图 ---
            hist_path = os.path.join(explore_y, f"_histogram_{selected_key}.png")
            plot_histogram(selected_file, hist_path)
            st.image(hist_path, caption=f"数值分布直方图 — {selected_key}", use_container_width=True)

            # --- 多文件对比预览（不同年份并排） ---
            if len(y_sorted) > 1:
                st.divider()
                st.subheader("\U0001f4ca 多年份对比预览")
                cols_per_row = 4
                rows = (len(y_sorted) + cols_per_row - 1) // cols_per_row
                for row_idx in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        flat_idx = row_idx * cols_per_row + col_idx
                        if flat_idx < len(y_sorted):
                            k = y_sorted[flat_idx]
                            comp_path = os.path.join(explore_y, f"_preview_thumb_{k}.png")
                            plot_raster_preview(y_files[k], comp_path)
                            with cols[col_idx]:
                                st.image(comp_path, caption=k, use_container_width=True)

        if len(y_files) > 1:
            st.subheader("\U0001f4c8 时间序列趋势")
            ts_path = os.path.join(explore_y, "_timeseries.png")
            plot_time_series(y_files, ts_path)
            if os.path.exists(ts_path):
                st.image(ts_path, caption="时间序列 (采样像素)", use_container_width=True)
            corr, labels, _ = compute_correlation_matrix(y_files, sample_step=20)
            corr_path = os.path.join(explore_y, "_correlation.png")
            plot_correlation_heatmap(corr, labels, corr_path, title="Y变量时间自相关")
            st.image(corr_path, caption="时间自相关热力图", use_container_width=True)

        # --- X变量探索 ---
        x_data = ed.get("x_data", [])
        if x_data:
            st.divider()
            st.subheader("\U0001f4e5 X 自变量探索")
            for i, xd in enumerate(x_data):
                xf = xd["path"]
                x_files = xd["files"]
                x_sorted = xd["sorted_keys"]
                x_info_list = xd["info_list"]
                st.write(f"**X{i+1}**: `{xd['name']}` — {len(x_files)} 个文件 ({xd['unit']})")

                if x_files:
                    with st.expander(f"元数据 — {xd['name']}"):
                        st.dataframe(pd.DataFrame(x_info_list), use_container_width=True)

                    if len(x_sorted) > 1:
                        x_sel = st.selectbox(
                            f"选择 {xd['name']} 年份",
                            options=x_sorted,
                            index=0,
                            key=f"exp_x_{i}_year"
                        )
                    else:
                        x_sel = x_sorted[0]

                    x_file = x_files[x_sel]
                    xc1, xc2 = st.columns(2)
                    xp_path = os.path.join(xf, f"_preview_{x_sel}.png")
                    plot_raster_preview(x_file, xp_path)
                    with xc1:
                        st.image(xp_path, caption=f"{xd['name']} — {x_sel}", use_container_width=True)
                    xh_path = os.path.join(xf, f"_histogram_{x_sel}.png")
                    plot_histogram(x_file, xh_path)
                    with xc2:
                        st.image(xh_path, caption=f"{xd['name']} 直方图", use_container_width=True)

elif page == "\U0001f4ca 结果分析":
    st.header("\U0001f4ca 历史结果分析")
    if not st.session_state.results:
        st.info("暂无训练结果，请先返回首页执行训练")
    else:
        results = st.session_state.results
        if results.get("summary_df") is not None:
            df = results["summary_df"]
            if "mean_r2" in df.columns and df["mean_r2"].notna().any():
                st.subheader("模型对比雷达图")
                import matplotlib.pyplot as plt
                metrics = ["mean_r2", "mean_rmse", "mean_mae"]
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist() + [0]
                for _, row in df.iterrows():
                    vals = [row.get(m, 0) for m in metrics]
                    vals_norm = []
                    for i, m in enumerate(metrics):
                        col_vals = df[m].dropna()
                        if len(col_vals) > 1 and col_vals.max() > col_vals.min():
                            if m == "mean_r2":
                                v = (vals[i]-col_vals.min())/(col_vals.max()-col_vals.min())
                            else:
                                v = 1-(vals[i]-col_vals.min())/(col_vals.max()-col_vals.min())
                        else:
                            v = 0.5
                        vals_norm.append(max(0, min(1, v)))
                    ax.plot(angles, vals_norm + [vals_norm[0]], label=row["model"])
                    ax.fill(angles, vals_norm + [vals_norm[0]], alpha=0.1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.legend(bbox_to_anchor=(1.3, 1.1))
                st.pyplot(fig)
            else:
                st.info("暂无有效模型数据，无法生成雷达图")

# ============================================================
# PAGE: 模型复用 — 加载已保存模型进行预测
# ============================================================
elif page == "\U0001f4e6 模型复用":
    st.header("\U0001f4e6 模型复用 — 加载已保存模型进行预测")

    # List saved models
    from src.model_io import list_saved_models, load_model_package
    saved = list_saved_models()

    if not saved:
        st.info("未找到已保存的模型。请先在「首页与运行」完成训练，模型会自动保存到 outputs/models/ 目录。")
    else:
        st.success(f"找到 {len(saved)} 个已保存的模型包")

        for i, s in enumerate(saved):
            with st.expander(f"📦 {os.path.basename(s['dir'])} — {s['saved_at']}"):
                st.write(f"训练时段: {s['train']} | 预测时段: {s['predict']}")
                st.write(f"模型: {', '.join(s['models'])} | 大小: {s['size_mb']:.1f}MB")
                st.code(s['dir'], language=None)

        # Select and load
        st.divider()
        st.subheader("\U0001f4e5 加载模型并预测新数据")
        model_dir = st.text_input(
            "模型包路径 (outputs目录)",
            value=saved[0]['dir'] if saved else "",
            placeholder="输入包含 models/ 子目录的输出路径"
        )

        if st.button("\U0001f4e5 加载模型", type="primary"):
            if not model_dir or not os.path.isdir(os.path.join(model_dir, "models")):
                st.error("无效路径: 该目录下未找到 models/ 子目录")
            else:
                try:
                    pixel_models, meta = load_model_package(model_dir)
                    st.session_state.loaded_models = pixel_models
                    st.session_state.loaded_meta = meta
                    st.success(f"✅ 已加载 {len(pixel_models)} 个模型")
                    st.json({
                        "训练时段": meta.get("train_period", "?"),
                        "预测时段": meta.get("predict_period", "?"),
                        "模型": list(pixel_models.keys()),
                        "Y文件夹": meta.get("y_folder", "?"),
                    })
                except Exception as e:
                    st.error(f"加载失败: {e}")

        # Prediction using loaded model
        if "loaded_models" in st.session_state and st.session_state.loaded_models:
            st.divider()
            st.subheader("\U0001f4ca 使用已加载模型预测新数据")

            new_x_str = st.text_area(
                "新 X 自变量文件夹 (每行一个，留空则使用原数据)",
                value="",
                placeholder="留空则使用训练时的X数据路径",
                height=60,
            )
            new_y = st.text_input("新 Y 文件夹 (可选，用于评估)", value="")

            pred_output = st.text_input("预测输出文件夹", value=os.path.join(model_dir, "new_predictions"))

            if st.button("\U0001f680 运行预测", type="primary"):
                meta = st.session_state.loaded_meta
                pixel_models = st.session_state.loaded_models

                # Determine X folders
                new_x_list = [p.strip() for p in new_x_str.split("\n") if p.strip()]
                if not new_x_list:
                    new_x_list = meta.get("x_folders", [])
                    if not new_x_list:
                        st.error("请提供X文件夹路径")
                        st.stop()

                with st.spinner("加载数据并预测..."):
                    from src.data_loader import discover_tiff_files, load_raster_stack, get_valid_mask
                    from src.preprocessing import scale_data
                    from src.predictor import generate_predictions

                    # Load X data
                    x_cubes = []
                    x_labels_list = []
                    x_profiles = []
                    for xf in new_x_list:
                        xfd, _ = discover_tiff_files(xf)
                        if not xfd:
                            st.warning(f"X文件夹无数据: {xf}")
                            continue
                        xl, xc, xp = load_raster_stack(xfd)
                        x_cubes.append(xc)
                        x_labels_list.append(xl)
                        x_profiles.append(xp)

                    if not x_cubes:
                        st.error("无有效X数据")
                        st.stop()

                    n_pixels = x_cubes[0].shape[1] * x_cubes[0].shape[2]

                    # Use Y data if available, else dummy
                    y_test_2d = np.zeros((n_pixels, len(x_labels_list[0])))
                    if new_y and os.path.isdir(new_y):
                        yfd, _ = discover_tiff_files(new_y)
                        if yfd:
                            yl, yc, yp = load_raster_stack(yfd)
                            y_test_2d = yc.reshape(yc.shape[0], -1).T

                    # Build X_test
                    from src.preprocessing import build_design_matrix
                    n_x_vars = len(x_cubes)
                    n_time = len(x_labels_list[0])
                    mask = ~np.isnan(x_cubes[0][0])  # simple mask
                    valid_indices = np.where(mask.ravel())[0]
                    n_valid = len(valid_indices)

                    # Simple design matrix: use all time steps
                    ti = list(range(n_time))
                    _, X_train_3d, _, X_test_3d, _ = build_design_matrix(
                        x_cubes[0][:1], x_cubes, ti, ti, ti, ti,
                        mask.reshape(1, *mask.shape) if mask.ndim == 2 else mask
                    )

                    # Scale
                    X_train_scaled, X_test_scaled, _ = scale_data(
                        X_train_3d, X_test_3d, method=meta.get("scaling", "standard"))

                    # Generate predictions
                    y_profile = x_profiles[0]
                    labels = x_labels_list[0]
                    predict_outputs = generate_predictions(
                        pixel_models, X_test_scaled, valid_indices,
                        y_profile, pred_output, labels,
                        folder_name="predictions"
                    )

                    st.success(f"✅ 预测完成! 输出保存到: {pred_output}")
                    for mn, files in predict_outputs.items():
                        st.write(f"**{mn}**: {len(files)} 个 TIFF 文件")
                        for f in files[:3]:
                            st.code(f, language=None)


# ============================================================
# PAGE: SHAP 独立分析
# ============================================================
elif page == "\U0001f52c SHAP解释":
    st.header("\U0001f52c SHAP 模型可解释性分析")

    # ── 简要说明 ──
    with st.expander("\U0001f4d6 什么是 SHAP？点击查看说明", expanded=False):
        st.markdown("""
### SHAP (SHapley Additive exPlanations) 是什么？

SHAP 是一种基于博弈论的机器学习模型解释方法，能告诉你每个输入变量对预测值的贡献大小。
- **正值** = 该变量推高了预测值，**负值** = 该变量拉低了预测值
- **mean(|SHAP|)** 越大 → 该变量对模型预测越重要

### 在地理空间建模中的用途

1. **找到驱动因素**：哪些 X 变量最能解释 Y 的空间变化？
2. **验证模型合理性**：SHAP 特征排序是否符合先验认知？
3. **发现空间异质性**：某个变量在不同区域的贡献是否不同？

### 使用步骤

1. 导入 Y 和 X 数据（和训练页面一样的格式）
2. 选择模型类型并设置参数
3. 点击"训练并分析" → 自动检测年份范围、训练模型并计算 SHAP
        """)

    # ── 数据导入 ──
    st.subheader("\U0001f4c2 导入数据")
    col1, col2 = st.columns(2)
    with col1:
        shap_y = st.text_input("Y 因变量文件夹", key="shap_y",
                                placeholder="输入Y变量TIFF文件夹路径")
    with col2:
        shap_mode = st.selectbox("数据时间单位", ["yearly", "monthly"],
                                  format_func=lambda x: "年度数据" if x=="yearly" else "月度数据",
                                  key="shap_mode")

    shap_x_str = st.text_area("X 自变量文件夹 (每行一个)", key="shap_x",
                               placeholder="每行一个路径，训练和预测的X数据都放这里", height=60)
    shap_x_list = [p.strip() for p in shap_x_str.split("\n") if p.strip()]
    # Auto-detect years — use ALL available years for training
    shap_years = []
    if shap_y and os.path.isdir(shap_y):
        from src.data_loader import discover_tiff_files
        yf, _ = discover_tiff_files(shap_y)
        if yf:
            shap_years = sorted(set(int(k.split("-")[0]) if "-" in k else int(k) for k in yf.keys()))
            st.caption(f"📅 自动检测到数据年份: {shap_years[0]}—{shap_years[-1]}（共 {len(shap_years)} 年），将使用全部年份训练并做 SHAP 分析")


    # ── 模型选择 ──
    st.subheader("\U0001f916 选择模型")
    tree_options = ["rf", "xgboost", "lightgbm", "extratrees", "gbr"]
    shap_model_name = st.selectbox(
        "模型类型",
        tree_options,
        format_func=lambda x: {"rf":"Random Forest","xgboost":"XGBoost","lightgbm":"LightGBM",
                                "extratrees":"Extra Trees","gbr":"Gradient Boosting"}[x],
        key="shap_model_sel"
    )

    # Model params
    col_p1, col_p2 = st.columns(2)
    params = {}
    if shap_model_name in ("rf", "extratrees"):
        with col_p1: params["n_estimators"] = st.number_input("树数量", 10, 500, 100, 10, key="shp_n")
        with col_p2: params["max_depth"] = st.number_input("最大深度", 2, 50, 10, 1, key="shp_d")
    elif shap_model_name in ("xgboost", "lightgbm", "gbr"):
        with col_p1: params["n_estimators"] = st.number_input("树数量", 10, 500, 100, 10, key="shp_n")
        with col_p2: params["max_depth"] = st.number_input("最大深度", 2, 20, 6, 1, key="shp_d")
        params["learning_rate"] = st.slider("学习率", 0.01, 0.5, 0.1, 0.01, key="shp_lr")

    sample_pct = st.slider("SHAP 采样比例 (%)", 1, 100, 50, 1,
                           help="从有效像素中按百分比随机采样用于 SHAP 计算", key="shp_pct")
    max_s = None  # 由采样比例动态计算

    # ── 运行 ──
    if st.button("\U0001f52c 训练模型并 SHAP 分析", type="primary", disabled=not (shap_y and shap_x_list)):
        with st.spinner("训练模型 + 计算 SHAP ..."):
            try:
                # Pre-check: SHAP library available
                try:
                    import shap as _shap_check
                except ImportError:
                    st.error("SHAP 库未安装。请在终端运行: pip install shap")
                    st.stop()

                from src.data_loader import discover_tiff_files, load_raster_stack, get_valid_mask
                from src.preprocessing import build_design_matrix, scale_data
                from src.trainer import train_all_pixels
                from src.config import ModelConfig
                from src.shap_analysis import compute_shap_values, plot_shap_summary

                # Load data
                st.info("加载数据...")
                yf, yu = discover_tiff_files(shap_y)
                if not yf:
                    st.error("Y文件夹无TIFF文件"); st.stop()
                y_labels, y_cube, y_profile = load_raster_stack(yf)

                x_cubes, x_names_shap = [], []
                for xf in shap_x_list:
                    xfd, _ = discover_tiff_files(xf)
                    if xfd:
                        xl, xc, xp = load_raster_stack(xfd)
                        x_cubes.append(xc)
                        x_names_shap.append(os.path.basename(xf))

                if not x_cubes:
                    st.error("X文件夹无有效数据"); st.stop()

                n_time = y_cube.shape[0]
                n_rows, n_cols = y_cube.shape[1], y_cube.shape[2]
                mask = get_valid_mask(y_cube, *x_cubes, min_valid_ratio=0.3)
                n_valid = mask.sum()
                st.info(f"有效像素: {n_valid} / {n_rows*n_cols}")

                if n_valid == 0:
                    st.error("无有效像素"); st.stop()

                # 全自动：使用所有时间步训练，SHAP 不需要预测期
                y_tr_idx = list(range(n_time))
                y_pr_idx = y_tr_idx  # SHAP 不使用预测期，与训练期相同即可

                # Build design matrix
                yt2d, Xt3d, yp2d, Xp3d, vi = build_design_matrix(
                    y_cube, x_cubes, y_tr_idx, y_pr_idx,
                    y_tr_idx, [y_tr_idx[-1]]*len(y_pr_idx) if not y_pr_idx else y_pr_idx,
                    mask
                )
                Xt_sc, Xp_sc, _ = scale_data(Xt3d, Xp3d, method="standard")

                n_train_steps = Xt3d.shape[1]
                st.info(f"训练时间步数: {n_train_steps}（全部年份），训练像素数: {yt2d.shape[0]}")
                if n_train_steps < 3:
                    st.warning(f"⚠️ 数据年份只有 {n_train_steps} 年，树模型可能过于简单，建议补充更多年份的数据")

                # ── Y 方差诊断（SHAP 全零的常见根因）──
                y_valid = yt2d[~np.isnan(yt2d)]
                y_std = float(np.std(y_valid)) if len(y_valid) > 1 else 0.0
                y_range = float(np.max(y_valid) - np.min(y_valid)) if len(y_valid) > 1 else 0.0
                print(f"[SHAP-DIAG] Y变量: std={y_std:.6g}, range=[{np.min(y_valid):.6g}, {np.max(y_valid):.6g}], n_valid={len(y_valid)}", flush=True)
                if y_std < 1e-8 or y_range < 1e-8:
                    st.error(
                        f"❌ **Y 变量方差极小（std={y_std:.2e}），模型无法学习任何模式，SHAP 必然全为零。**\n\n"
                        f"请更换方差更大的 Y 变量（如 T2M、降水等），或检查数据是否全为同一值。"
                    )
                    st.stop()
                elif y_std < 1e-4:
                    st.warning(
                        f"⚠️ Y 变量方差极小（std={y_std:.2e}, range={y_range:.2e}），"
                        f"SHAP 值可能接近零或完全为零。建议更换方差更大的 Y 变量。"
                    )

                # Train models
                st.info("训练中...")
                mc = ModelConfig(name=shap_model_name, enabled=True, params=params)
                pixel_models = train_all_pixels(yt2d, Xt_sc, [mc], n_jobs=-1)

                # SHAP — 按百分比采样
                n_pixels_total = yt2d.shape[0]
                max_s = max(1, int(round(n_pixels_total * sample_pct / 100.0)))
                st.info(f"计算 SHAP（采样 {sample_pct}% = {max_s} / {n_pixels_total} 个像素）...")

                # DEBUG: 打印关键中间状态到终端
                print(f"[SHAP-DEBUG] X特征名: {x_names_shap}")
                print(f"[SHAP-DEBUG] Xt_sc形状: {Xt_sc.shape}, NaN比例: {np.isnan(Xt_sc).mean():.4f}")
                print(f"[SHAP-DEBUG] 模型成功率: {sum(1 for m in pixel_models[shap_model_name] if m is not None)} / {len(pixel_models[shap_model_name])}")

                shap_results = compute_shap_values(
                    pixel_models,
                    Xt_sc,               # 训练期特征（完整时间步，更可靠）
                    [shap_model_name],
                    x_names=x_names_shap,
                    max_samples=max_s,
                )

                # DEBUG: 打印SHAP结果摘要
                if shap_model_name in shap_results:
                    d = shap_results[shap_model_name]
                    print(f"[SHAP-DEBUG] n_used={d.get('n_used',0)}, method={d.get('method','?')}")
                    print(f"[SHAP-DEBUG] values范围: [{d['values'].min():.6e}, {d['values'].max():.6e}]")
                    print(f"[SHAP-DEBUG] mean(|SHAP|): {np.abs(d['values']).mean(axis=0)}")

                if shap_results:
                    # 检查是否有全局错误
                    if "_error" in shap_results:
                        st.error(f"❌ {shap_results['_error']}")
                    elif shap_model_name in shap_results:
                        data = shap_results[shap_model_name]
                        if "_error" in data and data.get("n_used", 0) == 0:
                            # 完全失败：展示真实错误信息
                            st.error(f"❌ {data['_error']}")
                            errs = data.get("errors", [])
                            if errs:
                                with st.expander("🔍 查看详细错误（前5条）"):
                                    for e in errs:
                                        st.code(e)
                        else:
                            # 成功（含 feature_importances_ 兜底）
                            default_out = os.path.join(os.getcwd(), "outputs")
                            out_dir = getattr(st.session_state, "output_dir", default_out)
                            shap_out = os.path.join(out_dir, "charts", "shap")
                            os.makedirs(shap_out, exist_ok=True)
                            shap_charts = plot_shap_summary(shap_results, shap_out)

                            n_used = data.get("n_used", data["values"].shape[0])
                            method = data.get("method", "shap_explainer")

                            if method == "feature_importances":
                                st.warning(
                                    "⚠️ SHAP 计算全部失败，已自动回退到模型内置的 **feature_importances_**。"
                                    "这是基于基尼不纯度的特征重要性，不是 SHAP 值。"
                                )
                            st.success(f"✅ 完成! 基于 {n_used} 个像素的分析，图表保存到 {shap_out}")
                            if method != "feature_importances":
                                st.metric("SHAP 基准值 (均值)", f"{data['base_value']:.4f}")

                            # ── 检测 SHAP 全零 ──
                            vals_check = data["values"]
                            shap_abs_max = float(np.abs(vals_check).max()) if vals_check.size > 0 else 0.0
                            if shap_abs_max < 1e-15 and method != "feature_importances":
                                st.warning(
                                    "⚠️ **SHAP 值全为零！**\n\n"
                                    "这说明模型对所有输入都预测了几乎相同的值，"
                                    "导致每个特征的 SHAP 贡献为 0。\n\n"
                                    "**常见原因**：Y 变量的方差极小（如 LAI 值在 0.1–0.5 之间几乎不变），"
                                    "模型简化为常数预测器。\n\n"
                                    "**建议**：更换方差更大的 Y 变量（如 T2M、降水等），"
                                    "或检查输入数据是否存在全 NaN / 异常填充。"
                                )

                            # 特征贡献表
                            vals = data["values"]
                            fnames = data["feature_names"]
                            if method == "feature_importances":
                                mean_vals = vals.mean(axis=0)
                                col_label = "mean(feature_importance)"
                            else:
                                mean_vals = np.abs(vals).mean(axis=0)
                                col_label = "mean(|SHAP|)"

                            order = np.argsort(mean_vals)[::-1]
                            st.markdown(f"### 各特征平均贡献 ({col_label})")

                            # 计算相对重要性百分比
                            total_mean = float(mean_vals.sum())
                            rel_pct = [(float(mean_vals[i]) / total_mean * 100.0) if total_mean > 0 else 0.0
                                       for i in order]

                            def _fmt(v):
                                """对极小值自动用科学计数法，避免全显示 0.000000"""
                                av = abs(float(v))
                                if av == 0:
                                    return "0"
                                if av < 1e-4:
                                    return f"{v:.2e}"
                                if av < 1:
                                    return f"{v:.6f}"
                                return f"{v:.4f}"

                            sh_df = pd.DataFrame({
                                "排名": list(range(1, len(order)+1)),
                                "特征": [fnames[i] for i in order],
                                col_label: [_fmt(mean_vals[i]) for i in order],
                                "相对重要性 (%)": [f"{v:.2f}" for v in rel_pct],
                            })
                            st.dataframe(sh_df, use_container_width=True)

                            # 错误提示
                            errs = data.get("errors", [])
                            n_failed = data.get("n_failed", 0)
                            if n_failed > 0:
                                st.info(f"ℹ️ {n_failed} 个像素计算失败（共 {data.get('n_total', '?')} 个）")
                            if errs:
                                with st.expander("🔍 查看错误详情"):
                                    for e in errs:
                                        st.code(e)

                            st.markdown("### 图表")
                            for sc in shap_charts:
                                if os.path.exists(sc):
                                    st.image(sc, use_container_width=True)
                    else:
                        st.error("❌ SHAP 结果中未找到所选模型")

            except Exception as e:
                st.error(f"失败: {e}")
                import traceback; st.code(traceback.format_exc())


# ============================================================
# PAGE: MK趋势检验
# ============================================================
elif page == "\U0001f4ca MK趋势检验":
    st.header("\U0001f4ca Mann-Kendall 趋势检验 + Sen's 斜率分析")
    st.caption("逐像元 MK 检验 · Sen's 斜率 · 五级分类 · 自动识别年份")

    with st.expander("\U0001f4d6 方法说明", expanded=False):
        st.markdown("""
### Mann-Kendall 趋势检验
非参数统计方法，检测时间序列单调趋势，不要求正态分布。

### Sen's 斜率 (Theil-Sen Estimator)
计算所有时间点对之间斜率的中位数，对异常值鲁棒。

### 五级分类
| 类别 | 含义 | 判定 |
|------|------|------|
| +3 | 极显著增加 | Sen>0 且 \|Z\|>2.58 (p<0.01) |
| +2 | 显著增加   | Sen>0 且 1.96<\|Z\|≤2.58 (p<0.05) |
| 0  | 无变化     | \|Z\|≤1.96 或 Sen=0 |
| -2 | 显著降低   | Sen<0 且 1.96<\|Z\|≤2.58 |
| -3 | 极显著降低 | Sen<0 且 \|Z\|>2.58 |
        """)

    # ── 数据输入 + 自动识别年份 ──
    st.subheader("\U0001f4c2 数据输入")
    mk_data_dir = st.text_input(
        "TIFF 数据目录",
        placeholder="输入包含年度 TIFF 的目录，文件名含年份如 2006.tif 或 LAI_2006.tif",
        key="mk_data_dir"
    )

    mk_years_detected = []
    if mk_data_dir and os.path.isdir(mk_data_dir):
        try:
            from src.mk_trend import discover_years
            mk_years_detected = discover_years(mk_data_dir)
            st.success(f"📅 自动识别到 {len(mk_years_detected)} 个年份: "
                       f"{mk_years_detected[0]}—{mk_years_detected[-1]} "
                       f"({', '.join(str(y) for y in mk_years_detected[:5])}"
                       f"{'...' if len(mk_years_detected) > 5 else ''})")
        except Exception as e:
            st.warning(f"⚠️ 年份识别失败: {e}")

    # 自定义输出目录
    default_out = os.path.join(mk_data_dir, "MK_results") if mk_data_dir else ""
    mk_output_dir = st.text_input(
        "\U0001f4c1 输出目录",
        value=default_out if mk_data_dir else "",
        placeholder="自定义输出路径，默认 数据目录/MK_results",
        key="mk_out_dir"
    )

    # ── 参数 ──
    st.subheader("\u2699\ufe0f 参数设置")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        mk_valid_thresh = st.number_input("有效值阈值", 0.0, 1000.0, 0.0, 0.1,
            help="像元值 > 阈值视为有效", key="mk_vthresh")
    with col_p2:
        mk_min_years = st.number_input("最少有效年份", 3, 50, 8, 1,
            help="不足此数不参与计算", key="mk_miny")
    with col_p3:
        mk_tie_corr = st.checkbox("Ties 修正", value=True, key="mk_tie")

    can_run = mk_data_dir and os.path.isdir(mk_data_dir) and len(mk_years_detected) >= mk_min_years
    if st.button("\U0001f4ca 运行 MK 趋势检验 + Sen's 斜率分析", type="primary", disabled=not can_run):
        out_dir = mk_output_dir.strip() if mk_output_dir.strip() else os.path.join(mk_data_dir, "MK_results")

        with st.spinner(f"正在分析 {mk_years_detected[0]}—{mk_years_detected[-1]} ({len(mk_years_detected)}年) ..."):
            try:
                from src.mk_trend import run_mk_analysis

                result = run_mk_analysis(
                    data_dir=mk_data_dir,
                    years=mk_years_detected,
                    output_dir=out_dir,
                    valid_threshold=mk_valid_thresh,
                    minimum_valid_years=mk_min_years,
                    use_tie_correction=mk_tie_corr,
                )

                st.success(f"✅ 分析完成！输出: {out_dir}")
                st.balloons()

                cls_map = result["class_map"]
                stats = result["stats"]
                charts = result.get("charts", {})

                # ═══════ 结果总览 ═══════
                st.subheader("\U0001f4ca 结果总览")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("总像素", f"{stats['n_pixels_total']:,}")
                with c2:
                    st.metric("有效像素", f"{stats['n_pixels_valid']:,}")
                with c3:
                    st.metric("分析年份", f"{stats['n_years']}年")
                with c4:
                    st.metric("Sen's 斜率中位数",
                              f"{stats['slope_median']:.6f}"
                              if np.isfinite(stats['slope_median']) else "N/A")
                with c5:
                    st.metric("SEM 中位数",
                              f"{stats.get('sem_median', 0):.6f}"
                              if np.isfinite(stats.get('sem_median', np.nan)) else "N/A")

                # ═══════ 趋势摘要 ═══════
                inc_mask = cls_map >= 2
                dec_mask = cls_map <= -2
                change_mask = np.abs(cls_map) >= 2
                nochange_mask = cls_map == 0
                total = cls_map.size

                st.subheader("\U0001f4c8 趋势摘要")
                ci1, ci2, ci3, ci4 = st.columns(4)
                with ci1:
                    st.metric("📈 显著增加 (+2/+3)", f"{inc_mask.sum():,}",
                              f"{inc_mask.sum()/total*100:.1f}%")
                with ci2:
                    st.metric("📉 显著降低 (-2/-3)", f"{dec_mask.sum():,}",
                              f"{dec_mask.sum()/total*100:.1f}%")
                with ci3:
                    st.metric("📊 显著变化合计", f"{change_mask.sum():,}",
                              f"{change_mask.sum()/total*100:.1f}%")
                with ci4:
                    st.metric("➖ 无明显变化", f"{nochange_mask.sum():,}",
                              f"{nochange_mask.sum()/total*100:.1f}%")

                # ═══════ 五级分类表 ═══════
                with st.expander("\U0001f3f7\ufe0f 五级分类明细", expanded=False):
                    unique, counts = np.unique(cls_map, return_counts=True)
                    cls_names = {-3: "极显著降低", -2: "显著降低", 0: "无明显变化",
                                  2: "显著增加", 3: "极显著增加"}
                    stats_data = []
                    for cv in sorted(cls_names.keys()):
                        cnt = int(counts[unique == cv][0]) if cv in unique else 0
                        stats_data.append({
                            "类别": f"{cv:+d}  {cls_names[cv]}",
                            "像素数": f"{cnt:,}",
                            "占比 (%)": f"{cnt/total*100:.1f}" if total > 0 else "0.0",
                        })
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

                # ═══════ Sen's 斜率统计 ═══════
                with st.expander("\U0001f4c8 Sen's 斜率统计", expanded=False):
                    ks1, ks2, ks3, ks4 = st.columns(4)
                    with ks1:
                        st.metric("均值", f"{stats['slope_mean']:.6f}")
                    with ks2:
                        st.metric("中位数", f"{stats['slope_median']:.6f}")
                    with ks3:
                        st.metric("标准差", f"{stats['slope_std']:.6f}")
                    with ks4:
                        st.metric("范围",
                                  f"[{stats['slope_min']:.4f}, {stats['slope_max']:.4f}]")
                    ks5, ks6, _, _ = st.columns(4)
                    with ks5:
                        st.metric("斜率>0", f"{stats['slope_pos_pct']:.1f}%")
                    with ks6:
                        st.metric("斜率<0", f"{stats['slope_neg_pct']:.1f}%")

                # ═══════ 可视化图表 ═══════
                if charts:
                    st.divider()
                    st.subheader("\U0001f4ca 可视化分析")

                    # 图表按重要性排列
                    chart_order = [
                        ('classification', "五级分类空间分布"),
                        ('pie', "五级分类占比"),
                        ('slope', "Sen's 斜率分布"),
                        ('zscore', "MK Z 统计量分布"),
                        ('sem', "SEM 分布"),
                    ]

                    for key, title in chart_order:
                        if key in charts and os.path.exists(charts[key]):
                            st.markdown(f"#### {title}")
                            st.image(charts[key], use_container_width=True)

                # ═══════ 输出文件 ═══════
                with st.expander("\U0001f4c4 输出文件列表", expanded=False):
                    for f in result.get("output_files", []):
                        fname = os.path.basename(f)
                        desc = {
                            "MK.tif": "五级分类趋势图",
                            "MK_Z.tif": "MK Z 统计量",
                            "MK_slope.tif": "Sen's 斜率",
                            "MK_sem.tif": "Sen's 斜率 SEM",
                        }.get(fname, "")
                        st.caption(f"📄 `{f}` — {desc}")

            except Exception as e:
                st.error(f"MK 检验失败: {e}")
                import traceback
                st.code(traceback.format_exc())

elif page == "\u2699\ufe0f 配置历史":
    st.header("\u2699\ufe0f 配置历史记录")
    if not st.session_state.config_history:
        st.info("暂无保存的配置，在首页点击「保存配置」即可添加")
    else:
        for i, entry in enumerate(reversed(st.session_state.config_history)):
            with st.expander(f"配置 {len(st.session_state.config_history)-i} \u2014 {entry['time']}"):
                st.json(entry["config"])
                if st.button("\U0001f4e5 加载此配置", key=f"load_{i}"):
                    cfg = entry["config"]
                    st.session_state.y_folder = cfg.get("y_folder", "")
                    st.session_state.x_folders_str = "\n".join(cfg.get("x_folders", []))
                    st.session_state.output_dir = cfg.get("output_dir", "./outputs")
                    st.success("\u2705 配置已加载!")
    st.divider()
    st.subheader("\U0001f4e4 批量导入配置")
    uploaded_file = st.file_uploader("上传 JSON/YAML 配置文件", type=["json", "yaml", "yml"])
    if uploaded_file:
        content = uploaded_file.read()
        try:
            imported = json.loads(content) if uploaded_file.name.endswith(".json") else yaml.safe_load(content)
            st.json(imported)
            if st.button("\U0001f4e5 加载上传的配置"):
                st.session_state.y_folder = imported.get("y_folder", "")
                st.session_state.x_folders_str = "\n".join(imported.get("x_folders", []))
                st.session_state.output_dir = imported.get("output_dir", "./outputs")
                st.success("\u2705 已加载!")
        except Exception as e:
            st.error(f"解析失败: {e}")

st.sidebar.divider()
st.sidebar.caption("地理空间TIFF机器学习平台 V2.0")
st.sidebar.caption("支持年度/月度数据 | 双击 启动应用.vbs 打开")
