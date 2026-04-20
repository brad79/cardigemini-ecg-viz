# -*- coding: utf-8 -*-
"""
CardiGemini — 次世代心臟缺血風險評估與成像技術
app.py — Streamlit 主程式

啟動方式:
    streamlit run app.py
"""
import sys
import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 本檔案所在目錄（所有模組均平攤於此）────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from ecg_loader import load_ecg, DEFAULT_12LEAD
from compute_average_beat import beat_alignment_individual as preprocess_beat_alignment
from detect_jt_points import process_JT_point
from extract_jt_features import extract_jt_features
from predict_function import src_predict_fast_with_confidence
from label_decoder import decode_label, REGION_MAP, SEVERITY_MAP
from step2_quality import analyze_signal_quality
from mesh3DIschemia import mesh3DIschemia as _mesh3d_ischemia

# ── Bull's-Eye 圖片目錄 ────────────────────────────────────────────
_BULLSEYE_DIR = os.path.join(_APP_DIR, "data", "bullseye")

# ── MI 深度學習模型路徑（models/ → Docker → 環境變數）────────────
_MI_MODEL_DIR_CANDIDATES = [
    os.path.join(_APP_DIR, "models"),
    "/app/models",
    os.environ.get("MI_MODEL_DIR", ""),
]
_MI_MODEL_DIR = next(
    (p for p in _MI_MODEL_DIR_CANDIDATES if p and os.path.isdir(p)),
    _MI_MODEL_DIR_CANDIDATES[0]
)

_MI_MODEL_PTH      = os.path.join(_MI_MODEL_DIR, "MI_nonMI_model_10sec.pth")
_MI_MODEL_PTH_2S   = os.path.join(_MI_MODEL_DIR, "MI_nonMI_model_2.5sec.pth")
MI_THRESHOLD = 0.25

# ── 常數 ─────────────────────────────────────────────────────────────
TRAINSET_PATH = os.path.join(_APP_DIR, "data", "src_trainset.npz")
LEAD_ORDER = DEFAULT_12LEAD
FEATURE_NAMES = ["T_amp", "J_amp", "JT25_amp", "JT50_amp", "JT_slope"]

ARTERY_COLOR = {"LAD": "#E74C3C", "LCx": "#3498DB", "RCA": "#2ECC71", "Unknown": "#95A5A6"}
SEVERITY_COLOR = {1: "#27AE60", 2: "#F1C40F", 3: "#E67E22", 4: "#E74C3C", 5: "#8E44AD"}

# 參考圖片目錄（assets/）
_IMG_DIR = os.path.join(_APP_DIR, "assets")

# 冠狀動脈狹窄臨床解說（artery, sev_cat → dict）
_STENOSIS_EXPLAIN = {
    ("LAD", "輕度阻塞"): dict(
        title="LAD 前降支 輕度狹窄 (<50%)",
        territory="前壁 (Anterior Wall)、前間隔 (Anteroseptal)、心尖 (Apex)",
        risk="低－中風險",
        finding="前壁灌流輕度減少，運動時誘發缺血，SRC 呈現 Lv1–2 特徵（T 波輕度壓低或平坦）。",
        action="積極控制危險因子（高血壓、血脂、糖尿病）、抗血小板藥物治療、心臟復健計畫、定期追蹤心電圖。",
    ),
    ("LAD", "中度阻塞"): dict(
        title="LAD 前降支 中度狹窄 (50–69%)",
        territory="前壁、前間隔、前外側壁、心尖",
        risk="中－高風險 (NSTEMI 表現)",
        finding="前壁及心尖明顯缺血，靜息或運動時 ST 壓低，SRC 呈現 Lv3 特徵（T 波倒置、ST 改變）。",
        action="強化藥物治療（他汀類、β阻滯劑）、建議冠狀動脈造影（CAG），評估心導管（FFR）後決定是否 PCI。",
    ),
    ("LAD", "重度阻塞"): dict(
        title="LAD 前降支 重度狹窄 (≥70%）",
        territory="大面積前壁 + 間隔 + 心尖（左心室 40–50% 心肌）",
        risk="高風險 — STEMI / 急性心肌梗塞",
        finding="大面積前壁缺血或梗塞，ST 段明顯抬高，SRC 呈現 Lv4–5 特徵（深度 Q 波、ST 抬高）。",
        action="緊急冠狀動脈造影，優先考慮 PCI（氣球擴張 + 支架）或 CABG 外科搭橋，時間即心肌（<90 分鐘）。",
    ),
    ("LCx", "輕度阻塞"): dict(
        title="LCx 迴旋支 輕度狹窄 (<50%)",
        territory="外側壁 (Lateral Wall)、後外側壁",
        risk="低－中風險",
        finding="外側壁輕度灌流不足，I/aVL 導程輕微 T 波變化，SRC Lv1–2 特徵。",
        action="藥物控制危險因子，改善生活型態，定期心電圖追蹤。",
    ),
    ("LCx", "中度阻塞"): dict(
        title="LCx 迴旋支 中度狹窄 (50–69%)",
        territory="外側壁、後壁 (Posterior Wall)、下外側壁",
        risk="中－高風險",
        finding="外側及後壁缺血，V5–V6 / I / aVL 導程 ST 壓低，SRC Lv3 特徵，心電圖常為「靜默性」改變。",
        action="冠狀動脈造影評估，功能性缺血試驗（FFR / iFR），必要時 PCI。",
    ),
    ("LCx", "重度阻塞"): dict(
        title="LCx 迴旋支 重度狹窄 (≥70%)",
        territory="大面積外側壁、後壁、下外側壁",
        risk="高風險 — 後壁 STEMI（後導程可見 R 波增高）",
        finding="後壁大面積缺血，心電圖 12 導程可呈「鏡像」改變，SRC Lv4–5 特徵。",
        action="緊急血管重建，LCx 支配範圍廣泛時需優先 PCI / CABG。",
    ),
    ("RCA", "輕度阻塞"): dict(
        title="RCA 右冠狀動脈 輕度狹窄 (<50%)",
        territory="下壁 (Inferior Wall)、後降支 (PDA) 供血區",
        risk="低－中風險",
        finding="下壁輕度缺血，II / III / aVF 導程輕度 T 波改變，SRC Lv1–2 特徵。",
        action="藥物治療，控制危險因子，追蹤觀察。",
    ),
    ("RCA", "中度阻塞"): dict(
        title="RCA 右冠狀動脈 中度狹窄 (50–69%)",
        territory="下壁、後壁、右心室游離壁",
        risk="中－高風險，傳導系統受影響",
        finding="下壁缺血，II / III / aVF ST 壓低，可能累及竇房結 / 房室結（RCA 供血），SRC Lv3 特徵。注意心搏過緩或 AVB。",
        action="冠狀動脈造影，評估 PCI，注意心律監測（起搏備用）。",
    ),
    ("RCA", "重度阻塞"): dict(
        title="RCA 右冠狀動脈 重度狹窄 (≥70%)",
        territory="大面積下壁 + 後壁 + 右心室",
        risk="高風險 — 下壁 STEMI，房室傳導阻滯",
        finding="下壁大面積梗塞，II / III / aVF ST 抬高，可能出現三度 AVB 或心源性休克，SRC Lv4–5 特徵。",
        action="緊急 PCI，臨時心臟節律器備用，右心室梗塞需謹慎補液，避免使用硝酸鹽。",
    ),
}

# ── Bull's-Eye 極座標扇形表 ──────────────────────────────────────────
# (region_id, ring, theta_start, theta_end)
# ring 0=心尖, 1=遠端, 2=中段, 3=基部（最外圈）
# Plotly barpolar: rotation=90, direction=clockwise
#   → theta=0°=頂部(12點)=前壁ANT,  90°=右=間隔SEPT,
#     180°=底部=下壁INF,            270°=左=外側壁LAT
# 標準臨床 Bull's-Eye 方位：前壁頂部, 外側左側, 下壁底部, 間隔右側
POLAR_SECTORS = [
    # Ring 3 — 基部 (basal, outermost)
    (1,  3, 330,  30),   # Anterior Proximal LAD       (top, wraps 0°)
    (7,  3, 290, 330),   # Lateral LAD-LCx border      (upper-left)
    (8,  3, 240, 290),   # Lateral Mid LCx             (left)
    (11, 3, 200, 240),   # Inferior Lateral            (lower-left)
    (14, 3, 150, 200),   # Inferior Proximal RCA       (bottom)
    (24, 3, 100, 150),   # Inferior RCA+PDA            (lower-right)
    # Ring 2 — 中段 (mid)
    (2,  2, 330,  30),   # Anterior Mid LAD            (top)
    (5,  2, 290, 330),   # Anterolateral LAD           (upper-left)
    (9,  2, 250, 290),   # Posterolateral LAD-LCx      (left)
    (10, 2, 210, 250),   # Posterior LCx               (lower-left)
    (13, 2, 175, 210),   # Inf Posterior Lateral LCx   (lower-left→bottom)
    (12, 2, 145, 175),   # Inf Posterior LCx-RCA       (bottom)
    (15, 2, 110, 145),   # Inferior Mid RCA            (lower-right)
    (22, 2,  50, 110),   # Right Ventricular           (right)
    (25, 2,  30,  50),   # Posterior RCA dominant      (upper-right)
    (26, 2,   0,  30),   # Inferoposterior RCA dom     (near top-right)
    # Ring 1 — 遠端 (distal)
    (3,  1, 330,  30),   # Anterior Distal LAD         (top)
    (4,  1,  30,  75),   # Anteroseptal LAD            (upper-right)
    (16, 1,  75, 115),   # Inferior Distal RCA         (right)
    (17, 1, 115, 155),   # Inferior Posterior RCA      (lower-right)
    (18, 1, 155, 190),   # Inf Posterior Lateral RCA   (bottom-right)
    (19, 1, 190, 220),   # Posterior RCA-LAD border    (bottom)
    (20, 1, 220, 255),   # Posterior Lateral RCA       (lower-left)
    (21, 1, 255, 295),   # Inferior Lateral RCA        (left)
    (23, 1, 295, 330),   # Inferior Septal RCA         (upper-left→left)
    # Ring 0 — 心尖 (apical center, full circle)
    (6,  0, 0,   360),   # Apical LAD
]

# ── 3D 左心室區域映射 ────────────────────────────────────────────────
# (u_center, u_hw, v_center, v_hw)  — u/v 為弧度實際值
# u ∈ [0, 0.85π]：u 小=心尖, u 大=基部
# v ∈ [-π, π]：v≈0 前壁, v≈π/2 外側(右), v≈±π 下壁, v≈-π/2 後壁
import math as _math
_PI = _math.pi
REGION_3D_MAP = {
    1:  (0.78*_PI, 0.12*_PI, 0.00,         0.45),   # Anterior Proximal LAD
    2:  (0.60*_PI, 0.12*_PI, 0.00,         0.45),   # Anterior Mid LAD
    3:  (0.42*_PI, 0.12*_PI, 0.00,         0.45),   # Anterior Distal LAD
    4:  (0.55*_PI, 0.12*_PI, 0.30*_PI,     0.35),   # Anteroseptal
    5:  (0.55*_PI, 0.12*_PI, -0.35*_PI,    0.35),   # Anterolateral
    6:  (0.15*_PI, 0.14*_PI, 0.00,         _PI),    # Apical (full azimuth)
    7:  (0.78*_PI, 0.12*_PI, -0.50*_PI,    0.35),   # Lateral LAD-LCx border
    8:  (0.60*_PI, 0.12*_PI, -0.50*_PI,    0.35),   # Lateral Mid LCx
    9:  (0.50*_PI, 0.12*_PI, -0.65*_PI,    0.35),   # Posterolateral LAD-LCx
    10: (0.55*_PI, 0.12*_PI, -0.85*_PI,    0.35),   # Posterior LCx
    11: (0.72*_PI, 0.12*_PI, -0.75*_PI,    0.35),   # Inferior Lateral LAD-LCx
    12: (0.62*_PI, 0.12*_PI, -0.90*_PI,    0.30),   # Inferior Posterior LCx-RCA
    13: (0.58*_PI, 0.12*_PI, -0.95*_PI,    0.30),   # Inferior Posterior Lateral
    14: (0.78*_PI, 0.12*_PI, _PI,          0.45),   # Inferior Proximal RCA
    15: (0.62*_PI, 0.12*_PI, _PI,          0.45),   # Inferior Mid RCA
    16: (0.44*_PI, 0.12*_PI, _PI,          0.40),   # Inferior Distal RCA
    17: (0.60*_PI, 0.12*_PI, 0.90*_PI,     0.35),   # Inferior Posterior
    18: (0.56*_PI, 0.12*_PI, 0.95*_PI,     0.30),   # Inferior Posterior Lateral
    19: (0.55*_PI, 0.12*_PI, 0.80*_PI,     0.30),   # Posterior RCA-LAD border
    20: (0.58*_PI, 0.12*_PI, 0.65*_PI,     0.35),   # Posterior Lateral
    21: (0.65*_PI, 0.12*_PI, 0.75*_PI,     0.35),   # Inferior Lateral
    22: (0.68*_PI, 0.12*_PI, -0.60*_PI,    0.35),   # Right Ventricular
    23: (0.52*_PI, 0.12*_PI, 0.85*_PI,     0.30),   # Inferior Septal
    24: (0.70*_PI, 0.12*_PI, _PI,          0.40),   # Inferior RCA+PDA
    25: (0.62*_PI, 0.12*_PI, 0.75*_PI,     0.35),   # Posterior RCA dominant
    26: (0.66*_PI, 0.12*_PI, 0.90*_PI,     0.40),   # Inferoposterior RCA dominant
}


# ── Helper: hex → rgba string ────────────────────────────────────────
def _hex_to_rgba(hex_str: str, alpha: float = 1.0) -> str:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Bull's-Eye Polar Map（26 個缺血區域 + 冠狀動脈供血範圍）────────
def make_bullseye_polar(region: int, severity: int, artery: str) -> go.Figure:
    """
    臨床標準 Bull's-Eye Polar Map
    - 前壁在頂部(ANT/LAD), 外側壁在左(LAT/LCx), 下壁在底部(INF/RCA), 間隔在右(SEPT)
    - 預測缺血區域：以 severity 色高亮
    - 同血管供血區：以血管色半透明底色
    - 其他區域：以各血管淡色區別
    - 顯示冠狀動脈名稱與供血範圍標注、供血界線
    """
    RING_RADII = {0: (0.0, 0.22), 1: (0.22, 0.50), 2: (0.50, 0.75), 3: (0.75, 1.00)}

    r_vals, theta_vals, widths, bases, colors, hovertexts = [], [], [], [], [], []

    for reg_id, ring, theta_s, theta_e in POLAR_SECTORS:
        inner, outer = RING_RADII[ring]
        r_span = outer - inner
        if theta_e > theta_s:
            theta_mid = (theta_s + theta_e) / 2.0
            ang_width  = theta_e - theta_s
        else:
            theta_mid = (theta_s + theta_e + 360) / 2.0 % 360
            ang_width  = 360 - theta_s + theta_e

        reg_artery = REGION_MAP[reg_id]["artery"]
        if reg_id == region:
            color = SEVERITY_COLOR[severity]
        elif reg_artery == artery:
            color = _hex_to_rgba(ARTERY_COLOR[reg_artery], 0.55)
        else:
            color = _hex_to_rgba(ARTERY_COLOR[reg_artery], 0.15)

        htext = (f"<b>R{reg_id}: {REGION_MAP[reg_id]['locationEn']}</b><br>"
                 f"Artery: {reg_artery}")
        if reg_id == region:
            htext += f"<br><b>Severity Lv{severity} ← PREDICTED</b>"

        r_vals.append(r_span);    theta_vals.append(theta_mid)
        widths.append(ang_width); bases.append(inner)
        colors.append(color);     hovertexts.append(htext)

    fig = go.Figure()

    # ── Layer 1: region sectors ───────────────────────────────────────
    fig.add_trace(go.Barpolar(
        r=r_vals, theta=theta_vals, width=widths, base=bases,
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.55)",
        marker_line_width=0.7,
        hovertext=hovertexts, hoverinfo="text",
        showlegend=False,
    ))

    # ── Layer 2: territory boundary lines ────────────────────────────
    # 三條分界線: LAD/LCx (upper-left ~290°), LCx/RCA (lower-left ~200°), RCA/LAD (right ~100°)
    for ang, lbl in [(290, "LAD|LCx"), (200, "LCx|RCA"), (100, "RCA|LAD")]:
        fig.add_trace(go.Scatterpolar(
            r=[0.0, 1.06], theta=[ang, ang],
            mode="lines",
            line=dict(color="rgba(60,60,80,0.55)", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Layer 3: artery name labels inside each territory ─────────────
    # (r, theta, label, artery_key)
    territory_info = [
        (0.63,   0,  "LAD\n前降支",   "LAD"),   # top  = anterior
        (0.63, 270,  "LCx\n迴旋支",   "LCx"),   # left = lateral
        (0.63, 158,  "RCA\n右冠狀動脈","RCA"),   # bottom = inferior
    ]
    for r_pos, theta_pos, lbl, art in territory_info:
        fig.add_trace(go.Scatterpolar(
            r=[r_pos], theta=[theta_pos],
            mode="text",
            text=[f"<b>{lbl}</b>"],
            textfont=dict(size=11, color=ARTERY_COLOR[art]),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Layer 4: compass direction labels ─────────────────────────────
    compass = [
        (1.18,   0, "ANT<br>前壁"),
        (1.18, 180, "INF<br>下壁"),
        (1.18, 270, "LAT<br>外側壁"),
        (1.18,  90, "SEPT<br>間隔"),
    ]
    for r_pos, theta_pos, lbl in compass:
        fig.add_trace(go.Scatterpolar(
            r=[r_pos], theta=[theta_pos],
            mode="text", text=[lbl],
            textfont=dict(size=9, color="rgba(60,60,80,0.90)"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f"Bull's-Eye Polar Map — 缺血於 <b style='color:{ARTERY_COLOR[artery]}'>"
                  f"{artery}</b> 供血區"),
            font=dict(size=13, color="#1A1A2E"), x=0.5,
        ),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.30]),
            angularaxis=dict(direction="clockwise", rotation=90, visible=False),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        font_color="#1A1A2E",
        height=460,
        margin=dict(t=50, b=50, l=70, r=70),
    )
    return fig


# ── 完整 3D 心臟模型（LV + RV + LA + RA + 冠狀動脈 + 臨床標注）────
def make_heart_3d(region: int, severity: int, artery: str) -> go.Figure:
    """
    完整三維心臟模型（Plotly 參數化曲面）：
    - LV（左心室）：依 SRC 缺血區域與嚴重度著色
      • 亮色 = 預測缺血區（以 severity 色標示）
      • 半透明 = 同血管供血範圍
      • 灰色 = 正常心肌
    - RV（右心室）、LA（左心房）、RA（右心房）：淺灰
    - 冠狀動脈走行：
      LAD（前降支）→ 前壁縱行  RED
      LCx（迴旋支）→ 外側/後壁弧行  BLUE
      RCA（右冠狀動脈）→ 下壁/後降支  GREEN
    - 臨床解剖標注：前壁、下壁、外側壁、心尖、基部、間隔
    座標系：+X=前壁, +Y=間隔(患者右側), -Y=外側壁(患者左側),
             -Z=心尖, +Z=基部
    """
    # ── LV 幾何 ───────────────────────────────────────────────────────
    n_u, n_v = 55, 80
    u_vals = np.linspace(0, 0.85 * _PI, n_u)
    v_vals = np.linspace(-_PI, _PI, n_v)
    U, V = np.meshgrid(u_vals, v_vals, indexing="ij")
    a, b, c = 2.8, 2.8, 5.0
    X_lv = a * np.sin(U) * np.cos(V)
    Y_lv = b * np.sin(U) * np.sin(V)
    Z_lv = -c * np.cos(U)   # apex 朝下

    # ── LV 著色 grid ──────────────────────────────────────────────────
    color_grid = np.zeros((n_u, n_v))

    def angular_mask(V_arr, vc, vhw):
        diff = np.arctan2(np.sin(V_arr - vc), np.cos(V_arr - vc))
        return np.abs(diff) < vhw

    for rid, (uc, uhw, vc, vhw) in REGION_3D_MAP.items():
        if REGION_MAP[rid]["artery"] == artery and rid != region:
            mask = (np.abs(U - uc) < uhw) & angular_mask(V, vc, vhw)
            color_grid = np.where(mask & (color_grid == 0.0), 0.5, color_grid)

    uc, uhw, vc, vhw = REGION_3D_MAP[region]
    pred_mask = (np.abs(U - uc) < uhw) & angular_mask(V, vc, vhw)
    color_grid[pred_mask] = 1.0

    sev_hex  = SEVERITY_COLOR[severity]
    art_rgba = _hex_to_rgba(ARTERY_COLOR.get(artery, "#95A5A6"), 0.55)
    custom_scale = [
        [0.00, "rgb(205,205,215)"], [0.38, "rgb(205,205,215)"],
        [0.39, art_rgba],           [0.61, art_rgba],
        [0.62, sev_hex],            [1.00, sev_hex],
    ]

    fig = go.Figure()

    # ── 右心室 RV（間隔側，+Y 方向）──────────────────────────────────
    n_rv = 22
    u_rv = np.linspace(0.08*_PI, 0.75*_PI, n_rv)
    v_rv = np.linspace(0.15*_PI, 0.58*_PI, n_rv)
    Urv, Vrv = np.meshgrid(u_rv, v_rv, indexing="ij")
    X_rv = 1.55 * np.sin(Urv) * np.cos(Vrv) + 0.4
    Y_rv = 1.40 * np.sin(Urv) * np.sin(Vrv) + 3.0
    Z_rv = -3.70 * np.cos(Urv) + 0.7
    fig.add_trace(go.Surface(
        x=X_rv.tolist(), y=Y_rv.tolist(), z=Z_rv.tolist(),
        colorscale=[[0,"rgb(175,183,198)"],[1,"rgb(185,193,208)"]],
        showscale=False, opacity=0.52,
        hovertemplate="右心室 Right Ventricle (RV)<extra></extra>",
        name="RV 右心室",
    ))

    # ── 左心房 LA（後方偏左，-Y 方向）────────────────────────────────
    n_a = 16
    u_a = np.linspace(0, _PI, n_a)
    v_a = np.linspace(0, 2*_PI, n_a)
    Ua, Va = np.meshgrid(u_a, v_a, indexing="ij")
    X_la = 1.75 * np.sin(Ua) * np.cos(Va) - 0.3
    Y_la = 1.50 * np.sin(Ua) * np.sin(Va) - 2.2
    Z_la = 1.20 * np.cos(Ua) + 5.1
    fig.add_trace(go.Surface(
        x=X_la.tolist(), y=Y_la.tolist(), z=Z_la.tolist(),
        colorscale=[[0,"rgb(205,182,182)"],[1,"rgb(218,195,195)"]],
        showscale=False, opacity=0.45,
        hovertemplate="左心房 Left Atrium (LA)<extra></extra>",
        name="LA 左心房",
    ))

    # ── 右心房 RA（右上方，+Y 方向）──────────────────────────────────
    X_ra = 1.45 * np.sin(Ua) * np.cos(Va) + 0.4
    Y_ra = 1.40 * np.sin(Ua) * np.sin(Va) + 3.8
    Z_ra = 1.15 * np.cos(Ua) + 5.0
    fig.add_trace(go.Surface(
        x=X_ra.tolist(), y=Y_ra.tolist(), z=Z_ra.tolist(),
        colorscale=[[0,"rgb(205,182,182)"],[1,"rgb(218,195,195)"]],
        showscale=False, opacity=0.45,
        hovertemplate="右心房 Right Atrium (RA)<extra></extra>",
        name="RA 右心房",
    ))

    # ── LV 主表面（放在 RV/心房之後渲染，確保正確覆蓋）─────────────
    fig.add_trace(go.Surface(
        x=X_lv.tolist(), y=Y_lv.tolist(), z=Z_lv.tolist(),
        surfacecolor=color_grid.tolist(),
        colorscale=custom_scale, cmin=0.0, cmax=1.0,
        showscale=False, opacity=0.93, hoverinfo="skip",
        name="LV 左心室",
    ))

    # ── 冠狀動脈走行（Scatter3d 線條）────────────────────────────────
    # LAD 前降支：沿前壁(v=0)縱行，從基部到心尖
    lad_u = np.linspace(0.08*_PI, 0.84*_PI, 50)
    lad_x = a * np.sin(lad_u) * 1.08          # v=0 前壁，略微偏離以免嵌入
    lad_y = np.zeros(50)
    lad_z = -c * np.cos(lad_u)
    fig.add_trace(go.Scatter3d(
        x=lad_x.tolist(), y=lad_y.tolist(), z=lad_z.tolist(),
        mode="lines",
        line=dict(color=ARTERY_COLOR["LAD"], width=6),
        name="LAD 前降支",
        hovertemplate="<b>LAD 前降支</b><br>Left Anterior Descending Artery<extra></extra>",
    ))

    # LCx 迴旋支：沿外側壁(v 負方向)弧行
    lcx_v = np.linspace(-0.08*_PI, -0.88*_PI, 40)
    lcx_u = np.linspace(0.79*_PI, 0.73*_PI, 40)
    lcx_x = a * np.sin(lcx_u) * np.cos(lcx_v) * 1.06
    lcx_y = b * np.sin(lcx_u) * np.sin(lcx_v) * 1.06
    lcx_z = -c * np.cos(lcx_u)
    fig.add_trace(go.Scatter3d(
        x=lcx_x.tolist(), y=lcx_y.tolist(), z=lcx_z.tolist(),
        mode="lines",
        line=dict(color=ARTERY_COLOR["LCx"], width=6),
        name="LCx 迴旋支",
        hovertemplate="<b>LCx 迴旋支</b><br>Left Circumflex Artery<extra></extra>",
    ))

    # RCA 右冠狀動脈：沿間隔/下壁弧行，再沿後降支(PDA)下行
    rca_v_av = np.linspace(0.55*_PI, _PI, 35)
    rca_u_av = np.linspace(0.80*_PI, 0.78*_PI, 35)
    rca_x_av = a * np.sin(rca_u_av) * np.cos(rca_v_av) * 1.06
    rca_y_av = b * np.sin(rca_u_av) * np.sin(rca_v_av) * 1.06
    rca_z_av = -c * np.cos(rca_u_av)
    # PDA 後降支段
    pda_u = np.linspace(0.78*_PI, 0.85*_PI, 20)
    pda_x = a * np.sin(pda_u) * np.cos(_PI) * 1.06
    pda_y = np.zeros(20)
    pda_z = -c * np.cos(pda_u)
    fig.add_trace(go.Scatter3d(
        x=np.concatenate([rca_x_av, pda_x]).tolist(),
        y=np.concatenate([rca_y_av, pda_y]).tolist(),
        z=np.concatenate([rca_z_av, pda_z]).tolist(),
        mode="lines",
        line=dict(color=ARTERY_COLOR["RCA"], width=6),
        name="RCA 右冠狀動脈",
        hovertemplate="<b>RCA 右冠狀動脈</b><br>Right Coronary Artery<extra></extra>",
    ))

    # ── 臨床解剖標注 ──────────────────────────────────────────────────
    # (x, y, z, 中文, 英文)
    anatomy_labels = [
        ( a*1.05,  0,      0,      "前壁",   "Anterior Wall (LAD)"),
        (-a*1.05,  0,     -1.5,    "下壁",   "Inferior Wall (RCA)"),
        ( 0,      -b*1.05, 0,      "外側壁", "Lateral Wall (LCx)"),
        ( 0,       0,     -c*0.90, "心尖",   "Apex"),
        ( 0,       0,      c*0.65, "基部",   "LV Base"),
        ( a*0.35,  b*1.0,  0,      "間隔",   "Septum (LAD/RCA)"),
    ]
    for tx, ty, tz, zh, en in anatomy_labels:
        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[tz],
            mode="text",
            text=[f"<b>{zh}</b><br><span style='font-size:8px'>{en}</span>"],
            textfont=dict(size=9, color="rgba(40,40,60,0.85)"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── 預測缺血區域標記 ──────────────────────────────────────────────
    cx = float(a * np.sin(uc) * np.cos(vc))
    cy = float(b * np.sin(uc) * np.sin(vc))
    cz = float(-c * np.cos(uc))
    loc_zh   = REGION_MAP[region]["location"]
    loc_en   = REGION_MAP[region]["locationEn"]
    sev_lbl  = SEVERITY_MAP[severity]["labelEn"]
    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[cz + 0.6],
        mode="markers+text",
        marker=dict(size=11, color=sev_hex, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"<b>R{region} {loc_zh}</b><br>Lv{severity}: {sev_lbl}"],
        textposition="top center",
        textfont=dict(size=10, color="#1A1A2E"),
        showlegend=False,
        hovertemplate=(f"<b>Region {region}</b><br>{loc_en}<br>"
                       f"Artery: {artery}<br>Severity Lv{severity}: {sev_lbl}<extra></extra>"),
    ))

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f"3D Heart — R{region} <b>{loc_zh}</b> "
                  f"({artery}, Lv{severity} {sev_lbl})"),
            font=dict(size=13), x=0.5,
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(245,247,252)",
            # 前偏右視角，可見前壁(LAD)、外側壁(LCx)、RV
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.5)),
            aspectmode="data",
        ),
        legend=dict(
            x=0.01, y=0.98, orientation="v",
            bgcolor="rgba(245,247,252,0.92)",
            bordercolor="rgba(60,60,80,0.25)",
            borderwidth=1,
            font=dict(size=9, color="white"),
            itemsizing="constant",
        ),
        paper_bgcolor="white",
        font_color="#1A1A2E",
        height=510,
        margin=dict(t=55, b=10, l=0, r=0),
    )
    return fig


# ── 頁面設定 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="次世代心臟缺血風險評估與成像技術",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 次世代心臟缺血風險評估與成像技術")
st.subheader("冠狀動脈狹窄及心肌病變檢測與定位")
st.caption("12 導程 ECG → 訊號品質評估 → 波形平均 → SRC 稀疏表示分類 → 缺血定位與解剖圖解")


# ── MI 深度學習模型載入（只載入一次）────────────────────────────
@st.cache_resource(show_spinner="載入 MI 深度學習模型…")
def _load_mi_engine():
    from ecg_engine import ECGAppEngine
    return ECGAppEngine(model_pth=_MI_MODEL_PTH, n_classes=1, target_sec=10)


@st.cache_resource(show_spinner="載入 MI 2.5s 深度學習模型…")
def _load_mi_engine_2s():
    from ecg_engine import ECGAppEngine
    return ECGAppEngine(model_pth=_MI_MODEL_PTH_2S, n_classes=1, target_sec=2.5)


# ── ECG 濾波器輔助函數 ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _apply_ecg_filter(data: np.ndarray, fs: float,
                      hp: float, lp: float,
                      notch_on: bool, notch_freq: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, iirnotch
    nyq = fs / 2.0
    out = data.astype(np.float64)
    if hp > 0:
        b_hp, a_hp = butter(2, hp / nyq, btype="high")
        out = filtfilt(b_hp, a_hp, out, axis=0)
    if lp < nyq:
        b_lp, a_lp = butter(4, min(lp / nyq, 0.9999), btype="low")
        out = filtfilt(b_lp, a_lp, out, axis=0)
    if notch_on and notch_freq < nyq:
        b_n, a_n = iirnotch(notch_freq / nyq, Q=30.0)
        out = filtfilt(b_n, a_n, out, axis=0)
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# ── STEP 1：載入 ECG ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.header("Step 1　載入 12 導程心電圖")

col_upload, col_hea = st.columns([3, 2])
with col_upload:
    uploaded_main = st.file_uploader(
        "上傳 ECG 檔案（.mat / .dat / .edf / .pdf）",
        type=["mat", "dat", "edf", "bdf", "pdf"],
        key="main_file",
    )
with col_hea:
    uploaded_hea = st.file_uploader(
        "若為 WFDB .dat，請一併上傳 .hea",
        type=["hea"],
        key="hea_file",
        help="WFDB 格式需要 header 檔；PDF 格式不需上傳此檔",
    )

if uploaded_main is None:
    st.info("請上傳 ECG 檔案以開始分析。")
    st.stop()

# 儲存到暫存目錄並載入
@st.cache_data(show_spinner="載入 ECG 中…")
def load_ecg_cached(main_bytes: bytes, main_name: str,
                    hea_bytes: bytes | None, hea_name: str | None):
    ext = os.path.splitext(main_name)[1].lower()

    if ext == ".pdf":
        from ECGDataLoader import ECGDataLoader
        with tempfile.TemporaryDirectory() as tmp:
            main_path = os.path.join(tmp, main_name)
            with open(main_path, "wb") as f:
                f.write(main_bytes)
            loader = ECGDataLoader(target_fs=500)
            signal_12xN, fs, _ = loader.load_from_pdf(main_path)
        # signal_12xN: (12, n_samples) → (n_samples, 12)
        data = signal_12xN.T.astype(np.float32)
        labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        return float(fs), data, labels

    with tempfile.TemporaryDirectory() as tmp:
        main_path = os.path.join(tmp, main_name)
        with open(main_path, "wb") as f:
            f.write(main_bytes)

        if hea_bytes and hea_name:
            hea_path = os.path.join(tmp, hea_name)
            with open(hea_path, "wb") as f:
                f.write(hea_bytes)

        fs, data, labels = load_ecg(main_path)
    return fs, data, labels


hea_bytes = uploaded_hea.read() if uploaded_hea else None
hea_name  = uploaded_hea.name  if uploaded_hea else None

try:
    fs_raw, data_raw, labels_raw = load_ecg_cached(
        uploaded_main.read(), uploaded_main.name, hea_bytes, hea_name
    )
except Exception as e:
    st.error(f"載入失敗：{e}")
    st.stop()

if fs_raw is None:
    fs_raw = st.number_input("無法從檔案取得取樣率，請手動輸入 (Hz)", value=500, min_value=100, max_value=5000)

_is_pdf_upload = uploaded_main.name.lower().endswith(".pdf")
n_samples, n_ch = data_raw.shape
st.success(f"載入成功  |  取樣率 {fs_raw} Hz  |  {n_samples} samples  |  {n_ch} 導程")

# 顯示原始 12 導程波形（臨床標準 4×3 格式）
with st.expander("12 導程心電圖波形預覽（前 2.5 秒）", expanded=True):
    # ── 振幅增益控制 ───────────────────────────────────────────────
    gain_col, info_col = st.columns([2, 3])
    with gain_col:
        _GAIN_MAP = {
            "0.25×  (低增益)": 0.25,
            "0.5×": 0.50,
            "1×  標準 (10 mm/mV)": 1.00,
            "2×": 2.00,
            "4×  (高增益)": 4.00,
        }
        gain_label = st.select_slider(
            "振幅增益 Amplitude Gain",
            options=list(_GAIN_MAP.keys()),
            value="1×  標準 (10 mm/mV)",
        )
        gain_factor = _GAIN_MAP[gain_label]
    with info_col:
        st.caption("臨床標準格式：3 列 × 4 行  |  走紙速度 25 mm/s  |  標準增益 10 mm/mV")

    # ── ECG 濾波器控制 ─────────────────────────────────────────────
    st.markdown("**ECG 濾波器**")
    _fc1, _fc2, _fc3, _fc4, _fc5 = st.columns([1.2, 1.2, 1.2, 1.0, 1.2])
    with _fc1:
        _filter_on = st.toggle("啟用濾波器", value=False, key="filter_on")
    with _fc2:
        _hp_freq = st.number_input(
            "高通截止 (Hz)", min_value=0.05, max_value=5.0,
            value=0.5, step=0.05, format="%.2f",
            disabled=not _filter_on, key="filter_hp",
        )
    with _fc3:
        _lp_freq = st.number_input(
            "低通截止 (Hz)", min_value=10.0, max_value=150.0,
            value=40.0, step=1.0, format="%.0f",
            disabled=not _filter_on, key="filter_lp",
        )
    with _fc4:
        _notch_on = st.toggle("Notch", value=False, key="filter_notch_on",
                              disabled=not _filter_on)
    with _fc5:
        _notch_freq_str = st.selectbox(
            "Notch 頻率", ["50 Hz", "60 Hz"],
            disabled=(not _filter_on or not _notch_on), key="filter_notch_freq",
        )
        _notch_freq_val = 50.0 if _notch_freq_str == "50 Hz" else 60.0

    # 套用濾波器（僅影響預覽，不改變後續分析用的 data_raw）
    if _filter_on:
        data_proc = _apply_ecg_filter(
            data_raw, float(fs_raw),
            _hp_freq, _lp_freq, _notch_on, _notch_freq_val,
        )
        st.caption(f"濾波中：高通 {_hp_freq:.2f} Hz | 低通 {_lp_freq:.0f} Hz"
                   + (f" | Notch {_notch_freq_val:.0f} Hz" if _notch_on else ""))
    else:
        data_proc = data_raw

    n_show   = min(n_samples, int(fs_raw * 2.5))   # 顯示前 2.5 秒（臨床標準每欄 2.5s）
    t_axis   = np.arange(n_show) / fs_raw

    # ── 臨床 4×3 導程位置表（4 行 × 3 列）─────────────────────────
    # 格式：(row, col, 標準導程名稱)
    _LAYOUT_4X3 = [
        (1, 1, "I"),   (1, 2, "aVR"), (1, 3, "V1"), (1, 4, "V4"),
        (2, 1, "II"),  (2, 2, "aVL"), (2, 3, "V2"), (2, 4, "V5"),
        (3, 1, "III"), (3, 2, "aVF"), (3, 3, "V3"), (3, 4, "V6"),
    ]

    # 大小寫不敏感的導程名稱查找表
    _label_upper_map = {l.strip().upper(): i for i, l in enumerate(labels_raw)}
    _avx_alias = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

    def _find_ch(lead_name: str) -> int | None:
        key = lead_name.strip().upper()
        idx = _label_upper_map.get(key)
        if idx is None:
            # 嘗試別名（AVR↔aVR 等）
            for alias, canon in _avx_alias.items():
                if key == alias or key == canon.upper():
                    idx = _label_upper_map.get(alias) or _label_upper_map.get(canon.upper())
                    break
        return idx if (idx is not None and idx < n_ch) else None

    fig_raw = make_subplots(
        rows=3, cols=4,
        # shared_xaxes 與 scaleanchor 衝突，改為手動統一 x 範圍
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
        subplot_titles=[ln for _, _, ln in _LAYOUT_4X3],
    )

    for row, col, lead_name in _LAYOUT_4X3:
        ch_idx = _find_ch(lead_name)
        if ch_idx is not None:
            y_sig = (data_proc[:n_show, ch_idx] * gain_factor).tolist()
        else:
            y_sig = [0.0] * n_show   # 導程不存在則顯示平線

        fig_raw.add_trace(
            go.Scatter(
                x=t_axis, y=y_sig,
                mode="lines",
                line=dict(width=1.2, color="#1A237E"),   # 深靛藍
                showlegend=False,
                name=lead_name,
            ),
            row=row, col=col,
        )

    # ECG 心電圖紙風格：標準雙層格線（大格 0.2s/0.5mV、小格 0.04s/0.1mV）
    _y_major = 0.5 * gain_factor   # 大格 0.5 mV（依增益縮放）
    _y_minor = 0.1 * gain_factor   # 小格 0.1 mV

    # 計算正方形格線所需圖高
    # ECG 標準：25 mm/s、10 mm/mV → 1 mV = 0.4 s → scaleratio = 0.4
    # subplot_h = subplot_w × (y_range) / (x_range × 2.5)
    _x_range_s = n_show / fs_raw
    _subplot_w_px = (1150 - 50) * (1 - 3 * 0.05) / 4   # ≈ 234 px（假設 1150px 容器）
    _subplot_h_px = _subplot_w_px * (4.0 * gain_factor) / (_x_range_s * 2.5)
    # row_fraction ≈ (1 - 2×v_spacing)/3 = 0.267，圖高 = subplot_h/0.267 + margins
    _fig12_h = int(max(300, min(1200, _subplot_h_px / 0.267 + 70)))

    fig_raw.update_layout(
        height=_fig12_h,
        margin=dict(t=40, b=30, l=30, r=20),
        paper_bgcolor="white",
        plot_bgcolor="#FFFDE7",      # 淡黃（心電圖紙）
        font=dict(color="#1A1A2E"),
    )
    fig_raw.update_xaxes(
        showgrid=True,
        gridcolor="#F48FB1",  # 大格深粉紅（0.2 s）
        gridwidth=1.0,
        dtick=0.2,
        minor=dict(showgrid=True, dtick=0.04, gridcolor="#FCE4EC", gridwidth=0.5),
        zeroline=False, showticklabels=False, ticks="",
    )
    fig_raw.update_yaxes(
        showgrid=True,
        gridcolor="#F48FB1",  # 大格深粉紅（0.5 mV）
        gridwidth=1.0,
        dtick=_y_major,
        minor=dict(showgrid=True, dtick=_y_minor, gridcolor="#FCE4EC", gridwidth=0.5),
        zeroline=True, zerolinecolor="#E91E63", zerolinewidth=1.2,
        showticklabels=False, ticks="",
        range=[-2.0 * gain_factor, 2.0 * gain_factor],  # 統一刻度 ±2 mV（依增益縮放）
    )
    # 統一所有子圖的 x 軸範圍（取代 shared_xaxes）
    fig_raw.update_xaxes(range=[0, _x_range_s])
    # 最下排顯示時間軸標籤
    for c in range(1, 5):
        fig_raw.update_xaxes(showticklabels=True, title_text="s", row=3, col=c)

    # 每個子圖設定 scaleanchor，確保 ECG 格線為正方形（scaleratio=0.4：1 mV = 0.4 s）
    for _i in range(12):
        _sfx = "" if _i == 0 else str(_i + 1)
        fig_raw.update_layout(**{
            f"yaxis{_sfx}": {"scaleanchor": f"x{_sfx}", "scaleratio": 0.4, "constrain": "domain"}
        })

    st.plotly_chart(fig_raw, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ── STEP 2：訊號品質 / 心律不整 ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.header("Step 2　訊號品質評估 & 心律不整偵測")

with st.spinner("評估訊號品質…"):
    sqi_results, arrhythmia_info = analyze_signal_quality(data_raw, int(fs_raw), labels_raw)

# 心率 + 心律
col_hr, col_arr = st.columns([1, 2])
with col_hr:
    st.metric("心率（Lead II）", f"{arrhythmia_info['heart_rate']:.0f} bpm")

with col_arr:
    arrs = arrhythmia_info.get("arrhythmias", [])
    if arrs:
        for a in arrs:
            st.warning(f"⚠️  {a}")
    else:
        st.success("未偵測到明顯心律不整")

# SQI 表格
import pandas as pd
sqi_df = pd.DataFrame(sqi_results)
sqi_df.columns = ["導程", "SQI (0-1)", "品質可靠"]
sqi_df["品質可靠"] = sqi_df["品質可靠"].map({True: "✅", False: "❌"})

low_quality = sqi_df[sqi_df["品質可靠"] == "❌"]["導程"].tolist()
if low_quality:
    st.warning(f"低品質導程（kurtosis < 5）：{', '.join(low_quality)}")

st.dataframe(sqi_df, use_container_width=True, hide_index=True)

# SQI bar chart
fig_sqi = go.Figure(go.Bar(
    x=sqi_df["導程"],
    y=[r["sqi"] for r in sqi_results],
    marker_color=["#2ECC71" if r["reliable"] else "#E74C3C" for r in sqi_results],
    text=[f"{r['sqi']:.2f}" for r in sqi_results],
    textposition="outside",
))
fig_sqi.add_hline(y=0.29, line_dash="dash", line_color="orange",
                  annotation_text="品質門檻 (kurtosis=5)")
fig_sqi.update_layout(
    title="各導程訊號品質指數（SQI）",
    yaxis=dict(range=[0, 1.1], title="SQI"),
    height=300,
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig_sqi, use_container_width=True)

# 確認繼續按鈕（session_state 持久化，確保手動調整特徵點時頁面不中斷）
st.divider()
# 切換新檔案時重置 proceed 狀態
if st.session_state.get("_proceed_ecg_file") != uploaded_main.name:
    st.session_state.src_proceed = False
    st.session_state._proceed_ecg_file = uploaded_main.name

if arrs or low_quality:
    st.warning("偵測到訊號問題，確認是否仍要繼續進行 SRC 分析？")
    if st.button("繼續分析（忽略品質警告）", type="primary"):
        st.session_state.src_proceed = True
        st.rerun()
else:
    if st.button("繼續分析 →", type="primary"):
        st.session_state.src_proceed = True
        st.rerun()

if not st.session_state.get("src_proceed", False):
    st.info("點擊上方按鈕以執行波形平均與 SRC 分析。")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# ── STEP 3：波形平均 + 特徵點提取 ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.header("Step 3　波形平均 & 特徵點提取")

with st.spinner("執行波形平均、J-point / T-peak 偵測…"):
    avg_ecg, rri, fs_avg, all_locs, quality_report = preprocess_beat_alignment(data_raw, int(fs_raw))

# beat 品質統計（從 per-channel quality_report 彙總）
_ch_reports = quality_report.get("channels", [])
_beats_used_list = [r.get("beats_used", 0) for r in _ch_reports if "beats_used" in r]
_n_valid_ch = len(_beats_used_list)
_avg_beats  = int(round(sum(_beats_used_list) / _n_valid_ch)) if _n_valid_ch > 0 else 0

col_q1, col_q2, col_q3 = st.columns(3)
col_q1.metric("有效導程數", f"{_n_valid_ch} / {len(_ch_reports)}")
col_q2.metric("平均納入 Beats／導程", _avg_beats)
col_q3.metric("RR Interval", f"{rri:.3f} s" if rri else "N/A")

if avg_ecg is None or np.all(np.isnan(avg_ecg)):
    st.error("無法計算平均波形（R-peak 偵測失敗，請檢查訊號）。")
    st.stop()

# J-point / T-peak 偵測
with st.spinner("偵測 J-point / T-peak…"):
    try:
        Jpos, Tpeak, Tend, DC = process_JT_point(avg_ecg, fs_avg, rri)
    except Exception as e:
        st.error(f"JT 特徵點偵測失敗：{e}")
        st.stop()

# ── 初始化可編輯特徵點（換檔案時重設 slider + committed session state）──
_jt_init_key = f"_jt_init_{uploaded_main.name}"
if not st.session_state.get(_jt_init_key):
    n_avg_init = avg_ecg.shape[1]
    _init_j, _init_t = [], []
    for _ch in range(12):
        _j_a = int(Jpos[_ch]) if not np.isnan(Jpos[_ch]) and Jpos[_ch] >= 0 else max(0, n_avg_init // 4)
        _t_a = int(Tpeak[_ch]) if not np.isnan(Tpeak[_ch]) and Tpeak[_ch] >= 0 else min(n_avg_init - 1, n_avg_init * 3 // 4)
        st.session_state[f"jslider_{_ch}"] = _j_a
        st.session_state[f"tslider_{_ch}"] = _t_a
        _init_j.append(_j_a)
        _init_t.append(_t_a)
    # committed = 已確認送給 SRC 的特徵點（初始 = 自動偵測）
    st.session_state["jpos_committed"]  = _init_j
    st.session_state["tpeak_committed"] = _init_t
    st.session_state[_jt_init_key] = True

# ── 即時預覽用特徵點（滑桿目前值）/ 已確認送 SRC 的特徵點 ─────────────
n_avg = avg_ecg.shape[1]
t_avg = np.arange(n_avg) / fs_avg * 1000  # ms

_j_auto_list = [int(Jpos[ch]) if not np.isnan(Jpos[ch]) and Jpos[ch] >= 0 else -1 for ch in range(12)]
_t_auto_list = [int(Tpeak[ch]) if not np.isnan(Tpeak[ch]) and Tpeak[ch] >= 0 else -1 for ch in range(12)]

# ── 重設為自動偵測（pending flag 模式，必須在 slider 建立前執行）────────
if st.session_state.pop("_jt_reset_pending", False):
    for _c in range(12):
        _ja = _j_auto_list[_c] if _j_auto_list[_c] >= 0 else max(0, n_avg // 4)
        _ta = _t_auto_list[_c] if _t_auto_list[_c] >= 0 else min(n_avg - 1, n_avg * 3 // 4)
        st.session_state[f"jslider_{_c}"] = _ja
        st.session_state[f"tslider_{_c}"] = _ta
    st.session_state["jpos_committed"]  = [st.session_state[f"jslider_{c}"] for c in range(12)]
    st.session_state["tpeak_committed"] = [st.session_state[f"tslider_{c}"] for c in range(12)]
    st.rerun()

# ── 驗證 slider session_state 完整性（防止 init key 存在但值遺失）────────
_needs_reinit = any(
    f"jslider_{_ch}" not in st.session_state or f"tslider_{_ch}" not in st.session_state
    for _ch in range(12)
)
if _needs_reinit:
    st.session_state.pop(_jt_init_key, None)
    st.rerun()

# 滑桿目前值（即時預覽用）
Jpos_use  = np.array([st.session_state.get(f"jslider_{ch}", max(0, _j_auto_list[ch])) for ch in range(12)])
Tpeak_use = np.array([st.session_state.get(f"tslider_{ch}", max(0, _t_auto_list[ch])) for ch in range(12)])

# 已確認提交值（SRC 使用）
Jpos_committed  = np.array(st.session_state.get("jpos_committed",  Jpos_use.tolist()), dtype=int)
Tpeak_committed = np.array(st.session_state.get("tpeak_committed", Tpeak_use.tolist()), dtype=int)

# 特徵提取（使用已確認的特徵點）
with st.spinner("提取 JT 特徵…"):
    try:
        feature = extract_jt_features(avg_ecg[:12], Jpos_committed, Tpeak_committed, DC[:12])
    except Exception as e:
        st.error(f"特徵提取失敗：{e}")
        st.stop()

# ── 是否有未確認的修改 ─────────────────────────────────────────────
_is_jt_modified = any(
    int(Jpos_use[ch]) != _j_auto_list[ch] or int(Tpeak_use[ch]) != _t_auto_list[ch]
    for ch in range(12)
)
_is_jt_pending = (
    list(Jpos_use.astype(int).tolist()) != list(Jpos_committed.tolist())
    or list(Tpeak_use.astype(int).tolist()) != list(Tpeak_committed.tolist())
)

# ── 12 導程波形 + 手動調整滑桿（4×3，每格：波形圖 + J/T 滑桿）──────
st.markdown("#### 12 導程平均波形　— 紅點 J-point ｜ 藍三角 T-peak")
st.caption(
    "拖曳各導程下方滑桿可即時調整特徵點位置（波形圖即時更新）。"
    "調整完畢後點擊下方「確認修改」按鈕以重新執行 SRC 分析。"
    "　🔴 實心 = 目前滑桿值　⭕ 空心 = 上次確認值"
)

# 計算 Step 3 子圖尺寸（正方形格線）
# 平均波形用 50 mm/s（大格 100ms）×10 mm/mV（大格 0.5mV）→ scaleratio=200（1mV=200ms）
# 正方形格線條件：subplot_h = col_w × (dtick_x/beat_ms) × (y_range/dtick_y)
#              = col_w × (100/beat_ms) × (2×y_half/0.5)
_beat_ms = float(t_avg[-1]) if len(t_avg) > 1 else 600.0
_y3_max_amp = float(max(abs(avg_ecg[_ch]).max() for _ch in range(min(12, avg_ecg.shape[0]))))
_y3_half = max(1.0, _y3_max_amp * 1.25)   # ±半幅，至少 ±1 mV，留 25% 餘量
_col3_px = 380   # 3-column Streamlit 版面約 380 px/欄
_fig3_h = max(150, min(450, int(_col3_px * (100.0 / _beat_ms) * (2 * _y3_half / 0.5))))

for _ri in range(4):
    _gcols = st.columns(3)
    for _ci in range(3):
        _ch = _ri * 3 + _ci
        if _ch >= 12:
            break
        with _gcols[_ci]:
            _j_cur = int(Jpos_use[_ch])
            _t_cur = int(Tpeak_use[_ch])
            _j_com = int(Jpos_committed[_ch])
            _t_com = int(Tpeak_committed[_ch])
            _y     = avg_ecg[_ch].tolist()

            _pending_ch = (_j_cur != _j_com or _t_cur != _t_com)
            _mod_ch     = (_j_cur != _j_auto_list[_ch] or _t_cur != _t_auto_list[_ch])
            _badge = " ⏳" if _pending_ch else (" ✎" if _mod_ch else "")

            # ── 導程小圖 ─────────────────────────────────────────
            _fig = go.Figure()
            _fig.add_trace(go.Scatter(
                x=t_avg.tolist(), y=_y, mode="lines",
                line=dict(width=1.4, color="#2C3E50"),
                showlegend=False,
            ))
            # J-point 目前滑桿值（實心紅點）
            if 0 <= _j_cur < n_avg:
                _fig.add_trace(go.Scatter(
                    x=[t_avg[_j_cur]], y=[_y[_j_cur]], mode="markers",
                    marker=dict(color="#E74C3C", size=9, symbol="circle",
                                line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=f"J-point（目前）<br>{t_avg[_j_cur]:.1f} ms<extra></extra>",
                ))
            # T-peak 目前滑桿值（實心藍三角）
            if 0 <= _t_cur < n_avg:
                _fig.add_trace(go.Scatter(
                    x=[t_avg[_t_cur]], y=[_y[_t_cur]], mode="markers",
                    marker=dict(color="#3498DB", size=9, symbol="triangle-up",
                                line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=f"T-peak（目前）<br>{t_avg[_t_cur]:.1f} ms<extra></extra>",
                ))
            # 已確認值（若與目前不同，顯示空心灰色）
            if _pending_ch:
                if 0 <= _j_com < n_avg:
                    _fig.add_trace(go.Scatter(
                        x=[t_avg[_j_com]], y=[_y[_j_com]], mode="markers",
                        marker=dict(color="rgba(180,0,0,0.40)", size=9, symbol="circle-open",
                                    line=dict(width=2)),
                        showlegend=False,
                        hovertemplate=f"J-point（已確認）<br>{t_avg[_j_com]:.1f} ms<extra></extra>",
                    ))
                if 0 <= _t_com < n_avg:
                    _fig.add_trace(go.Scatter(
                        x=[t_avg[_t_com]], y=[_y[_t_com]], mode="markers",
                        marker=dict(color="rgba(0,80,200,0.40)", size=9, symbol="triangle-up-open",
                                    line=dict(width=2)),
                        showlegend=False,
                        hovertemplate=f"T-peak（已確認）<br>{t_avg[_t_com]:.1f} ms<extra></extra>",
                    ))

            _fig.update_layout(
                title=dict(
                    text=f"<b>{LEAD_ORDER[_ch]}</b>{_badge}",
                    font=dict(size=12), x=0.5, xanchor="center",
                ),
                height=_fig3_h,
                margin=dict(t=28, b=2, l=8, r=8),
                paper_bgcolor="#FFFDE7",
                plot_bgcolor="#FFFDE7",
                xaxis=dict(
                    range=[0, _beat_ms],
                    zeroline=False, showticklabels=False,
                ),
                yaxis=dict(
                    range=[-_y3_half, _y3_half],
                    zeroline=True, zerolinecolor="#E91E63",
                    zerolinewidth=1.2, showticklabels=False,
                    scaleanchor="x", scaleratio=200,  # 1 mV = 200 ms（50 mm/s, 10 mm/mV）
                    constrain="domain",
                ),
            )
            # 心電圖紙格線（50 mm/s：大格 100ms，小格 20ms）
            _fig.update_xaxes(
                showgrid=True,
                gridcolor="#F48FB1", gridwidth=1.0, dtick=100,   # 大格 100 ms（50 mm/s）
                minor=dict(showgrid=True, dtick=20, gridcolor="#FCE4EC", gridwidth=0.5),
            )
            _fig.update_yaxes(
                showgrid=True,
                gridcolor="#F48FB1", gridwidth=1.0, dtick=0.5,   # 大格 0.5 mV
                minor=dict(showgrid=True, dtick=0.1, gridcolor="#FCE4EC", gridwidth=0.5),
            )
            st.plotly_chart(_fig, use_container_width=True,
                            config={"displayModeBar": False})

            # ── J / T 滑桿 ──────────────────────────────────────
            st.slider(
                "J-point",
                min_value=0, max_value=n_avg - 1,
                key=f"jslider_{_ch}",
                help=f"自動偵測值：{_j_auto_list[_ch]} 樣本 "
                     f"（{_j_auto_list[_ch]/fs_avg*1000:.0f} ms）",
            )
            st.slider(
                "T-peak",
                min_value=0, max_value=n_avg - 1,
                key=f"tslider_{_ch}",
                help=f"自動偵測值：{_t_auto_list[_ch]} 樣本 "
                     f"（{_t_auto_list[_ch]/fs_avg*1000:.0f} ms）",
            )
            st.caption(
                f"J: **{_j_cur/fs_avg*1000:.0f} ms**"
                f"　T: **{_t_cur/fs_avg*1000:.0f} ms**"
                + ("　⏳" if _pending_ch else "")
            )

# ── 確認修改 / 重設 按鈕列 ──────────────────────────────────────────
st.divider()
_cb1, _cb2, _cb3 = st.columns([3, 2, 4])
with _cb1:
    if st.button(
        "✅ 確認修改並重新執行 SRC 分析",
        type="primary",
        disabled=not _is_jt_pending,
        use_container_width=True,
    ):
        st.session_state["jpos_committed"]  = [int(st.session_state.get(f"jslider_{c}", _j_auto_list[c])) for c in range(12)]
        st.session_state["tpeak_committed"] = [int(st.session_state.get(f"tslider_{c}", _t_auto_list[c])) for c in range(12)]
        st.rerun()
with _cb2:
    if st.button("↺ 重設為自動偵測值", type="secondary", use_container_width=True):
        st.session_state["_jt_reset_pending"] = True
        st.rerun()
with _cb3:
    if _is_jt_pending:
        st.warning("特徵點已調整但尚未確認 — SRC 分析仍使用上次確認的特徵點。")
    elif _is_jt_modified:
        st.info("使用手動調整後的特徵點進行 SRC 分析。")
    else:
        st.success("使用自動偵測特徵點進行 SRC 分析。")

# ── 特徵矩陣 heatmap ──────────────────────────────────────────
with st.expander("JT 特徵矩陣（12 Lead × 5 特徵）", expanded=False):
    feat_arr = np.array(feature[:12])
    fig_hm = go.Figure(go.Heatmap(
        z=feat_arr.tolist(),
        x=FEATURE_NAMES,
        y=LEAD_ORDER[:12],
        colorscale="RdBu_r",
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in feat_arr.tolist()],
        texttemplate="%{text}",
    ))
    fig_hm.update_layout(
        title="JT 特徵矩陣（正規化後，使用已確認特徵點）",
        height=420, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# ── STEP 4：心肌缺血預測 ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.divider()
_step4_header = "Step 4　心肌缺血預測"
if _is_pdf_upload:
    _step4_header += "　*(PDF 模式 — 2.5s 模型)*"
st.header(_step4_header)


@st.cache_data(show_spinner="執行深度學習 MI 篩檢…")
def _cached_mi_predict(data_bytes: bytes, fs: float, is_pdf: bool = False) -> float:
    import torch
    signal_arr = np.frombuffer(data_bytes, dtype=np.float64).reshape(12, -1)
    engine = _load_mi_engine_2s() if is_pdf else _load_mi_engine()
    engine.model.eval()
    engine.processor.target_sec = 2.5 if is_pdf else 10.0
    engine.processor.target_len = int(engine.processor.target_fs * engine.processor.target_sec)
    tensor = engine.processor.preprocess(signal_arr, fs, as_3x4=False).unsqueeze(0).to(engine.device)
    with torch.no_grad():
        logits = engine.model(tensor)
        prob = torch.sigmoid(logits).cpu().item()
    return prob


_signal_12xN = data_raw[:, :12].T.astype(np.float64)  # (12, n_samples)
_mi_prob = _cached_mi_predict(_signal_12xN.tobytes(), float(fs_raw), _is_pdf_upload)

_mi_threshold_pct = st.slider(
    "判斷閾值（%）",
    min_value=0, max_value=100, value=25, step=1,
    key="mi_threshold_slider",
    help="調整此閾值可改變 MI 陽性判斷標準。預設值 25%。",
)
_mi_threshold = _mi_threshold_pct / 100.0
_mi_positive = _mi_prob > _mi_threshold

# Expander 標題直接顯示結果，讓使用者不展開也知道結論
_mi_expander_label = (
    f"🔴 篩檢結果：陽性（心肌缺血，MI 機率 {_mi_prob * 100:.1f}%）— 點擊展開詳情"
    if _mi_positive else
    f"✅ 篩檢結果：陰性（無心肌缺血，MI 機率 {_mi_prob * 100:.1f}%）— 點擊展開詳情"
)
with st.expander(_mi_expander_label, expanded=not _mi_positive):
    col_mi1, col_mi2, col_mi3 = st.columns(3)
    col_mi1.metric("MI 機率", f"{_mi_prob * 100:.1f} %")
    col_mi2.metric("判斷結果", "陽性（心肌缺血）" if _mi_positive else "陰性（無心肌缺血）")
    col_mi3.metric("使用閾值", f"{_mi_threshold_pct} %")
    st.progress(min(_mi_prob, 1.0), text=f"MI Probability: {_mi_prob * 100:.1f}%")

    if not _mi_positive:
        st.success("### 深度學習篩檢結果：陰性 — 未偵測到心肌缺血")
        st.markdown("""
> 深度學習模型分析 10 秒 12 導程心電圖，**未發現心肌缺血訊號**。
> 本次分析不執行 SRC 缺血定位（Step 5）。
        """)
        _img_normal = os.path.join(_IMG_DIR, "心臟剖面圖-正常無阻塞-企業色明亮風格.png")
        col_ni, col_nt = st.columns([3, 2])
        with col_ni:
            if os.path.exists(_img_normal):
                st.image(_img_normal, use_container_width=True,
                         caption="正常冠狀動脈 — 無狹窄，血流通暢")
        with col_nt:
            st.markdown("### 正常冠狀動脈")
            st.markdown("""
**供血狀態：** 三條主要冠狀動脈血流正常，心肌灌流充足。

| 血管 | 供血範圍 |
|------|---------|
| **LAD 前降支** | 前壁、前間隔、心尖 |
| **LCx 迴旋支** | 外側壁、後外側壁 |
| **RCA 右冠** | 下壁、後壁、傳導系統 |

**臨床意義：** 深度學習模型未偵測到缺血性 ECG 模式，心肌灌流正常，建議定期追蹤。
            """)
    else:
        st.warning("⚠️ 深度學習模型偵測到心肌缺血訊號，進一步執行 SRC 缺血定位（Step 5）…")

# 流程控制必須在 expander 外，才能正確阻斷 Step 5 渲染
if not _mi_positive:
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# ── STEP 5：SRC 缺血定位 ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════
st.divider()
_src_header = "Step 5　SRC 心肌缺血定位"
if _is_jt_modified and not _is_jt_pending:
    _src_header += "　*(使用手動調整特徵點)*"
st.header(_src_header)


@st.cache_data(show_spinner="執行 SRC 稀疏表示分類（OMP）…")
def _cached_src(feat_tuple: tuple, trainset_path: str):
    flat = np.array(feat_tuple, dtype=float)
    return src_predict_fast_with_confidence(flat, train_npz_path=trainset_path)


try:
    flat_feature = np.nan_to_num(feature[:12].flatten(), nan=0.0)
    label_pred, confidence = _cached_src(tuple(flat_feature.tolist()), TRAINSET_PATH)
except Exception as e:
    st.error(f"SRC 預測失敗：{e}")
    st.stop()

is_normal = (label_pred == 131)

if is_normal:
    # ── 正常 ────────────────────────────────────────────────────
    st.success("### 判讀結果：正常（無心肌缺血）")
    col_conf, _ = st.columns([1, 2])
    col_conf.metric("信心度", f"{confidence * 100:.1f} %")
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    # 正常解剖圖解
    st.divider()
    _img_normal = os.path.join(_IMG_DIR, "心臟剖面圖-正常無阻塞-企業色明亮風格.png")
    col_ni, col_nt = st.columns([3, 2])
    with col_ni:
        if os.path.exists(_img_normal):
            st.image(_img_normal, use_container_width=True,
                     caption="正常冠狀動脈 — 無狹窄，血流通暢")
    with col_nt:
        st.markdown("### 正常冠狀動脈")
        st.markdown("""
**供血狀態：** 三條主要冠狀動脈血流正常，心肌灌流充足。

| 血管 | 供血範圍 |
|------|---------|
| **LAD 前降支** | 前壁、前間隔、心尖 |
| **LCx 迴旋支** | 外側壁、後外側壁 |
| **RCA 右冠** | 下壁、後壁、傳導系統 |

**臨床意義：** ECG 未見缺血性 ST-T 改變，心肌灌流正常，建議每年定期追蹤。
        """)

else:
    # ── 缺血 ────────────────────────────────────────────────────
    decoded = decode_label(label_pred)
    artery   = decoded["artery"]
    region   = decoded["region"]
    severity = decoded["severity"]
    location = decoded["location"]
    sev_label = decoded["severityLabel"]
    icd10     = decoded["icd10"]
    is_critical = decoded["isCritical"]

    a_color  = ARTERY_COLOR.get(artery, "#95A5A6")
    sv_color = SEVERITY_COLOR.get(severity, "#95A5A6")

    # 嚴重度警示
    if is_critical:
        st.error(f"⚠️ 危急值（Severity {severity}）：{sev_label}")
    else:
        st.warning(f"Severity {severity}：{sev_label}")

    # 主要診斷卡片
    col_a, col_r, col_s, col_c = st.columns(4)
    col_a.metric("血管", artery)
    col_r.metric("缺血區域", f"Region {region}")
    col_s.metric("嚴重度", f"Lv {severity} — {sev_label}")
    col_c.metric("信心度", f"{confidence * 100:.1f} %")

    st.markdown(f"**位置**：{location}")
    st.markdown(f"**ICD-10**：`{icd10}`")

    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    # ── 血管 & 嚴重度視覺化 ────────────────────────────────────
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        with st.expander("預測缺血血管", expanded=False):
            # 血管圓餅（只標注預測血管）
            artery_labels = ["LAD", "LCx", "RCA"]
            artery_colors = [ARTERY_COLOR[a] for a in artery_labels]

            fig_artery = go.Figure(go.Pie(
                labels=artery_labels,
                values=[0.34, 0.33, 0.33],
                marker_colors=artery_colors,
                hole=0.5,
                textinfo="label",
                pull=[0.12 if a == artery else 0 for a in artery_labels],
            ))
            fig_artery.add_annotation(
                text=f"<b>{artery}</b>", x=0.5, y=0.5,
                font=dict(size=22, color=a_color),
                showarrow=False,
            )
            fig_artery.update_layout(
                title="預測缺血血管",
                height=300, margin=dict(t=40, b=10),
                showlegend=True,
            )
            st.plotly_chart(fig_artery, use_container_width=True)

    with col_v2:
        with st.expander("預測嚴重度", expanded=False):
            # 嚴重度量表
            sev_labels_all = [SEVERITY_MAP[i]["labelEn"] for i in range(1, 6)]
            sev_colors = [SEVERITY_COLOR[i] for i in range(1, 6)]
            sev_vals = [1] * 5

            fig_sev = go.Figure(go.Bar(
                x=list(range(1, 6)),
                y=sev_vals,
                marker_color=["#BDC3C7"] * 5,
                showlegend=False,
            ))
            # 高亮預測嚴重度
            fig_sev.add_trace(go.Bar(
                x=[severity],
                y=[1],
                marker_color=[sv_color],
                showlegend=False,
            ))
            fig_sev.update_layout(
                barmode="overlay",
                title="預測嚴重度",
                xaxis=dict(
                    tickvals=list(range(1, 6)),
                    ticktext=[f"Lv{i}<br>{SEVERITY_MAP[i]['labelEn']}" for i in range(1, 6)],
                ),
                yaxis=dict(showticklabels=False, range=[0, 1.5]),
                height=300,
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_sev, use_container_width=True)

    # ── 26 區域分佈（標示預測區域）──────────────────────────────
    with st.expander("26 缺血區域分佈", expanded=False):
        region_arteries = [REGION_MAP[r]["artery"] for r in range(1, 27)]
        region_colors = [
            "#FF6B6B" if r == region else ARTERY_COLOR.get(REGION_MAP[r]["artery"], "#95A5A6")
            for r in range(1, 27)
        ]
        region_texts = [
            f"R{r}: {REGION_MAP[r]['locationEn']}<br>Artery: {REGION_MAP[r]['artery']}"
            for r in range(1, 27)
        ]

        fig_regions = go.Figure(go.Bar(
            x=[f"R{r}" for r in range(1, 27)],
            y=[1] * 26,
            marker_color=region_colors,
            text=[f"R{r}" for r in range(1, 27)],
            hovertext=region_texts,
            hoverinfo="text",
        ))
        fig_regions.add_annotation(
            x=f"R{region}", y=1.1,
            text=f"▲ Region {region}<br>{decoded['location']}",
            showarrow=False,
            font=dict(size=12, color="#E74C3C"),
        )
        fig_regions.update_layout(
            title="26 個缺血區域（紅色 = 預測結果）",
            yaxis=dict(showticklabels=False, range=[0, 1.5]),
            height=280,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_regions, use_container_width=True)

    # ── 3D 心臟模型 & Bull's-Eye 極座標圖 ─────────────────────────
    st.subheader("缺血定位圖")
    col_3d, col_polar = st.columns(2)

    with col_3d:
        st.caption("3D 完整心臟模型（真實心室網格，可拖曳旋轉）— 橘紅色＝缺血區域，白色＝正常心肌")
        st.plotly_chart(_mesh3d_ischemia(label_pred), use_container_width=True)

    with col_polar:
        st.caption("Bull's-Eye Polar Map — 對應 SRC 預測區域（共 131 種情境預渲染圖）")
        _bullseye_path = os.path.join(_BULLSEYE_DIR, f"area_{label_pred}.png")
        if os.path.exists(_bullseye_path):
            st.image(_bullseye_path, use_container_width=True)
        else:
            st.warning(f"未找到 Bull's-Eye 圖片：area_{label_pred}.png")

    # ── 冠狀動脈狹窄解剖圖解 ─────────────────────────────────────
    st.divider()
    st.subheader("冠狀動脈狹窄解剖圖解與臨床說明")

    # 選擇對應圖片
    _art_key = "LCX" if artery == "LCx" else artery
    _sev_cat = ("輕度阻塞" if severity <= 2
                else ("中度阻塞" if severity == 3 else "重度阻塞"))
    _img_fname = f"心臟剖面圖-{_art_key}{_sev_cat}-企業色明亮風格.png"
    _img_path  = os.path.join(_IMG_DIR, _img_fname)

    col_img, col_txt = st.columns([3, 2])
    with col_img:
        if os.path.exists(_img_path):
            st.image(_img_path, use_container_width=True,
                     caption=f"{artery} {_sev_cat} — 冠狀動脈橫截面與心臟解剖示意")
        else:
            st.warning(f"圖片未找到：{_img_fname}")

    with col_txt:
        _exp = _STENOSIS_EXPLAIN.get((artery, _sev_cat), {})
        if _exp:
            st.markdown(f"### {_exp['title']}")
            st.markdown(f"**供血範圍：** {_exp['territory']}")

            # 風險等級 badge
            _risk_color = {"低－中風險": "green",
                           "中－高風險": "orange",
                           "中－高風險 (NSTEMI 表現)": "orange",
                           "中－高風險，傳導系統受影響": "orange"}.get(
                            _exp['risk'], "red")
            st.markdown(
                f"**風險等級：** :{_risk_color}[**{_exp['risk']}**]"
            )
            st.markdown("---")
            st.markdown(f"**心電圖發現：**\n\n{_exp['finding']}")
            st.markdown("---")
            st.markdown(f"**建議處置：**\n\n{_exp['action']}")

            # ICD-10
            st.markdown(f"**ICD-10 診斷碼：** `{icd10}`")
        else:
            st.info("臨床說明暫無此組合資料。")

# ── 完整結果摘要 ────────────────────────────────────────────────
st.divider()
with st.expander("完整分析摘要", expanded=False):
    summary = {
        "SRC Label": int(label_pred),
        "Is MI": "否" if is_normal else "是",
        "Confidence": f"{confidence:.4f}",
    }
    if not is_normal:
        summary.update({
            "Region": decoded["region"],
            "Artery": decoded["artery"],
            "Location (zh)": decoded["location"],
            "Location (en)": decoded["locationEn"],
            "Severity": decoded["severity"],
            "Severity Label": decoded["severityLabel"],
            "ICD-10": decoded["icd10"],
            "Is Critical": decoded["isCritical"],
        })
    for k, v in summary.items():
        st.text(f"{k:22s}: {v}")
