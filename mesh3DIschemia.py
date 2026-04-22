# -*- coding: utf-8 -*-
"""
mesh3DIschemia.py
真實心臟 PLY 網格 3D 缺血視覺化（移植自 3D_2D plot/GUI_12LeadECG/mainsite/mesh3DIschemia.py）

差異：
- 以模組所在路徑推算 ventrical.ply / EpicPos_C.mat 的絕對路徑
- 回傳 go.Figure（而非 JSON），與 Streamlit plotly_chart 相容
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plyfile import PlyData
from scipy.io import loadmat

# ── 資料檔路徑（相對於本模組向上兩層進入 3D_2D plot）────────────────
_MAINSITE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../3D_2D plot/GUI_12LeadECG/mainsite",
    )
)
_PLY_PATH = os.path.join(_MAINSITE_DIR, "ventrical.ply")
_MAT_PATH = os.path.join(_MAINSITE_DIR, "EpicPos_C.mat")


def mesh3DIschemia(ishemiaNumber: int) -> go.Figure:
    """
    回傳 Plotly Figure（Mesh3d），依 ishemiaNumber 標示缺血區域。

    Parameters
    ----------
    ishemiaNumber : int
        1–130  → 缺血（severity = ishemiaNumber//26+1，region = ishemiaNumber%26）
        0      → 正常（無高亮）
    """
    plydata = PlyData.read(_PLY_PATH)
    EpicPos_C = loadmat(_MAT_PATH)

    kind26 = {
        0:  [0],
        1:  [1],
        2:  [2],
        3:  [0, 1],
        4:  [1, 2],
        5:  [0, 1, 2],
        6:  [3],
        7:  [4],
        8:  [5],
        9:  [3, 4],
        10: [4, 5],
        11: [3, 5],
        12: [3, 4, 5],
        13: [6],
        14: [7],
        15: [8],
        16: [9],
        17: [10],
        18: [6, 7],
        19: [6, 8],
        20: [7, 8],
        21: [9, 10],
        22: [8, 9],
        23: [8, 9, 10],
        24: [6, 7, 8],
        25: [6, 7, 8, 9, 10],
    }

    hazerFactor = np.zeros(256)
    num = int(ishemiaNumber) - 1

    if 0 <= num < 130:
        amp = float(int(num / 26)) * 0.2 + 0.2
        region = int(np.mod(num, 26))
        kind_regions = kind26[region]
        for k in kind_regions:
            cohort = EpicPos_C["EpicPos_C"][k]
            for idx in cohort[0]:
                hazerFactor[idx[0]] = amp

    # ── 讀取網格頂點與面 ──────────────────────────────────────────────
    nr_faces = plydata.elements[1].count
    x_data = plydata.elements[0].data["x"]
    y_data = plydata.elements[0].data["y"]
    z_data = plydata.elements[0].data["z"]
    faces = [plydata["face"][k][0] for k in range(nr_faces)]
    faces_df = pd.DataFrame(faces)

    # ── Plotly Mesh3d ─────────────────────────────────────────────────
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_data,
            y=y_data,
            z=z_data,
            colorbar_title="Ischemia_Level",
            cmax=1.0,
            cmin=0.0,
            colorscale=[
                [0.0, "rgb(252,78,42)"],
                [0.4, "rgb(253,141,60)"],
                [0.5, "rgb(254,178,76)"],
                [0.6, "rgb(254,217,118)"],
                [0.7, "rgb(255,237,160)"],
                [0.8, "rgb(255,255,204)"],
                [1.0, "rgb(255,255,255)"],
            ],
            intensity=hazerFactor,
            i=np.array(faces_df[0]),
            j=np.array(faces_df[1]),
            k=np.array(faces_df[2]),
            showscale=True,
        )
    ])

    AXIS_CONFIG = {
        "showgrid": False,
        "showline": False,
        "ticks": "",
        "title": "",
        "showticklabels": False,
        "zeroline": False,
        "showspikes": False,
        "spikesides": False,
        "showbackground": False,
    }
    LAYOUT = {
        "scene": {f"{dim}axis": AXIS_CONFIG for dim in ("x", "y", "z")},
        "paper_bgcolor": "#fff",
        "hovermode": False,
        "margin": {"l": 0.1, "r": 5, "b": 0.1, "t": 0.1, "pad": 0.1},
    }
    fig.update_layout(
        scene_camera={"eye": {"x": 1.25, "y": 1.25, "z": 1.25},
                      "up": {"x": 0, "y": 0, "z": 1},
                      "center": {"x": 0, "y": 0, "z": 0}},
        **LAYOUT,
    )
    return fig
