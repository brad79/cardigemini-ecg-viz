# -*- coding: utf-8 -*-
"""
label_decoder.py
純 Python（無 FastAPI 依賴）版本的 REGION_MAP / SEVERITY_MAP / decode_label。
供 app.py 及其他非 API 模組直接 import。
"""

REGION_MAP = {
    # LAD 供血區域（前壁）— regions 1–6
    1:  {"artery": "LAD", "location": "前壁（近端 LAD）",         "locationEn": "Anterior (Proximal LAD)"},
    2:  {"artery": "LAD", "location": "前壁（中段 LAD）",         "locationEn": "Anterior (Mid LAD)"},
    3:  {"artery": "LAD", "location": "前壁（遠端 LAD）",         "locationEn": "Anterior (Distal LAD)"},
    4:  {"artery": "LAD", "location": "前間隔",                   "locationEn": "Anteroseptal"},
    5:  {"artery": "LAD", "location": "前外側壁",                 "locationEn": "Anterolateral"},
    6:  {"artery": "LAD", "location": "心尖部",                   "locationEn": "Apical"},
    # LCx 供血區域（外側壁）— regions 7–13；7/9/11 為 LAD–LCx 邊界，12 為 LCx–RCA 邊界
    7:  {"artery": "LCx", "location": "外側壁（LAD–LCx 邊界）",  "locationEn": "Lateral (LAD-LCx border)"},
    8:  {"artery": "LCx", "location": "外側壁（中段 LCx）",       "locationEn": "Lateral (Mid LCx)"},
    9:  {"artery": "LCx", "location": "後外側壁（LAD–LCx 邊界）","locationEn": "Posterolateral (LAD-LCx border)"},
    10: {"artery": "LCx", "location": "後壁",                     "locationEn": "Posterior"},
    11: {"artery": "LCx", "location": "下外側壁（LAD–LCx 邊界）","locationEn": "Inferior Lateral (LAD-LCx border)"},
    12: {"artery": "LCx", "location": "下後壁（LCx–RCA 邊界）",  "locationEn": "Inferior Posterior (LCx-RCA border)"},
    13: {"artery": "LCx", "location": "下後外側壁",               "locationEn": "Inferior Posterior Lateral"},
    # RCA 供血區域（下壁）— regions 14–26；19 為 RCA–LAD 邊界，22 為 RCA–LCx 邊界
    14: {"artery": "RCA", "location": "下壁（近端 RCA）",         "locationEn": "Inferior (Proximal RCA)"},
    15: {"artery": "RCA", "location": "下壁（中段 RCA）",         "locationEn": "Inferior (Mid RCA)"},
    16: {"artery": "RCA", "location": "下壁（遠端 RCA）",         "locationEn": "Inferior (Distal RCA)"},
    17: {"artery": "RCA", "location": "下後壁",                   "locationEn": "Inferior Posterior"},
    18: {"artery": "RCA", "location": "下後外側壁",               "locationEn": "Inferior Posterior Lateral"},
    19: {"artery": "RCA", "location": "後壁（RCA–LAD 邊界）",    "locationEn": "Posterior (RCA-LAD border)"},
    20: {"artery": "RCA", "location": "後外側壁",                 "locationEn": "Posterior Lateral"},
    21: {"artery": "RCA", "location": "下外側壁",                 "locationEn": "Inferior Lateral"},
    22: {"artery": "LCx", "location": "右心室壁（RCA–LCx 邊界）","locationEn": "Right Ventricular (RCA-LCx border)"},
    23: {"artery": "RCA", "location": "下間隔",                   "locationEn": "Inferior Septal"},
    24: {"artery": "RCA", "location": "下壁（RCA＋PDA）",         "locationEn": "Inferior (RCA + PDA)"},
    25: {"artery": "RCA", "location": "後壁（RCA 優勢）",         "locationEn": "Posterior (RCA dominant)"},
    26: {"artery": "RCA", "location": "下後壁（RCA 優勢）",       "locationEn": "Inferoposterior (RCA dominant)"},
}

SEVERITY_MAP = {
    1: {"label": "輕微缺血",            "labelEn": "Mild ischemia",              "isCritical": False},
    2: {"label": "輕度缺血",            "labelEn": "Mild-moderate ischemia",     "isCritical": False},
    3: {"label": "中度缺血（NSTEMI）",  "labelEn": "Moderate ischemia (NSTEMI)", "isCritical": False},
    4: {"label": "重度缺血（STEMI）",   "labelEn": "Severe ischemia (STEMI)",    "isCritical": True},
    5: {"label": "完全阻塞（心肌梗塞）","labelEn": "Complete occlusion (MI)",    "isCritical": True},
}

ARTERY_ICD10 = {"LAD": "I21.0", "LCx": "I21.2", "RCA": "I21.1"}


def decode_label(label: int) -> dict:
    """將 SRC 輸出標籤（1–130）解碼為缺血區域與嚴重度
    編碼規則：
      Labels   1– 26：Level 1，Region 1–26
      Labels  27– 52：Level 2，Region 1–26
      Labels  53– 78：Level 3，Region 1–26
      Labels  79–104：Level 4，Region 1–26
      Labels 105–130：Level 5，Region 1–26
      Label  131    ：Normal（無 MI）
    """
    region   = (label - 1) % 26 + 1
    severity = (label - 1) // 26 + 1
    r = REGION_MAP.get(region,   {"artery": "Unknown", "location": "未知區域", "locationEn": "Unknown"})
    s = SEVERITY_MAP.get(severity, {"label": "未知",   "labelEn": "Unknown",   "isCritical": False})
    return {
        "region":          region,
        "severity":        severity,
        "artery":          r["artery"],
        "location":        r["location"],
        "locationEn":      r["locationEn"],
        "severityLabel":   s["label"],
        "severityLabelEn": s["labelEn"],
        "icd10":           ARTERY_ICD10.get(r["artery"], "I21.9"),
        "isCritical":      s["isCritical"],
    }
