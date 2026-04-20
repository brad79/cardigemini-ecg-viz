# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:59:10 2025

@author: BOX
"""
import numpy as np


#  normalized function
def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    mad = np.where(mad == 0, 1.0, mad)   # 防止除以零
    return (x - med) / mad


def extract_jt_features(beat_ecg,Jpos,Tpeak, DC_level=None, minus_data=10,
    dc_tail=20, t_win=10, missing_idx=-1):
    """
    使用 beat_ecg + Jpos + Tpeak (+ optional DC_level) 萃取 JT 相關 5 個特徵

    Parameters
    ----------
    beat_ecg : array-like
        1D: (L,) 單導 1-beat ECG
        2D: (n_leads, L) 多導 1-beat ECG
    Jpos : array-like or scalar
        每導 J point index (0-based)。可為 float（含 nan）或 int
    Tpeak : array-like or scalar
        每導 Tpeak index (0-based)。可為 float（含 nan）或 int
    DC_level : array-like or scalar or None
        若提供，直接用它做 DC 扣除；若 None，會用 tail 平均估 DC
    minus_data : int
        末端裁掉多少點（你原本 minus_data=10）
    dc_tail : int
        用末端多少點估 DC（預設取最後 20 點）
    t_win : int
        在 Tpeak 周圍 ±t_win 內找真正 T 極值（預設 ±10）
    missing_idx : int
        缺值 index 的 sentinel

    Returns
    -------
    feature : ndarray, shape (n_leads, 5)  (單導也會回 (1,5))
        [T_amp, J_amp, JT25_amp, JT50_amp, JT_slope]
    """


    x = np.asarray(beat_ecg, dtype=float)
    if x.ndim == 1:
        x = x[None, :]  # (1, L)

    n_leads, L0 = x.shape
    L = max(1, L0 - int(minus_data))
    sig = x[:, :L]

    J = np.asarray(Jpos, dtype=int).reshape(-1)
    T0 = np.asarray(Tpeak, dtype=int).reshape(-1)

    if J.size == 1 and n_leads > 1:
        J = np.full(n_leads, int(J.item()), dtype=int)
    if T0.size == 1 and n_leads > 1:
        T0 = np.full(n_leads, int(T0.item()), dtype=int)

    if J.size != n_leads or T0.size != n_leads:
        raise ValueError("Jpos / Tpeak 長度需與導程數一致（或提供單一 int）")

    # DC：若沒給，就尾端估
    if DC_level is None:
        tail = min(dc_tail, L)
        dc_used = np.mean(sig[:, -tail:], axis=1) if tail >= 1 else np.zeros(n_leads)
    else:
        dc_used = np.asarray(DC_level, dtype=float).reshape(-1)
        if dc_used.size == 1 and n_leads > 1:
            dc_used = np.full(n_leads, float(dc_used.item()), dtype=float)
        if dc_used.size != n_leads:
            raise ValueError("DC_level 長度需與導程數一致（或提供單一 scalar）")

    minus_dc = sig - dc_used[:, None]

    feature = np.full((n_leads, 5), np.nan, dtype=float)
    final_t_idx = np.full(n_leads, missing_idx, dtype=int)

    def clamp(i):
        return max(0, min(L - 1, int(i)))

    for r in range(n_leads):
        j = J[r]
        t0 = T0[r]

        # 缺值直接跳過
        if j == missing_idx or t0 == missing_idx:
            continue
        if j < 0 or j >= L or t0 < 0 or t0 >= L:
            continue

        # 在 Tpeak ±t_win 內找真正 T 極值（依 Tpeak 當下正負決定找 min/max）
        l = max(0, t0 - t_win)
        u = min(L - 1, t0 + t_win)
        seg = minus_dc[r, l:u + 1]

        if minus_dc[r, t0] < 0:
            t = l + int(np.argmin(seg))
        else:
            t = l + int(np.argmax(seg))

        # 確保 T 在 J 之後
        if t <= j:
            t = clamp(round((j + t0) / 2))
            if t <= j:
                t = min(L - 1, j + 1)

        final_t_idx[r] = t

        jt25 = clamp(round(j + (t - j) * 0.25))
        jt50 = clamp(round(j + (t - j) * 0.50))

        T_amp = minus_dc[r, t]
        J_amp = minus_dc[r, j]
        JT25_amp = minus_dc[r, jt25]
        JT50_amp = minus_dc[r, jt50]

        denom = (t - j) * 2.0
        JT_slope = (T_amp - J_amp) / denom if denom != 0 else 0.0

        feature[r, :] = [T_amp, J_amp, JT25_amp, JT50_amp, JT_slope]

    feature = robust_zscore(feature)
    
    return feature
