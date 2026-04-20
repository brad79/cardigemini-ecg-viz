# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:28:25 2025

@author: BOX
"""

import numpy as np


def _trapezium_ares(data: np.ndarray, xm: int, xr: int):
    """
    MATLAB:
    function [Pt,AreaM, A,T] = TrapeziumAres(data,xm,xr)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    xm = int(round(xm))
    xr = int(round(xr))
    xm = max(0, min(n - 1, xm))
    xr = max(0, min(n - 1, xr))
    if xr < xm:
        xr = xm

    T = np.arange(xm, xr + 1, dtype=int)
    A = np.zeros(len(T), dtype=float)
    # A(k) = 0.5*(data(xm)-data(i))*(2*xr-i-xm);
    dxm = data[xm]
    for k, i in enumerate(T):
        A[k] = 0.5 * (dxm - data[i]) * (2 * xr - i - xm)

    idx = int(np.argmax(np.abs(A)))
    AreaM = float(np.max(np.abs(A)))
    Pt = int(T[idx])
    return Pt, AreaM, A, T


def findTpoint(data, fs, RRI, Rpeak=None):
    """
    輸入：
      data: 1-beat ECG 波形 (1D array)
      fs:   sampling rate (Hz)
      RRs:  RR interval(秒)

    輸出（全為 0-based index）：
      Tpeak, Tend, Tpolar
    """
    if Rpeak is None:
          Rpeak = int(round(0.100 * fs))  # fs=500 -> 50
          
    s0 = np.asarray(data, dtype=float).reshape(-1)
    N = len(s0)
    if N < 10:
        return None, None, 0, 0, None

    fs = float(fs)
    if RRI is None or (isinstance(RRI, float) and np.isnan(RRI)):
        RRs = 0.8 * fs
    RRs = float(RRI*fs)

    # ---- 常數/參數（對應 MATLAB 預設） ----
    TsmallThd = 0.03
    fratio = fs / 250.0

    mthrld = 6.5
    swin = int(round(fs * 0.128))  # default
    if swin < 1:
        swin = 1

    s = s0.copy()

    ptwin = int(np.ceil(fs * 0.016))  # ceil(4*fratio)
    if ptwin < 1:
        ptwin = 1


    # ---- T-wave end search interval parameters ----
    ald = 0.15
    bld = int(round(fs * 0.08))       # round(fs*0.08)
    alu = 0.7
    blu = -int(round(fs * 0.036))     # -round(fs*0.036)
    ard = 0.0
    brd = int(round(fs * 0.28))       # round(fs*0.28)
    aru = 0.20
    bru = int(round(fs * 0.404))      # round(fs*0.404)

    if RRs < 220 * fratio:
        minRtoT = int(np.floor(ald * RRs + bld))
        maxRtoT = int(np.ceil(alu * RRs + blu))
    else:
        minRtoT = int(np.floor(ard * RRs + brd))
        maxRtoT = int(np.ceil(aru * RRs + bru))

    leftbound = Rpeak + minRtoT
    rightbound = Rpeak + maxRtoT

    rightbound = min(rightbound, N - 1 - ptwin)
    leftbound = min(leftbound, rightbound)

    # clamp
    leftbound = max(leftbound, ptwin)
    rightbound = max(rightbound, leftbound)

    # ---- Compute the area indicator ----
    # MATLAB areavalue(kT) 只在區間上有值
    T_range = np.arange(leftbound, rightbound + 1, dtype=int)
    areavalue = np.zeros(len(T_range), dtype=float)

    for idx, kT in enumerate(T_range):
        # cutlevel = mean(s((kT-ptwin):(kT+ptwin)))
        a = kT - ptwin
        b = kT + ptwin
        cutlevel = float(np.mean(s[a : b + 1]))

        # corsig = s((kT-swin+1):kT) - cutlevel
        start = kT - swin + 1
        if start < 0:
            start = 0
        corsig = s[start : kT + 1] - cutlevel
        areavalue[idx] = float(np.sum(corsig))

    # Tval = areavalue(leftbound:rightbound) 已經是局部陣列
    Tval = areavalue

    # MATLAB:
    # if isnumeric(mthrld) | mthrld(1)=='p': [dum, maxind] = max(Tval)
    # if isnumeric(mthrld) | mthrld(1)=='n': [duminv, maxindinv] = max(-Tval)
    dum = float(np.max(Tval))
    maxind = int(np.argmax(Tval))
    duminv = float(np.max(-Tval))
    maxindinv = int(np.argmax(-Tval))

    # MATLAB 的 mthrld 預設是 numeric => 走 else 分支
    # 選擇較遠的位置 + 閾值判斷
    if maxind < maxindinv:
        leftind, rightind = maxind, maxindinv
        leftdum, rightdum = dum, duminv
    else:
        leftind, rightind = maxindinv, maxind
        leftdum, rightdum = duminv, dum

    if leftdum > mthrld * rightdum:
        pick = leftind
    else:
        pick = rightind

    Tend1 = int(leftbound + pick)  # 0-based

    # ---- Tpeak ----
    bld2 = int(round(fs * 0.08))
    brd2 = int(round(fs * 0.28))

    minRtoT2 = int(np.floor(0.135 * RRs + bld2))
    if minRtoT2 > brd2:
        minRtoT2 = brd2

    leftbound2 = int(Rpeak + minRtoT2 - bld2)
    rightbound2 = int(round(Tend1 - fs * 0.02))

    leftbound2 = max(0, min(N - 1, leftbound2))
    rightbound2 = max(0, min(N - 1, rightbound2))
    if rightbound2 <= leftbound2 + 3:
        # 範圍太小，無法找 Tpeak
        Tpeak = None
        flag1 = 0
        Tend2 = None
        Tpolar = 0
        return Tend1, Tend2, flag1, Tpolar, Tpeak

    s1 = s0 - s0[Tend1]

    # MATLAB: Tval = s1(leftbound2+1:rightbound2-1)
    start = leftbound2 + 1
    end = rightbound2 - 1
    if end <= start:
        return Tend1, None, 0, 0, None

    Tseg = s1[start:end]
    if Tseg.size < 5:
        return Tend1, None, 0, 0, None

    # 正向/反向峰
    maxind_p = int(np.argmax(Tseg))
    dum_p = float(abs(Tseg[maxind_p]))
    maxind_n = int(np.argmax(-Tseg))
    dum_n = float(abs(Tseg[maxind_n]))

    # 邊界排除（對應 MATLAB 的一堆 if）
    def _zero_if_boundary(idx0, length):
        return (idx0 >= length - 2) or (idx0 <= 1)

    if _zero_if_boundary(maxind_p, len(Tseg)):
        dum_p = 0.0
    if _zero_if_boundary(maxind_n, len(Tseg)):
        dum_n = 0.0

    # 選擇較遠的位置
    if maxind_p < maxind_n:
        leftind, rightind = maxind_p, maxind_n
        leftdum, rightdum = dum_p, dum_n
    else:
        leftind, rightind = maxind_n, maxind_p
        leftdum, rightdum = dum_n, dum_p

    mthrld2 = 4.0
    if leftdum > mthrld2 * rightdum and leftdum > 0.03:
        pick2 = leftind
    else:
        pick2 = rightind

    Tpeak = int(start + pick2)  # 0-based

    flag1 = 1
    if abs(s0[Tpeak] - s0[Tend1]) < TsmallThd:
        flag1 = 0

    # ---- Tend2 (TrapeziumAres) ----
    swin_end = int(round(fs * 0.01))
    swin2_end = int(round(RRs * 0.22))

    xm = int(round(Tpeak + swin_end))
    xr = int(round(xm + swin2_end))

    # MATLAB: if xr >= length(s0)-swin => xr = length(s0)-swin
    # 0-based 最後可用 index = N-1-swin_end
    xr = min(xr, N - 1 - swin_end)
    xm = max(0, min(N - 1, xm))
    xr = max(xm, xr)

    Tend, AreaM, Area, Ts = _trapezium_ares(s0, xm, xr)

    T_amp_valid = 1
    if abs(s0[Tpeak] - s0[Tend]) < TsmallThd:
        T_amp_valid = 0

    return Tpeak, Tend, T_amp_valid
    #plt.plot(data),plt.plot(Tpeak,data[Tpeak],'o'),plt.plot(Tend,data[Tend],'o')

def refine_Tend_by_multileads(Tend, T_amp_valid, fs, dist_sec=0.08):
    """
    多導 T-wave end 修正函式

    Parameters
    ----------
    Tend : array-like, shape (n_leads,)
        每導的 T-wave end index
    T_amp_valid : array-like, shape (n_leads,)
        T-wave 振幅是否有效 (1=正常, 0=太小)
    fs : int or float
        sampling frequency (Hz)
    dist_sec : float
        容許 Tend 與平均值的最大差距（秒），預設 0.08s

    Returns
    -------
    Tend_new : ndarray
        修正後的 Tend
    meanTend : float
        用於修正的平均 Tend
    """

    Tend = np.asarray(Tend, dtype=int)
    T_amp_valid = np.asarray(T_amp_valid, dtype=int)

    if Tend.shape != T_amp_valid.shape:
        raise ValueError("Tend 與 T_amp_valid 維度必須相同")

    Tdist = fs * dist_sec  # sample-based threshold

    # ---- 1️⃣ 計算平均 Tend（只用 T_amp_valid == 1） ----
    valid_idx = (T_amp_valid == 1) & np.isfinite(Tend)

    if np.sum(valid_idx) == 0:
        # 極端情況：全部 T 波都太小 → 用整體中位數
        meanTend = int(np.nanmedian(Tend))
    else:
        meanTend = int(np.nanmedian(Tend[valid_idx]))

    # ---- 2️⃣ 建立修正後 Tend ----
    Tend_new = Tend.copy()

    # (a) T 太小 → 用平均值
    Tend_new[T_amp_valid == 0] = meanTend

    # (b) 距離平均值太遠 → 用平均值
    outlier_idx = np.abs(Tend_new - meanTend) > Tdist
    Tend_new[outlier_idx] = meanTend

    return Tend_new

def _smooth_ma(x, w):
    """等效 MATLAB smooth(x, w)：moving average"""
    w = int(round(w))
    if w < 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def reTpeak(signal, Tpeak, Tend, RRI, fs, Rpeak, minTval):
    """
    Python 版 reTpeak（不畫圖）

    Parameters
    ----------
    signal : ndarray, shape (n_leads, L)
        每導 1-beat ECG
    Tpeak : array-like, shape (n_leads,)
        初始 Tpeak (0-based)
    Tend : array-like, shape (n_leads,)
        T-wave end (0-based)
    RRI : float
        RR interval (秒)
    fs : float
        sampling frequency (Hz)
    Rpeak : int
        R peak index in beat window
    minTval : float
        T-wave amplitude threshold

    Returns
    -------
    Tpeak2 : ndarray, shape (n_leads,)
        修正後 Tpeak
    """

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("signal 必須是 (n_leads, L)")

    n_leads, L = signal.shape
    Tpeak = np.asarray(Tpeak, dtype=int)
    Tend = np.asarray(Tend, dtype=int)

    Tpeak2 = Tpeak.copy()

    # ---- 參數（對應 MATLAB）----
    ald = 0.125
    bs = int(round(fs * 0.02))
    bld = int(round(fs * 0.08))
    bld2 = int(round(fs * 0.20))
    brd = int(round(fs * 0.28))

    # minRtoT 計算
    minRtoT = int(np.floor(ald * RRI * fs + bld))
    if minRtoT > brd:
        minRtoT = brd
    if minRtoT < bld2:
        minRtoT = bld2

    leftbound = int(Rpeak + minRtoT - bld)
    leftbound = max(0, leftbound)

    for r in range(n_leads):
        ecg1 = signal[r, :]

        if not np.isfinite(Tend[r]):
            continue

        rightbound = int(Tend[r] + round(fs * 0.015))
        if rightbound > L - bs - 1:
            rightbound = L - bs - 1

        if rightbound <= leftbound + 2:
            Tpeak2[r] = int(round((leftbound + rightbound) / 2))
            continue

        # s1 = ecg1 - ecg1(Tend(r))
        s1 = ecg1 - ecg1[int(Tend[r])]
        s1 = _smooth_ma(s1, bs)

        # ---- 尋找局部極值（忠實 MATLAB for-loop）----
        dum = -1e10
        duminv = 1e10
        maxind = leftbound
        maxindinv = leftbound

        for i in range(rightbound - 1, leftbound, -1):
            if (s1[i] > s1[i - 1] and s1[i] >= s1[i + 1] and
                s1[i] > s1[i - bs] and s1[i] >= s1[i + bs]):
                dum = s1[i]
                maxind = i
                break

        for i in range(rightbound - 1, leftbound, -1):
            if (s1[i] < s1[i - 1] and s1[i] <= s1[i + 1] and
                s1[i] < s1[i - bs] and s1[i] <= s1[i + bs]):
                duminv = s1[i]
                maxindinv = i
                break

        if dum == -1e10:
            dum = 0
            maxind = maxindinv
        elif duminv == 1e10:
            duminv = 0
            maxindinv = maxind

        # 選較遠的
        if maxind < maxindinv:
            leftind, rightind = maxind, maxindinv
            leftdum, rightdum = abs(dum), abs(duminv)
        else:
            leftind, rightind = maxindinv, maxind
            leftdum, rightdum = abs(duminv), abs(dum)

        mthrld = 4
        if leftdum > mthrld * rightdum and leftdum > minTval:
            pick = leftind
            dumv = leftdum
        else:
            pick = rightind
            dumv = rightdum

        if dumv > minTval:
            Tpeak2[r] = pick
        else:
            Tpeak2[r] = int(round((leftbound + rightbound) / 2))

    return Tpeak2
def refine_Tend_if_too_close(signal,Tpeak,Tend,fs,RRI,dist_sec=0.04):
    """
    若 Tpeak 與 Tend 距離過近，使用 Trapezium Areas 重新估 Tend

    Parameters
    ----------
    signal : ndarray, shape (n_leads, L)
        每導 1-beat ECG
    Tpeak : array-like, shape (n_leads,)
        T-wave peak (0-based)
    Tend : array-like, shape (n_leads,)
        T-wave end (0-based)
    fs : float
        sampling frequency (Hz)
    RRI : float
        RR interval (秒)
    dist_sec : float
        Tpeak-Tend 最小容許距離（秒），預設 0.04s

    Returns
    -------
    Tend_new : ndarray
        修正後的 Tend
    """

    signal = np.asarray(signal, dtype=float)
    Tpeak = np.asarray(Tpeak, dtype=int)
    Tend = np.asarray(Tend, dtype=int)
    n_leads, L = signal.shape
    Tend_new = Tend.copy()
    TEdist = fs * dist_sec
    # 找 Tend - Tpeak < TEdist 的導程
    idx = np.where((Tend_new - Tpeak) < TEdist)[0]
    if len(idx) == 0:
        return Tend_new

    swin = int(round(fs * 0.01))
    swin2 = int(round(RRI * fs * 0.22))

    for r in idx:
        if not np.isfinite(Tpeak[r]):
            continue

        xm = int(round(Tpeak[r] + swin))
        xr = int(round(xm + swin2))

        s0 = signal[r, :]

        # MATLAB: if xr>=length(s0)-swin
        if xr >= len(s0) - swin:
            xr = len(s0) - swin - 1

        xm = max(0, min(len(s0) - 1, xm))
        xr = max(xm, min(len(s0) - 1, xr))

        # 重新用 Trapezium Areas 找 Tend
        Tend_new[r], _, _, _ = _trapezium_ares(s0, xm, xr)

    return Tend_new

def compute_DC_level(signal, Tend, fs):
    """
    計算每一導程在 T-end 之後的 DC level（支援 1D / 2D）

    Parameters
    ----------
    signal : array-like
        1D: (L,) 單導 1-beat ECG
        2D: (n_leads, L) 或 (L, n_leads) 多導 1-beat ECG
    Tend : float / int 或 array-like
        1D 對應：單一 Tend index
        2D 對應：每導 Tend index 陣列 (shape=(n_leads,))
    fs : float
        sampling frequency (Hz)

    Returns
    -------
    DC_level : float 或 ndarray(shape=(n_leads,))
        單導回傳 float；多導回傳每導 DC level
    """
    x = np.asarray(signal, dtype=float)
    dcwinR = int(round(fs * 0.05))  # 50 ms
    dcwinL = int(round(fs * 0.01))  # 10 ms

    # ---------- 單導（1D） ----------
    if x.ndim == 1:
        L = x.shape[0]
        if not np.isfinite(Tend):
            return np.nan

        idx = int(round(float(Tend)))
        idx = max(0, min(L - 1, idx))

        lidx = min(L - 1, idx + dcwinL)
        ridx = min(L - 1, idx + dcwinR)

        if ridx <= lidx:
            return np.nan

        return float(np.mean(x[lidx:ridx + 1]))

    # ---------- 多導（2D） ----------
    if x.ndim != 2:
        raise ValueError("signal 必須是 1D 或 2D array")

    # 容錯：若傳入 (L, n_leads) 就轉為 (n_leads, L)
    max_leads = 68
    if x.shape[0] > max_leads and x.shape[1] <= max_leads:
        x = x.T

    n_leads, L = x.shape
    Tend = np.asarray(Tend, dtype=float)
    if Tend.shape[0] != n_leads:
        raise ValueError("Tend 長度必須與 signal 的導程數一致")

    DC_level = np.full(n_leads, np.nan, dtype=float)

    for r in range(n_leads):
        if not np.isfinite(Tend[r]):
            continue

        idx = int(round(Tend[r]))
        idx = max(0, min(L - 1, idx))

        lidx = min(L - 1, idx + dcwinL)
        ridx = min(L - 1, idx + dcwinR)

        if ridx <= lidx:
            continue

        DC_level[r] = np.mean(x[r, lidx:ridx + 1])

    return DC_level


def findTpos_multileads(avgECG, fs, RRI=None, Rpeak=None):
    """
    avgECG: shape = (n_leads, L) 或 (L, n_leads)
             每個 lead 一段 1-beat 波形
    fs: 取樣率(Hz)
    RRI: RR interval (秒)  
    Rpeak: R 峰在 1-beat window 內的 index；若 None，預設 100ms 位置    
    """
    max_leads = 68   #max_leads: 最多支援導程數（預設 68）
    x = np.asarray(avgECG, dtype=float)
    # 預設：Rpeak 在 100ms 位置（若你的 beat align 是用 PR_interval=100ms）
    if Rpeak is None:
        Rpeak = int(round(0.100 * fs))  # fs=500 -> 50
    if RRI is None :
       RRI = 0.8 * fs

    # ---------- 單導（1D） ----------
    if x.ndim == 1:
       tpeak, tend, t_amp_valid  = findTpoint(x, fs, RRI, Rpeak=Rpeak)
       DC_level = compute_DC_level(avgECG, tend, fs)
       
       return tpeak,tend,DC_level


    if x.ndim != 2:
        raise ValueError("avgECG 必須是 2D array")

    # 容錯：如果給的是 (L, n_leads) 就轉成 (n_leads, L)
    if x.shape[0] > max_leads and x.shape[1] <= max_leads:
        x = x.T

    n_leads, L = x.shape
    if n_leads > max_leads:
        raise ValueError(f"導程數 n_leads={n_leads} 超過 max_leads={max_leads}")

    Tend = np.full(n_leads, 0, dtype=int)
    T_amp_valid = np.full(n_leads, 0, dtype=int)
    Tpeak  = np.full(n_leads, 0, dtype=int)

    for r in range(n_leads):
        ecg1 = x[r, :]

        # 你的 twaveend_1beat 回傳順序是 (Tpeak, Tend, T_amp_valid)
        tpeak, tend, t_amp_valid  = findTpoint(ecg1, fs, RRI, Rpeak=Rpeak)

        Tpeak[r]  = np.nan if tpeak  is None else tpeak
        Tend[r]   = np.nan if tend   is None else tend
        T_amp_valid[r] = 0 if t_amp_valid is None else t_amp_valid

    Tend_refined = refine_Tend_by_multileads(Tend, T_amp_valid, fs)
    Tpeak2 = reTpeak(signal=avgECG,Tpeak=Tpeak, Tend=Tend_refined, RRI=RRI, fs=fs, Rpeak=Rpeak, minTval=0.03)
    Tend_new = refine_Tend_if_too_close(avgECG,Tpeak,Tend_refined,fs,RRI,dist_sec=0.04)
    DC_level = compute_DC_level(avgECG, Tend_new, fs)
    return  Tpeak2, Tend_new, DC_level

#------------------------------------------------------------------------

def Jlinefitting(QRS, index, fs):
    """
    MATLAB: [Jpos,theta1,theta2,theta,T] = Jlinefitting(QRS,index,fs)
    回傳 Jpos 與 (theta1, theta2, theta, T) 方便你除錯/畫圖（你可不用）
    """
    QRS = np.asarray(QRS, dtype=float)
    n = len(QRS)

    win = int(round(fs * 0.1))
    pt = 5

    # T = index:index+win (MATLAB) -> Python inclusive end
    start = int(index)
    end = int(index + win)
    start = max(0, min(n - 1, start))
    end = max(start, min(n - 1, end))

    T = np.arange(start, end + 1, dtype=int)
    nT = len(T)

    theta1 = np.zeros(nT, dtype=float)
    theta2 = np.zeros(nT, dtype=float)

    # time vector and centering
    t = (np.arange(1, pt + 2) / fs) * 10.0
    x = t - np.mean(t)
    x_denom = np.sum(x ** 2)
    if x_denom == 0:
        x_denom = 1e-12

    # 逐點計算 slope -> atan
    for k, i in enumerate(T):
        # segA: QRS(i-pt:i)  segC: QRS(i:i+pt)
        a0 = i - pt
        a1 = i
        c0 = i
        c1 = i + pt

        # 邊界保護：不足就用截斷（比直接錯誤更穩）
        if a0 < 0 or c1 >= n:
            # 若靠近邊界，先跳過（theta=0）
            continue

        y1 = QRS[a0:a1 + 1].copy()
        y1 = y1 - np.mean(y1)
        slope1 = np.sum(x * y1) / x_denom
        theta1[k] = np.degrees(np.arctan(slope1))

        y2 = QRS[c0:c1 + 1].copy()
        y2 = y2 - np.mean(y2)
        slope2 = np.sum(x * y2) / x_denom
        theta2[k] = np.degrees(np.arctan(slope2))

    theta = np.zeros(nT, dtype=float)

    cond1 = (theta1 > 0) & (theta1 < 90) & (theta1 > theta2) & (np.abs(theta2) < 50)
    cond2 = (theta1 > -30) & (theta1 < theta2) & (np.abs(theta2) < 50)
    mask = cond1 | cond2
    theta[mask] = np.abs(theta1[mask] - theta2[mask])

    # MATLAB 從尾端找局部極大
    idx = 0
    for i in range(nT - 2, 0, -1):
        if (theta[i] > theta[i - 1] and theta[i] > theta[i + 1] and
            theta[i] > 0 and theta[i] > 7 and theta[i + 1] > 0):
            idx = i
            break

    if idx == 0:
        for i in range(1, nT - 1):
            if (theta[i] > theta[i - 1] and theta[i] > theta[i + 1] and
                theta[i] > 0 and theta[i + 1] > 0):
                idx = i
                break

    Jpos = int(T[idx]+1)
    return Jpos, theta1, theta2, theta, T


def findJpoint(data, fs, Rpeak=None, L2=None):
    """
    MATLAB: function [Jpos] = findJpoint(data,fs,Rpeak,L2,showfig)

    Inputs
    ------
    data : 1D array
        單一導程的 1-beat ECG
    fs : float
        取樣率 (Hz)
    Rpeak : int or None
        R peak 在 beat window 的位置（0-based）。None -> 預設 50
    L2 : int or None
        最遠的 QRS 結束點（0-based）。None -> round(fs*0.25)

    Returns
    -------
    Jpos : int
        J point index (0-based)
    sp   : int
        找到的 QRS offset 特徵點（你原碼的 sp）
    """

    s0 = np.asarray(data, dtype=float).reshape(-1)
    N = len(s0)
    if N < 20:
        return None, None

    if Rpeak is None:
        Rpeak = 50
    Rpeak = int(Rpeak)

    if L2 is None:
        L2 = int(round(fs * 0.25))
    L2 = int(L2)

    # L = round(fs*0.05) 這個變數在你貼的程式後面未使用
    # 1) 1階微分 + normalize
    datas = _smooth_ma(s0, 15)   # MATLAB smooth(data,15)'
    dT1 = np.concatenate([[0.0], np.diff(datas)])
    m = np.max(np.abs(dT1))
    if m == 0:
        m = 1e-12
    dT1 = dT1 / m

    # 2) 找最大 R peak 範圍
    R1 = 0
    R2 = int(round(fs * 0.2))
    R2 = min(R2, N - 1)

    if L2 < Rpeak:
        L2 = Rpeak + int(round(fs * 0.08))
    L2 = min(L2, N - 4)  # 因為下面會用 i+3

    # [~,idx] = max(abs(dT1(R1:R2))); Ridx = R1+idx-1;
    seg = np.abs(dT1[R1:R2 + 1])
    idx_local = int(np.argmax(seg))
    Ridx = R1 + idx_local
    if Ridx > Rpeak:
        Ridx = Rpeak

    # 找 QRS 的最大值/最小值 sp
    level = 0.08
    level2 = 0.35
    sp = Ridx

    # for i = L2:-1:Ridx
    for i in range(L2, Ridx - 1, -1):
        # 要確保 i-3 >=0, i+3 < N
        if i - 3 < 0 or i + 3 >= N:
            continue

        temp1 = (dT1[i] > dT1[i - 1] and
                 dT1[i] > dT1[i - 3] + level and
                 dT1[i] >= dT1[i + 1] and
                 dT1[i] >= dT1[i + 3] + level)

        temp2 = (dT1[i] > dT1[i - 1] and
                 dT1[i] >= dT1[i + 1] and
                 dT1[i] > level2)

        if temp1 or temp2:
            sp = i
            break

        temp3 = (dT1[i] < dT1[i - 1] and
                 dT1[i] < dT1[i - 3] - level and
                 dT1[i] <= dT1[i + 1] and
                 dT1[i] <= dT1[i + 3] - level)

        temp4 = (dT1[i] < dT1[i - 1] and
                 dT1[i] <= dT1[i + 1] and
                 dT1[i] < -level2)

        if temp3 or temp4:
            sp = i
            break

    if sp < Ridx and np.isclose(abs(dT1[sp]), 1.0):
        sp = sp + 8
        sp = min(sp, N - 1)

    # sp 後面找 J point（線段角度法）
    Jpos, *_ = Jlinefitting(s0, sp, fs)
    return Jpos, sp

def findJpos_multileads(signal, fs, Rpeak=None):
    """
    signal : array-like
       1D: (L,) 單導
       2D: (n_leads, L) 或 (L, n_leads)
    """   
       
    max_leads = 68
    x = np.asarray(signal, dtype=float)
    # ---------- 單導（1D） ----------
    if x.ndim == 1:
       jpos, _sp = findJpoint(x, fs, Rpeak=Rpeak)
       return jpos

    # ---------- 多導（2D） ----------
    if x.ndim != 2:
        raise ValueError("signal 必須是 2D array")
    
    # 容錯：若是 (L, n_leads) 就轉成 (n_leads, L)
    if x.shape[0] > max_leads and x.shape[1] <= max_leads:
        x = x.T
        
    n_leads, Lsig = x.shape        
    # 預設：Rpeak 在 100ms 位置（若你的 beat align 是用 PR_interval=100ms）

        
    Jdist = int(round(fs * 0.06))  # 60 ms
    # ---------- 第一次找 Jpoint ----------
    Jpoint  = np.full(n_leads, 0, dtype=int)

    for r in range(n_leads):
        ecg1 = x[r, :]
        jpos, _sp = findJpoint(ecg1, fs, Rpeak=Rpeak)
        if jpos is not None:
            Jpoint[r] = jpos
    # median omitnan
    if np.all(np.isnan(Jpoint)):
        return Jpoint
    meanJpoint = int(round(np.nanmedian(Jpoint)))
    # ---------- outlier 重新估一次 ----------
    idx_outliers = np.where(np.abs(Jpoint - meanJpoint) > Jdist)[0]

    Jpoint_refined = Jpoint.copy()
    L2_ref = meanJpoint + Jdist
    # 防呆：L2_ref 不要超出訊號範圍
    L2_ref = max(0, min(Lsig - 1, int(L2_ref)))

    for r in idx_outliers:
        ecg1 = signal[r, :]
        jpos, _sp = findJpoint(ecg1, fs, Rpeak=Rpeak, L2=L2_ref)
        if jpos is not None:
            Jpoint_refined[r] = jpos

    return Jpoint_refined


def process_JT_point(avgECG, fs, RRI, Rpeak=None)  :
    """
    一次完成多導（或單導）JT 相關點位偵測：
      - J point (Jpos)
      - T peak (Tpeak)
      - T end  (Tend)
      - DC level (DC_level)

    Parameters
    ----------
    avgECG : array-like
        1D: (L,) 單導 1-beat ECG
        2D: (n_leads, L) 或 (L, n_leads) 多導 1-beat ECG
    fs : float
        取樣率 (Hz)；建議已統一成 500Hz
    RRI : float
        RR interval（秒）。若你是 sample，請先換算成秒：RRI_sec = RRI_samples / fs
    Rpeak : int or None
        R peak 在 1-beat window 中的位置（0-based index）。
        若 None，預設以 PR_interval=100ms 對齊假設，設定 Rpeak = round(0.1*fs)
        (fs=500 -> 50)

    Returns
    -------
    Jpos : float or ndarray
        每導 J point index（0-based）；單導回 float，多導回 ndarray(shape=(n_leads,))
    Tpeak : float or ndarray
        每導 T peak index（0-based）
    Tend : float or ndarray
        每導 T end index（0-based）
    DC_level : float or ndarray
        每導 DC level（mV；取 Tend 後 10~50ms 的平均）

    Notes
    -----
    - 需要你已經有兩個函式：
        1) findTpos_multileads(avgECG, fs, RRI, Rpeak) -> (Tpeak, Tend, DC_level)
        2) findJpos_multileads(avgECG, fs, Rpeak)      -> Jpos
    - 本函式不負責畫圖；點位都是 index（樣本點）
    """
    # ---------- 輸入轉 ndarray，支援 1D/2D ----------
    x = np.asarray(avgECG, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("avgECG 必須是 1D(單導) 或 2D(多導)")
        
    # ---------- 預設 Rpeak：假設 beat window 以 PR_interval=100ms 對齊 ----------
    if Rpeak is None:
        Rpeak = int(round(0.100 * fs))  # fs=500 -> 50
    Rpeak = int(Rpeak)
 
    # ---------- 找 T 波相關點位（Tpeak, Tend, DC_level）----------            
    Tpeak, Tend, DC_level = findTpos_multileads(avgECG, fs, RRI, Rpeak)
    # ---------- 找 J point ----------
    Jpos = findJpos_multileads(avgECG, fs, Rpeak)
    
    return Jpos,Tpeak,Tend, DC_level
    