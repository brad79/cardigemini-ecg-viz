# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:04:05 2025

@author: BOX
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from math import gcd
from scipy.signal import resample_poly



def _smooth_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(round(window))
    if window < 1:
        window = 1
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

def derivative_based_method2(ecg: np.ndarray, fs: int, window: int):
    """
    MATLAB:
    function [postecg3,ecg3]=derivative_based_method2(ecg,fs,window)
    """
    ecg = np.asarray(ecg, dtype=float)

    # normalization: ecg = ecg / max(abs(ecg))
    m = np.max(np.abs(ecg)) if ecg.size else 1.0
    if m == 0:
        m = 1.0
    ecg = ecg / m

    n = len(ecg)

    # first derivative
    ecgd1 = np.zeros(n, dtype=float)
    # ecgd1(2:end-1) = ecg(3:end)-ecg(1:end-2);
    if n >= 3:
        ecgd1[1:-1] = ecg[2:] - ecg[:-2]

    # second derivative
    ecgd2 = np.zeros(n, dtype=float)
    # ecgd2(3:end-2) = ecg(5:end)-2*ecg(3:end-2)+ecg(1:end-4);
    if n >= 5:
        ecgd2[2:-2] = ecg[4:] - 2 * ecg[2:-2] + ecg[:-4]

    # combination result(first*1.3 + second*1.1)
    ecg2 = 1.3 * np.abs(ecgd1) + 1.1 * np.abs(ecgd2)

    # moving average smooth(ecg2, round(window))
    w = int(round(window))
    if w < 1:
        w = 1
    kernel = np.ones(w, dtype=float) / w
    ecg3 = np.convolve(ecg2, kernel, mode="same")

    # lowpass 5 Hz then filtfilt
    b2, a2 = butter(2, 5 / (fs / 2), btype="low")
    postecg3 = filtfilt(b2, a2, ecg3, padtype='odd',  padlen=3*(max(len(b2),len(a2))-1))

    return postecg3, ecg3

def downsample_to_target(x, Fs, target_fs=500):
    """
    x: (samples, channels) 或 (channels, samples)
    回傳: x_ds (samples_new, channels), target_fs
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D")

    # 轉成 (samples, channels)
    if x.shape[0] <= 32 and x.shape[1] > 1000:
        x = x.T

    if Fs == target_fs:
        return x, Fs

    g = gcd(Fs, target_fs)
    up = target_fs // g
    down = Fs // g

    # axis=0 代表沿時間軸 resample
    x_ds = resample_poly(x, up=up, down=down, axis=0)
    return x_ds, target_fs

def beat_alignment_unified(data: np.ndarray, Fs: int, target_fs: int = 500, lead_for_rr: int = 2):
    """
    輸入：
      data: ECG (samples, channels) 或 (channels, samples)
      Fs:   原始取樣率（例如 1000 或 500）

    輸出：
      saveECG: shape = (channels, PR_interval+Rend_interval)
      RR:      median RR（秒），若偵測不到回 None
      fs:      實際用於計算的取樣率（固定回 target_fs）
      locs:    R peaks（sample index，以 target_fs 座標）
    """
    
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data 必須是 2D array")

    # ---------- 先降到 500 Hz ----------
    x, fs = downsample_to_target(x, Fs, target_fs=target_fs)
    n_samples, n_ch = x.shape
    window = round(target_fs*0.108)
    # ---------- 濾波器設計 ----------
    B1, A1 = butter(4, 40 / (fs / 2), btype="low")  # (先低通 40Hz)
    B2, A2 = butter(4, 0.05 / (fs / 2), btype="high") #(再高通 0.05Hz)
    B3, A3 = iirnotch(w0=60, Q=30, fs=fs)
    B4, A4 = butter(4, 0.6 / (fs / 2), btype="low")

    # ---------- interval 設定（以 500Hz 換算） ----------
    PR_interval = int(round(100 * (fs / 1000)))   # 100ms -> 50 samples
    Rend_interval = int(round(500 * (fs / 1000))) # 500ms -> 250 samples
    win_len = PR_interval + Rend_interval
    saveECG = np.full((n_ch, win_len), np.nan, dtype=float)
    RR = None
    # ---------- 用 lead_for_rr 做 QRS 偵測 ----------
    lead_for_rr = int(lead_for_rr)
    lead_for_rr = max(0, min(n_ch - 1, lead_for_rr))

    sig = x[:, lead_for_rr]

    # ecg = filtfilt(B2,A2,filtfilt(B1,A1,data))
    ecg = filtfilt(B1, A1, sig, padtype='odd',  padlen=3*(max(len(B1),len(A1))-1))
    ecg = filtfilt(B2, A2, ecg, padtype='odd',  padlen=3*(max(len(B2),len(A2))-1))

    # DC = medfilt1(ecg, round(fs))  -> 這裡用移動中位數近似會比較大成本；
    # 你原片段後面又做 filtfilt(B4,A4,DC)，實務上用 moving average 也常可行。
    # 但你原碼是 medfilt1，所以這裡用「移動平均」會與 MATLAB 不同。
    # 若你堅持 1:1（medfilt1），我也可以改用 scipy.signal.medfilt(但 kernel 要奇數且大會慢)。
    # 這裡先提供「較快且穩」的 baseline：先用 1 秒 moving average 當 DC，再做 0.6Hz 低通
    DC = _smooth_moving_average(ecg, fs)
    DC1 = filtfilt(B4, A4, DC, padtype='odd',  padlen=3*(max(len(B4),len(A4))-1))
    ecg_nodc = ecg - DC1

    envelop, _ = derivative_based_method2(ecg_nodc, fs, window)

    # threshold = 前 10% 大的最低值 ≈ 90th percentile
    if len(envelop) >= 10:
        threshold = float(np.quantile(envelop, 0.90))
        locs, _ = find_peaks(envelop, height=threshold, distance=int(fs / 4))
        locs = locs.astype(int)
    else:
        locs = np.array([], dtype=int)

    # 去掉邊界不足切窗的 R peaks
    ok = (locs - PR_interval >= 0) & (locs + Rend_interval <= len(ecg))
    locs = locs[ok]
    #print(locs)
    # RR（秒）
    if len(locs) >= 2:
        RR = float(np.median(np.diff(locs)) / fs)

    # ---------- 對每個導程做 beat align / average ----------
    for ch in range(n_ch):
        sig_ch = x[:, ch]
        ecg_ch = filtfilt(B1, A1, sig_ch, padtype='odd',  padlen=3*(max(len(B1),len(A1))-1))
        ecg_ch = filtfilt(B2, A2, ecg_ch, padtype='odd',  padlen=3*(max(len(B2),len(A2))-1))
        ecg_ch = filtfilt(B3, A3, ecg_ch, padtype='odd',  padlen=3*(max(len(B2),len(A2))-1))
        beats = []
        for r in locs:
            temp = ecg_ch[r - PR_interval : r + Rend_interval]
            beats.append(temp - np.mean(temp))

        if len(beats) < 2:
            continue

        all_peak = np.vstack(beats)  # (n_beats, win_len)

        # self covariance remove bad beat：取相關係數平均 >= 第75百分位
        corr = np.corrcoef(all_peak)
        cc = corr.mean(axis=0)
        mask = cc >= np.percentile(cc, 50)
        good = all_peak[mask, :]

        if good.shape[0] > 0:
            saveECG[ch, :] = good.mean(axis=0)

    return saveECG, RR, fs, locs

def beat_alignment_individual(data: np.ndarray, Fs: int, target_fs: int = 500):
    """
    輸出：
      saveECG: shape = (channels, PR_interval+Rend_interval)
      RR:      所有導程偵測到的 median RR（秒）
      fs:      target_fs
      quality_report: 包含每個通道處理後的資訊
      all_locs: List[np.ndarray], 儲存每個 channel 各自偵測到的 R peak 索引 (以 target_fs 為準)
    """
    
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data 必須是 2D array")

    # ---------- 先降到 500 Hz ----------
    x, fs = downsample_to_target(x, Fs, target_fs=target_fs)
    n_samples, n_ch = x.shape
    window = round(target_fs*0.108)

    # 濾波器設計 (與原程式相同)
    B1, A1 = butter(4, 40 / (fs / 2), btype="low")
    B2, A2 = butter(4, 0.05 / (fs / 2), btype="high")
    B3, A3 = iirnotch(w0=60, Q=30, fs=fs)
    B4, A4 = butter(4, 0.6 / (fs / 2), btype="low")

    # Interval 設定
    PR_interval = int(round(100 * (fs / 1000)))
    Rend_interval = int(round(500 * (fs / 1000)))
    win_len = PR_interval + Rend_interval
    
    saveECG = np.full((n_ch, win_len), np.nan, dtype=float)
    all_channel_rrs = []
    all_locs = []  # <--- 新增：用來儲存每個 channel 的 peak 位置
    
    quality_report = {"channels": []}

    for ch in range(n_ch):
        sig_ch = x[:, ch]

        # 1. 預處理
        ecg_f = filtfilt(B1, A1, sig_ch, padtype='odd', padlen=3*(max(len(B1),len(A1))-1))
        ecg_f = filtfilt(B2, A2, ecg_f, padtype='odd', padlen=3*(max(len(B2),len(A2))-1))
        ecg_f = filtfilt(B3, A3, ecg_f, padtype='odd', padlen=3*(max(len(B3),len(A3))-1))

        # 2. 基線漂移校正
        dc_ma = _smooth_moving_average(ecg_f, fs)
        dc_low = filtfilt(B4, A4, dc_ma, padtype='odd', padlen=3*(max(len(B4),len(A4))-1))
        ecg_nodc = ecg_f - dc_low

        # 3. R Peak 偵測
        envelop, _ = derivative_based_method2(ecg_nodc, fs, window)
        
        if len(envelop) >= 10:
            threshold = float(np.quantile(envelop, 0.90))
            locs, _ = find_peaks(envelop, height=threshold, distance=int(fs / 4))
            
            # 過濾掉邊界不足的點
            ok = (locs - PR_interval >= 0) & (locs + Rend_interval <= len(ecg_f))
            locs = locs[ok]
        else:
            locs = np.array([], dtype=int)

        # 將此 channel 的 locs 加入總表
        all_locs.append(locs)

        # 4. 記錄 RR
        if len(locs) >= 2:
            all_channel_rrs.append(np.median(np.diff(locs)) / fs)

        # 5. Beat 疊加邏輯
        beats = []
        for r in locs:
            temp = ecg_f[r - PR_interval : r + Rend_interval]
            beats.append(temp - np.mean(temp))

        if len(beats) < 2:
            quality_report["channels"].append({"ch": ch, "beats_detected": len(locs), "status": "insufficient_beats"})
            continue

        all_peak = np.vstack(beats)
        corr = np.corrcoef(all_peak)
        
        if corr.ndim == 2:
            cc = corr.mean(axis=0)
            mask = cc >= np.percentile(cc, 50)
            good = all_peak[mask, :]
            saveECG[ch, :] = good.mean(axis=0)
            quality_report["channels"].append({"ch": ch, "beats_used": len(good)})
        else:
            saveECG[ch, :] = all_peak.mean(axis=0)
            quality_report["channels"].append({"ch": ch, "beats_used": len(beats)})

    RR = float(np.median(all_channel_rrs)) if all_channel_rrs else None

    return saveECG, RR, fs, all_locs, quality_report