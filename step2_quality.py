# -*- coding: utf-8 -*-
"""
step2_quality.py
訊號品質評估 + 心律不整偵測
移植自 ecg_cloud_platform/core/signal_processing.py 與 diagnosis_engine.py
不依賴 ecg_cloud_platform 專案路徑
"""
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.stats import kurtosis


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = 0.5,
                    highcut: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, signal)


def evaluate_sqi(signal: np.ndarray) -> tuple[float, bool]:
    """
    訊號品質指數（kurtosis-based）
    回傳: (score 0-1, is_reliable bool)
    """
    k = float(kurtosis(signal))
    score = round(min(1.0, max(0.0, (k - 3) / 7)), 2)
    return score, k > 5.0


def detect_r_peaks(filtered_signal: np.ndarray, fs: int) -> np.ndarray:
    """簡易微分平方法 R-peak 偵測，用於心律不整分析"""
    diff = np.diff(filtered_signal)
    squared = diff ** 2
    window_len = max(1, int(0.12 * fs))
    integrated = np.convolve(squared, np.ones(window_len) / window_len, mode='same')
    distance = max(1, int(0.2 * fs))
    peaks, _ = find_peaks(integrated, distance=distance, height=np.mean(integrated))
    return peaks


def detect_arrhythmia(r_peaks: np.ndarray, fs: int) -> dict:
    """
    基於 RR interval 的心律不整偵測。
    回傳 dict:
      heart_rate (float), rr_intervals (list), arrhythmias (list[str])
    """
    arrhythmias: list[str] = []
    rr_intervals: list[float] = []
    heart_rate: float = 0.0

    if len(r_peaks) < 2:
        return {"heart_rate": 0.0, "rr_intervals": [], "arrhythmias": ["Insufficient beats"]}

    rr = np.diff(r_peaks) / fs           # RR in seconds
    rr_valid = rr[(rr > 0.2) & (rr < 2.0)]
    rr_intervals = rr_valid.tolist()
    heart_rate = 60.0 / np.mean(rr_valid) if len(rr_valid) > 0 else 0.0

    if len(rr_valid) > 5:
        cv = np.std(rr_valid) / np.mean(rr_valid)
        if cv > 0.15:
            arrhythmias.append("Atrial Fibrillation (RR variability > 15%)")

    # PVC / 寬 QRS 偵測（以 RR 早搏準則）
    if len(rr_valid) > 2:
        median_rr = np.median(rr_valid)
        early_count = int(np.sum(rr_valid < 0.82 * median_rr))
        if early_count > 0:
            arrhythmias.append(f"Premature beats detected ({early_count} early RR)")

    return {
        "heart_rate": round(heart_rate, 1),
        "rr_intervals": rr_intervals,
        "arrhythmias": arrhythmias,
    }


def analyze_signal_quality(data: np.ndarray, fs: int,
                            lead_names: list[str]) -> list[dict]:
    """
    對 12 導程各自計算 SQI + 心律不整。
    data shape: (n_samples, n_channels)
    回傳 list of dict，每個 dict 對應一條導程。
    """
    n_ch = data.shape[1]
    results = []
    # 以 Lead II（index 1）做心律不整偵測
    lead_ii_idx = 1 if n_ch > 1 else 0
    arrhythmia_info: dict = {}

    try:
        lead_ii_filtered = bandpass_filter(data[:, lead_ii_idx], fs)
        r_peaks = detect_r_peaks(lead_ii_filtered, fs)
        arrhythmia_info = detect_arrhythmia(r_peaks, fs)
    except Exception:
        arrhythmia_info = {"heart_rate": 0.0, "rr_intervals": [], "arrhythmias": ["Detection failed"]}

    for ch in range(n_ch):
        name = lead_names[ch] if ch < len(lead_names) else f"CH{ch+1}"
        sig = data[:, ch].astype(float)
        try:
            filt = bandpass_filter(sig, fs)
            sqi, reliable = evaluate_sqi(filt)
        except Exception:
            sqi, reliable = 0.0, False

        results.append({
            "lead": name,
            "sqi": sqi,
            "reliable": reliable,
        })

    return results, arrhythmia_info
