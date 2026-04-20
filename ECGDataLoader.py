# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:26:20 2026

@author: BOX
"""
import wfdb
import numpy as np
import h5py
import cv2
import os
from scipy.io import loadmat
from typing import Optional,  Tuple, List,  Any, Dict
from scipy.signal import butter, filtfilt, iirnotch

from pathlib import Path
import fitz  # PyMuPDF

class ECGDataLoader:
    """負責從不同來源載入並標準化 ECG 數據"""
    
    def __init__(self, target_fs=500):
        self.target_fs = target_fs
        self.standard_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # 定義 PDF 解析時的物理常數與佈局
        self._pdf_mm_per_pt = 25.4 / 72
        self._pdf_pt_per_mv = 10 / self._pdf_mm_per_pt
        self._pdf_target_len = 1250 # 針對 2.5s @ 500Hz
        self._pdf_lead_layout = [
            ["I",   "II",  "III" ],
            ["AVR", "AVL", "AVF" ],
            ["V1",  "V2",  "V3"  ],
            ["V4",  "V5",  "V6"  ],
        ]

    def load_from_pdf(self, filepath: str) -> Tuple[np.ndarray, float]:
        """
        從 PDF 向量圖形中提取 ECG 訊號。
        """
        if fitz is None:
            raise ImportError("請先安裝 pymupdf: pip install pymupdf")

        doc = fitz.open(filepath)
        rhythm_data = 0
        try:
            page = doc[0]
            paths = page.get_drawings()
            # 篩選黑色且具有一定長度的路徑（排除網格或雜點）
            black_paths = [p for p in paths if p.get("color") == (0.0, 0.0, 0.0) and len(p.get("items", [])) >= 30]

            if len(black_paths) < 12:
                raise RuntimeError(f"PDF 內可識別的波形路徑不足 (僅找到 {len(black_paths)} 條)")

            path_info = []
            for p in black_paths:
                xs, ys = self._path_to_xy(p)
                if len(xs) == 0: continue
                path_info.append({
                    "xs": xs, "ys": ys, 
                    "y_max": ys.max(), 
                    "y_span": ys.max() - ys.min(), 
                    "x_mean": xs.mean()
                })

            # 排除最大（可能是外框）的路徑，取前 12 個波形段
            path_info.sort(key=lambda x: -x["y_span"])
            segment_paths = path_info[1:13] 

            # ==========================================
            # 新增：擷取下方的 Rhythm Data (10秒節律導程)
            # 根據排序，path_info[0] 即為 y_span 最長的線條 (Rhythm Strip)
            # ==========================================
            rhythm_seg = path_info[0]
            xs_r, ys_r = rhythm_seg["xs"], rhythm_seg["ys"]
            # 設定基準線
            x_baseline_r = xs_r[0]

            # 執行與 12 導程相同的座標轉換邏輯
            order_r = np.argsort(-ys_r)
            ys_r_sorted, xs_r_sorted = ys_r[order_r], xs_r[order_r]
            ys_r_u, idx_r, counts_r = np.unique(ys_r_sorted, return_index=True, return_counts=True)
            xs_r_u = np.array([xs_r_sorted[idx_r[i]:idx_r[i]+counts_r[i]].mean() for i in range(len(ys_r_u))])[::-1]

            amp_mv_r = (x_baseline_r - xs_r_u) / self._pdf_pt_per_mv
            
            # Rhythm strip 是 10 秒，所以插值目標長度為 12 導程(2.5秒)的 4 倍
            rhythm_target_len = self._pdf_target_len * 4
            
            if len(amp_mv_r) > 1:
                xp_r = np.linspace(0, 1, len(amp_mv_r))
                x_new_r = np.linspace(0, 1, rhythm_target_len)
                rhythm_data = np.interp(x_new_r, xp_r, amp_mv_r)
            else:
                rhythm_data = np.zeros(rhythm_target_len)
            # ==========================================

            # 依照垂直位置（row）與水平位置（column）排序 (維持原 12 導程邏輯不變)
            segment_paths.sort(key=lambda p: -p["y_max"])
            rows = [segment_paths[i*3:(i+1)*3] for i in range(4)]
            for row in rows: 
                row.sort(key=lambda p: p["x_mean"])

            # 建立訊號字典
            signals = {}
            col_baselines = [rows[0][ci]["xs"][0] for ci in range(3)]

            for ri, row in enumerate(rows):
                for ci, seg in enumerate(row):
                    lead_name = self._pdf_lead_layout[ri][ci]
                    xs, ys = seg["xs"], seg["ys"]
                    x_baseline = col_baselines[ci]
                    
                    # 座標轉換：將點座標轉為 mV
                    order = np.argsort(-ys)
                    ys_sorted, xs_sorted = ys[order], xs[order]
                    ys_u, idx, counts = np.unique(ys_sorted, return_index=True, return_counts=True)
                    xs_u = np.array([xs_sorted[idx[i]:idx[i]+counts[i]].mean() for i in range(len(ys_u))])[::-1]

                    amp_mv = (x_baseline - xs_u) / self._pdf_pt_per_mv
                    
                    # 插值到目標長度 (預設 1250 點)
                    if len(amp_mv) > 1:
                        xp = np.linspace(0, 1, len(amp_mv))
                        x_new = np.linspace(0, 1, self._pdf_target_len)
                        amp_interp = np.interp(x_new, xp, amp_mv)
                    else:
                        amp_interp = np.zeros(self._pdf_target_len)
                    
                    signals[lead_name] = amp_interp

            # 依照 standard_leads 順序排列輸出 (12, Samples)
            aligned_signal = np.zeros((12, self._pdf_target_len))
            for i, lead in enumerate(self.standard_leads):
                if lead in signals:
                    aligned_signal[i, :] = signals[lead]

            # 同時回傳 12 導程矩陣, 取樣率, 以及新擷取出的 10秒 Rhythm Data
            return aligned_signal, float(self.target_fs), rhythm_data

        finally:
            doc.close()              
            

    def _path_to_xy(self, path):
        """輔助函式：將 PyMuPDF 路徑轉換為座標陣列"""
        xs, ys = [], []
        for item in path.get("items", []):
            kind = item[0]
            if kind == "l": # Line
                xs.extend([item[1].x, item[2].x])
                ys.extend([item[1].y, item[2].y])
            elif kind == "c": # Curve
                for j in range(1, 4):
                    xs.append(item[j].x); ys.append(item[j].y)
            elif kind == "m": # Move
                xs.append(item[1].x); ys.append(item[1].y)
        return np.array(xs), np.array(ys)
    # --- 新增的 EDF/BDF 支援 ---
    def load_from_edf(self, filepath: str, only_standard_leads: bool = True, duration_sec: float = 10.0) -> Tuple[np.ndarray, float]:
        """
        讀取 EDF/BDF 檔案。
        
        Args:
            filepath: 檔案路徑。
            only_standard_leads: 若為 True，僅回傳對齊後的標準 12 導程；若為 False，回傳檔案內所有原始資料。
            
        Returns:
            Tuple[np.ndarray, float]: (訊號矩陣, 取樣率)
        """
        try:
            import mne #
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose="ERROR") #
            fs = float(raw.info["sfreq"]) #
            raw_data = raw.get_data() * 1000.0  # 轉為 mV
            lead_names = [self._clean_label(n) for n in raw.ch_names] #
        except ImportError:
            # 若無 mne 則嘗試使用 pyedflib
            try:
                import pyedflib #
                f = pyedflib.EdfReader(filepath) #
                n_ch = f.signals_in_file #
                fs = f.getSampleFrequency(0) #
                sigs = []
                lead_names = []
                for i in range(n_ch):
                    s = f.readSignal(i)
                    unit = f.getPhysicalDimension(i).upper()
                    # 單位補償
                    if 'UV' in unit or 'MICROVOLT' in unit:
                        s = s * 0.001
                    elif 'V' in unit and 'MV' not in unit:
                        s = s * 1000.0
                    sigs.append(s)
                    lead_names.append(self._clean_label(f.getLabel(i))) #
                f.close()
                raw_data = np.array(sigs)
            except Exception as e:
                print(f"EDF 解析失敗: {e}")
                return None, self.target_fs

        # 根據參數決定輸出內容
        if only_standard_leads:
            # 計算 10 秒對應的樣本數
            num_samples = int(duration_sec * fs)
            # 確保不會超過原始資料長度
            max_len = min(num_samples, raw_data.shape[1])
            
            # 建立空的標準 12 導程矩陣 (12, Samples)
            aligned_signal = np.zeros((12, max_len))
            for i, target_lead in enumerate(self.standard_leads):
                target_upper = target_lead.upper()
                if target_upper in lead_names:
                    idx = lead_names.index(target_upper)
                    # 僅提取前 10 秒的資料
                    aligned_signal[i, :] = raw_data[idx, :max_len]
            return aligned_signal, fs
        else:
            # 回傳全部通道與完整長度資料
            return self._ensure_2d_leads_samples(raw_data), fs

    def _clean_label(self, label: str) -> str:
        """輔助函數：清洗並標準化導程名稱，以便比對"""
        s = str(label).upper().replace("ECG", "").replace("LEAD", "")
        s = s.replace("-", "").replace(" ", "").strip()
        return s
    
    def load_from_mat(self, filepath: str) -> Tuple[np.ndarray, float]:
        """讀取 .mat 檔案，回傳 (signal, fs)"""
        try:
            try:
                raw_dict = loadmat(filepath)
            except NotImplementedError:
                with h5py.File(filepath, 'r') as f:
                    raw_dict = {k: np.array(f[k]) for k in f.keys()}

            data, fs, labels = self._pick_mat_data(raw_dict)
            return data, (fs if fs else self.target_fs)
        except Exception as e:
            print(f"MAT 檔案解析失敗: {e}")
            return None, self.target_fs

    def load_from_wfdb(self, record_path: str, duration_sec=10.0) -> Tuple[np.ndarray, float]:
        """讀取 WFDB 並自動對齊標準 12 導程"""
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        num_samples = int(duration_sec * fs)
        signals = record.p_signal[:num_samples, :]
        lead_names = [n.upper() for n in record.sig_name]
        
        aligned_signal = np.zeros((12, signals.shape[0]))
        for i, lead in enumerate(self.standard_leads):
            if lead in lead_names:
                aligned_signal[i, :] = signals[:, lead_names.index(lead)]
            else:
                # 導程補償邏輯 (Einthoven's Law)
                if lead == 'III' and 'I' in lead_names and 'II' in lead_names:
                    aligned_signal[i, :] = signals[:, lead_names.index('II')] - signals[:, lead_names.index('I')]
                elif lead == 'AVR' and 'I' in lead_names and 'II' in lead_names:
                    aligned_signal[i, :] = -(signals[:, lead_names.index('I')] + signals[:, lead_names.index('II')]) / 2
                elif lead == 'AVL' and 'I' in lead_names and 'II' in lead_names:
                    aligned_signal[i, :] = (signals[:, lead_names.index('I')] - (signals[:, lead_names.index('II')] - signals[:, lead_names.index('I')])) / 2
                elif lead == 'AVF' and 'I' in lead_names and 'II' in lead_names:
                    aligned_signal[i, :] = (signals[:, lead_names.index('II')] + (signals[:, lead_names.index('II')] - signals[:, lead_names.index('I')])) / 2
                else:
                    # 若真的完全沒有該導聯，則維持 0
                    pass
        return aligned_signal, fs

    
    def _pick_mat_data(self, mat: Dict[str, Any]) -> Tuple[np.ndarray, Optional[float], Optional[List[str]]]:
       """從 .mat 字典內盡量猜出 data / Fs / label"""
       
       # --- Fs 偵測 ---
       fs_keys = ["Fs", "fs", "FS", "sampFreq", "sampling_rate", "samplingRate", "sfreq", "SR"]
       fs = None
       for k in fs_keys:
           if k in mat:
               try:
                   fs = float(np.asarray(mat[k]).squeeze())
                   break
               except Exception: continue

       # --- Label 偵測 ---
       label_keys = ["label", "labels", "ch_names", "chan_names", "channel_names", "sig_name", "lead_names"]
       labels = None
       for k in label_keys:
           if k in mat:
               try:
                   arr = np.asarray(mat[k]).squeeze()
                   if arr.dtype.kind in {"U", "S"}: # String types
                       labels = [str(x) for x in arr.tolist()] if arr.ndim > 0 else [str(arr)]
                       break
                   if arr.dtype == object: # Object arrays (common in scipy.io)
                       labels = [str(x).strip() for x in arr.tolist()] if arr.ndim > 0 else [str(arr)]
                       break
               except Exception: continue

       # --- Data 偵測 ---
       # 優先嘗試特定 key
       data_keys = ["ecg_final", "data", "ecg", "ECG", "signal", "signals", "sig", "val", "x", "X"]
       for k in data_keys:
           if k in mat:
               v = mat[k]
               if isinstance(v, np.ndarray) and v.size > 0 and v.ndim in (1, 2):
                   return self._ensure_2d_leads_samples(v), fs, labels

       # 若沒找到，挑最大的數值陣列
       best, best_size = None, -1
       for k, v in mat.items():
           if k.startswith("__"): continue
           if isinstance(v, np.ndarray) and v.size > 0 and v.ndim in (1, 2):
               if np.issubdtype(v.dtype, np.number):
                   if v.size > best_size:
                       best, best_size = v, v.size

       if best is None:
           raise ValueError("找不到可用的數值數據")

       return self._ensure_2d_leads_samples(best), fs, labels
       
    def _ensure_2d_leads_samples(self, data: np.ndarray) -> np.ndarray:
       """確保輸出形狀為 (Leads, Samples)，例如 (12, 5000)"""
       if data.ndim == 1:
           # 如果是一維，假定為單導程 (1, Samples)
           return data[np.newaxis, :]
       
       # 如果是二維，判斷哪一邊是導程 (通常 ECG 導程數 < 樣本數)
       return data.T if data.shape[0] > data.shape[1] else data

    def _crop_data(self, data: np.ndarray, fs: float, duration_sec: float) -> np.ndarray:
        """
        將數據裁剪至指定的秒數。
        
        Args:
            data: (Leads, Samples) 的訊號矩陣
            fs: 取樣頻率
            duration_sec: 目標秒數
            
        Returns:
            np.ndarray: 裁剪後的訊號
        """
        if data is None:
            return None
            
        num_samples = int(duration_sec * fs)
        # 確保不會超過原始資料長度，若不足則維持原樣（或可視需求補零）
        curr_len = data.shape[1]
        target_len = min(num_samples, curr_len)
        
        return data[:, :target_len]
    def load_data(self, filepath: str, only_standard_leads: bool = True, duration_sec: float = 10.0) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        自動辨識副檔名並載入 ECG 資料。
        
        Args:
            filepath: 檔案路徑或 WFDB record name。
            only_standard_leads: 是否僅輸出對齊後的標準 12 導程（且限制時長）。
            duration_sec: 限制輸出的秒數（僅在 only_standard_leads=True 時生效）。
            
        Returns:
            Tuple[np.ndarray, float]: (處理後的訊號矩陣, 取樣率)
        """
        p = Path(filepath)
        ext = p.suffix.lower()
        rhythm_data = None
        
        if ext == ".pdf":
            data, fs, rhythm_data = self.load_from_pdf(filepath)
            # PDF 提取出的數據通常已經是固定長度，不額外 crop
        elif ext in {".edf", ".bdf"}:
            data, fs = self.load_from_edf(filepath, only_standard_leads, duration_sec)
        
        elif ext == ".mat":
            data, fs = self.load_from_mat(filepath)
            # MAT 檔案通常需要手動處理標準化與時長（若有需要）
            if only_standard_leads and data is not None:
                data = self._crop_data(data, fs, duration_sec)
        
        elif ext in {".hea", ".dat"} or ext == "":
            # 處理 WFDB 格式
            # 注意：WFDB 的 load_from_wfdb 本身就具備 duration 限制與 12 導對齊功能
            data, fs = self.load_from_wfdb(str(p.with_suffix("")), duration_sec=duration_sec)
        
        else:
            raise ValueError(f"不支援的檔案格式: {ext}")

        if data is None:
            raise IOError(f"無法從路徑載入資料: {filepath}")

    # --- 新增：針對非 PDF 格式，預設取 Lead II 作為 rhythm_data ---
        if rhythm_data is None:
            try:
                # 尋找 Lead II 在 standard_leads 中的索引 (通常是 1)
                lead_ii_idx = self.standard_leads.index('II')
                if data.shape[0] > lead_ii_idx:
                    rhythm_data = data[lead_ii_idx, :]
                else:
                    # 如果找不到，則取第一條導程
                    rhythm_data = data[0, :]
            except (ValueError, IndexError):
                # 若發生意外，回傳全零陣列避免程式崩潰
                rhythm_data = np.zeros(data.shape[1])
                
        return data, fs, rhythm_data

       
    def filter_signal(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        對 ECG 訊號進行濾波處理，使其易於觀察。
        
        Args:
            data: (Leads, Samples) 的原始訊號
            fs: 取樣頻率
            
        Returns:
            np.ndarray: 濾波後的訊號
        """
        if data is None or data.size == 0:
            return data

        # 1. 先行 DC 濾除 (Zero-mean)
        # 減去每個導程各自的平均值，防止濾波器在起始點產生巨大的突波 (Transient response)
        data_dc_removed = data - np.mean(data, axis=-1, keepdims=True)
    
        # 2. 帶通濾波器 (Bandpass Filter)
        # 00.5Hz ~ 45Hz。使用較低的階數 (如 2 或 3) 可以減少群延遲與邊界效應    
        lowcut = 0.05
        highcut = 45.0
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(2, [low, high], btype='band')
        
        # 使用 filtfilt 進行雙向濾波，可達到零相位失真，且波形不會位移
        filtered_data = filtfilt(b, a, data_dc_removed, axis=-1)
    
        # 3. 陷波濾波器 (Notch Filter)
        # 去除 60Hz 電源線干擾
        f0 = 60.0  
        Q = 30.0   
        b_notch, a_notch = iirnotch(f0, Q, fs)
        filtered_data = filtfilt(b_notch, a_notch, filtered_data, axis=-1)
    
        return filtered_data
   