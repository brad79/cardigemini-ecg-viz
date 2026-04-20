# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:53:36 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:54:01 2026

@author: BOX
"""

import numpy as np
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from util import filter_bandpass

# >>> 匯入獨立的 DataLoader 負責處理所有檔案 I/O
from ECGDataLoader import ECGDataLoader

class ECGProcessor:
    """負責深度學習相關的訊號預處理 (裁切、補零、3x4 轉換、濾波、正規化)"""
    def __init__(self, target_fs=500, target_sec=2.5, num_leads=12, apply_filter=True, normalize=True):
        self.target_fs = target_fs
        self.target_len = int(target_fs * target_sec)
        self.num_leads = num_leads
        self.apply_filter = apply_filter
        self.normalize = normalize

    def fix_length(self, signal):
        cur_len = signal.shape[1]

        if cur_len > self.target_len:
            start = (cur_len - self.target_len) // 2
            signal = signal[:, start:start + self.target_len]

        elif cur_len < self.target_len:
            # 資料太短：補零 (Pad)
            pad_width = self.target_len - cur_len
            signal = np.pad(signal, ((0, 0), (0, pad_width)))

        return signal

    def convert_to_3x4_layout(self, signal):
        """將 10秒 12導程轉為 2.5秒 3x4 佈局"""
        # 假設 target_fs 為 500，每段 2.5s = 1250 點
        segment_len = int(self.target_fs * 2.5) 
        aligned_signal = np.zeros((12, segment_len))
        
        # 1. Lead I, II, III -> 0.0 ~ 2.5s
        aligned_signal[0:3, :] = signal[0:3, 0:segment_len]
        # 2. aVR, aVL, aVF -> 2.5 ~ 5.0s
        aligned_signal[3:6, :] = signal[3:6, segment_len:segment_len*2]
        # 3. V1, V2, V3 -> 5.0 ~ 7.5s
        aligned_signal[6:9, :] = signal[6:9, segment_len*2:segment_len*3]
        # 4. V4, V5, V6 -> 7.5 ~ 10.0s
        aligned_signal[9:12, :] = signal[9:12, segment_len*3:segment_len*4]
        
        return aligned_signal
    
    def preprocess(self, signal, fs_in, as_3x4=False):
        # 1. 補值
        signal = np.nan_to_num(signal)        
        
        # 2. 重取樣
        if fs_in != self.target_fs:
            new_len = int(signal.shape[1] * self.target_fs / fs_in)
            signal = resample(signal, new_len, axis=1)
            
        # 3. 檢查是否需要執行 3x4 轉換
        if as_3x4:     
            signal = self.convert_to_3x4_layout(signal)
            
        # 4. 確保最終長度符合 target_len
        signal = self.fix_length(signal)      
           
        # 5. 濾波
        if self.apply_filter:
            signal = filter_bandpass(signal, self.target_fs)            
            
        # 6. 標準化
        if self.normalize:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
        return torch.FloatTensor(signal.copy())
    

# 適用於有答案的場景 (Train/Val/Test)
class ECGLabeledDataset(Dataset):    
    def __init__(self, ecg_path, labels_df, processor, file_format="auto", 
                 filename_col='filename', label_col='label', task_type="classification", as_3x4=True):
        self.ecg_path = ecg_path
        self.labels_df = labels_df
        self.processor = processor
        
        # >>> 初始化 ECGDataLoader (統一負責讀取各種格式)
        self.data_loader = ECGDataLoader(target_fs=processor.target_fs)
        
        self.filename_col = filename_col
        self.label_col = label_col
        self.task_type = task_type 
        self.as_3x4 = as_3x4
        # 註: file_format 參數保留以防 ecg_engine 傳入，但實際上 DataLoader 會根據副檔名自動判別

    def __len__(self):
        return len(self.labels_df)

    def parse_label(self, label):
        if isinstance(label, str) and label.startswith("["):
            label = ast.literal_eval(label)
            label = torch.tensor(label, dtype=torch.float32)    
        elif self.task_type == "classification":
            label = torch.tensor([label], dtype=torch.float32)    
        elif self.task_type == "regression":
            label = torch.tensor([label], dtype=torch.float32)
        elif self.task_type == "multiclass":
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        return label

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filepath = f"{self.ecg_path}{row[self.filename_col]}"

        # 1. 透過 DataLoader 讀取原始訊號 
        # 要求提取 10 秒，這樣後續如果要切換為 3x4 (2.5秒x4) 才有足夠的資料
        signal, fs, _ = self.data_loader.load_data(
            filepath, 
            only_standard_leads=True, 
            duration_sec=10.0
        )
        
        # 2. 交給 processor 處理 (轉換為 Tensor、濾波、正規化等)
        signal_tensor = self.processor.preprocess(signal, fs, as_3x4=self.as_3x4)
                
        # 3. 標籤解析
        label_tensor = self.parse_label(row[self.label_col])
        
        return signal_tensor, label_tensor
    

# 適用於完全沒答案的場景 (Production/Inference)
class ECGUnlabeledDataset(Dataset):    
    def __init__(self, file_list, processor, ecg_path="", file_format="auto"):
        self.file_list = file_list
        self.processor = processor
        self.ecg_path = ecg_path
        
        # >>> 初始化 ECGDataLoader
        self.data_loader = ECGDataLoader(target_fs=processor.target_fs)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = f"{self.ecg_path}{filename}"
        
        # 1. 讀取訊號
        signal, fs, _ = self.data_loader.load_data(
            filepath, 
            only_standard_leads=True, 
            duration_sec=10.0
        )
        
        # 2. 預處理 (推斷當前的 target_sec 是否為 2.5，如果是則代表為 3x4 模式)
        as_3x4 = True if self.processor.target_sec == 2.5 else False
        signal_tensor = self.processor.preprocess(signal, fs, as_3x4=as_3x4)
        
        return signal_tensor, filename