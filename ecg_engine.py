# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:19 2026

@author: BOX

輸入端：無論是呼叫 train_model、run_batch_inference 還是 predict_single，您只需要負責給「檔案路徑」，不需要再管它是什麼副檔名。
規格端：您完全掌握 target_sec (要截取多少秒) 以及 as_3x4 (是否要將訊號重新排列)，兩者職責分明。

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from dataset import ECGProcessor, ECGLabeledDataset
from ECGDataLoader import ECGDataLoader  # 新增這行

from tqdm import tqdm # 確保在檔案頂部導入
from util import my_eval_with_dynamic_thresh # 確保導入工具函式
from net1d import Net1D
from sklearn.metrics import confusion_matrix 
class ECGAppEngine:
    def __init__(self, model_pth, num_leads=12, n_classes=1, is_training=False, target_sec=10, device=None,target_fs=500):        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_leads = num_leads
        self.n_classes = n_classes
        self.is_training = is_training
        #self.reader = ECGReaderFactory.get_reader("wfdb")
        self.filename_col = 'filename' # 預設檔名欄位
        self.label_col = 'label'       # 預設標籤欄位
        
        # 1. 預設初始化 
        self.processor = ECGProcessor(num_leads=num_leads, target_fs=target_fs, target_sec=target_sec)
        
        # 2. 透過私有方法初始化模型與載入權重
        self.model = self._initialize_model(model_pth)
        self.model.to(self.device)
        

    def _initialize_model(self, model_pth):
        """ 建構 Net1D 架構並處理微調權重過濾 """
        model = Net1D(
            in_channels=self.num_leads, 
            base_filters=64, # 你可以根據需求調整為 128
            ratio=1, 
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],    
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4], 
            kernel_size=16, stride=2,
            groups_width=16,
            n_classes=self.n_classes,
            use_bn=False, use_do=False
        )

        if model_pth and os.path.exists(model_pth):
            # 處理 numpy 序列化相容性問題
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(model_pth, map_location=self.device, weights_only=False)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            if self.is_training:
                # 訓練/微調模式：過濾掉最後的分類層（dense）
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')}
                model.load_state_dict(state_dict, strict=False)
                print(f">>> [微調模式] 已載入預訓練骨架，排除分類層: {model_pth}")
            else:
                # 評估/推論模式：嚴格對齊所有參數
                model.load_state_dict(state_dict, strict=True)
                print(f">>> [評估模式] 已載入完整模型權重: {model_pth}")
        else:
            print(">>> [警告] 未偵測到模型權重路徑，將使用隨機初始化模型。")
            
        return model
    
    def _evaluate_metrics(self, dataloader):
        """ 內部驗證函式，計算 AUROC 等指標 """
        self.model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                outputs = self.model(signals)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_gts.append(labels.numpy())

        all_preds = np.vstack(all_preds)
        all_gts = np.vstack(all_gts)
        
        # 呼叫你現有的工具函式計算指標
        # res_avg 是平均 AUROC
        res_avg, _, _, _, _, _, _ = my_eval_with_dynamic_thresh(all_gts, all_preds)
        return res_avg
   
    # 1. 移除 file_format，新增 target_sec 參數 (預設給 2.5 或您常用的秒數)
    def train_model(self, train_df, val_df, ecg_path, as_3x4=False, target_sec=10,
                    save_path="best_model.pth", epochs=10, batch_size=64, lr=1e-5):
                        
        early_stop_lr = 1e-5
        weight_decay=1e-5
        num_workers = 8   
  
        self.processor.target_sec = target_sec
        self.processor.target_len = int(500 * target_sec)
        
        print(f">>> 訓練設定: {'3x4 Layout (2.5s)' if as_3x4 else 'Full Signal (10s)'}")
                     
        train_ds = ECGLabeledDataset(ecg_path, train_df, self.processor, as_3x4=as_3x4)
        val_ds = ECGLabeledDataset(ecg_path, val_df, self.processor, as_3x4=as_3x4)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_steps_per_epoch = len(train_loader)
        eval_steps = total_steps_per_epoch                
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 修改 Scheduler：改為監控 'max' (AUROC 越大越好)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)
        criterion = nn.BCEWithLogitsLoss()  # 配合 Notebook 的二分類任務

        # 改為追蹤最佳 AUROC
        best_val_auroc = 0.0
        step = 0 # 追蹤總步數

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            # 使用 tqdm 包裝 train_loader
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
            for signals, labels in pbar:
                signals, labels = signals.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(signals), labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                step += 1
                
                # 每隔固定 step 進行驗證
                if step % eval_steps == 0:
                    current_val_auroc = self._evaluate_metrics(val_loader)
                    print(f"\n[Step {step}] Mid-epoch Val AUROC: {current_val_auroc:.4f}")
                    
                    # 檢查並儲存
                    if current_val_auroc > best_val_auroc:
                        best_val_auroc = current_val_auroc
                        torch.save({'state_dict': self.model.state_dict(), 'auroc': best_val_auroc}, save_path)
                        print(f"--> 節點驗證發現更優模型，已更新儲存。")
                    
                    # 更新學習率與 Early Stop 檢查
                    scheduler.step(current_val_auroc)
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr < early_stop_lr:
                        print("Early stop triggered")
                        break # 跳出 batch 迴圈                    
                    self.model.train() # 驗證完切回訓練模式                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            if optimizer.param_groups[0]['lr'] < early_stop_lr:
                break # 跳出 epoch 迴圈
                  
        print(f"訓練完成！最佳驗證 AUROC: {best_val_auroc:.4f}")


    def run_batch_inference(self, test_df, ecg_path, as_3x4=False, target_sec=10, batch_size=32, num_workers=8):
    # 明確區分這只是單純的「批次推論」，不含指標計算。

 
        self.processor.target_sec = target_sec
        self.processor.target_len = int(self.processor.target_fs * self.processor.target_sec)                  
        self.model.eval()
        ds = ECGLabeledDataset(ecg_path, test_df, self.processor,  as_3x4=as_3x4)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        all_preds, all_gts = [], []
        print(f">>> 開始推論測試集 (Inference)... [3x4 Layout: {as_3x4} | 目標秒數: {target_sec}s]")
        with torch.no_grad():
            for signals, labels in tqdm(loader, desc="Inference"):
                signals = signals.to(self.device)
                outputs = self.model(signals)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_gts.append(labels.numpy())

        # 合併並轉為一維陣列
        all_preds = np.vstack(all_preds).flatten()
        all_gts = np.vstack(all_gts).flatten()
        
        return all_preds, all_gts

    
    # 「產生效能報告」
    def generate_performance_report(self, test_df, all_preds, all_gts, excel_name="evaluation_results.xlsx", thresholds=None):
        """
        第二步：僅執行計算與存檔。
        如果不滿意 Excel 格式，修改此函式後重新執行即可，不需重新推論。
        """
        print(f">>> 正在計算指標並產生 Excel: {excel_name}")
        # 1. 計算指標
        # 如果 thresholds 是 None，此函數通常會自動尋找最佳門檻並回傳於 final_thresholds
        mean_auroc, rocaucs, sens, specs, f1, auprcs, final_thresholds = \
            my_eval_with_dynamic_thresh(all_gts.reshape(-1, 1), all_preds.reshape(-1, 1), input_thresholds=thresholds)
        
        # 2. 修正錯誤點：
        # 如果外部沒給 thresholds，就用函數算出來的 final_thresholds
        if thresholds is not None:
            # 如果外部有給 list，取第一個
            best_thresh = thresholds[0] if isinstance(thresholds, (list, np.ndarray)) else thresholds
        else:
            # 如果外部沒給 (None)，則使用 my_eval_with_dynamic_thresh 算出來的最佳門檻
            best_thresh = final_thresholds[0]
            
        # 3. 將機率轉為二值化標籤 (0 或 1)
        final_binary = (all_preds > best_thresh).astype(int)
         
        # --- 核心修改：計算混淆矩陣 ---
        # labels=[0, 1] 確保矩陣順序為 [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(all_gts, final_binary, labels=[0, 1]).ravel()
        
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")



        # 準備明細資料
        export_data = {}
        if 'ecg_id' in test_df.columns:
            export_data['ECG_ID'] = test_df['ecg_id'].values
        
        export_data['FileName'] = test_df[self.filename_col].values
        export_data['True_Label'] = all_gts
        export_data['Prediction_Prob'] = all_preds
        export_data['Binary_Prediction'] = final_binary
        export_data['Correct'] = (all_gts == final_binary)

        detail_df = pd.DataFrame(export_data)

        # 準備摘要資料
        summary_df = pd.DataFrame({
            'Metric': ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'F1', 'Best_Threshold'],
            'Value': [mean_auroc, np.mean(auprcs), np.mean(sens), np.mean(specs), np.mean(f1), best_thresh]
        })
  

        # 5. 準備「Summary」工作表資料：包含所有關鍵指標與矩陣數值
        summary_df = pd.DataFrame({
            'Metric': [
                'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'F1-Score', 
                'Best_Threshold', 'True_Positive (TP)','False_Positive (FP)',
                 'False_Negative (FN)', 'True_Negative (TN)', 'Total_Samples'
            ],
            'Value': [
                mean_auroc, np.mean(auprcs), np.mean(sens), np.mean(specs), np.mean(f1), 
                best_thresh, tn, fp, fn, tp, len(all_gts)
            ]
        })
        # 寫入 Excel
        with pd.ExcelWriter(excel_name) as writer:
            detail_df.to_excel(writer, sheet_name='Details', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print("成功！報表已儲存。")
        return summary_df

    def run_full_validation_pipeline(self, test_df, ecg_path, output_filename="Final_Report.xlsx", thresholds=None, as_3x4=True, target_sec=2.5):
        """
        一鍵啟動驗證流水線 (推論 + 產出報告)
        """
        # 正確將控制參數傳遞給我們修改過的 run_batch_inference
        preds, gts = self.run_batch_inference(test_df, ecg_path, as_3x4=as_3x4, target_sec=target_sec)        
        # 產出報告邏輯維持不變
        summary_df = self.generate_performance_report(test_df, preds, gts, excel_name=output_filename, thresholds=thresholds)       
        return summary_df



    def predict_single(self, file_path, target_sec=10.0, as_3x4=False):
                   
        self.model.eval()        
        # 1. 實例化 DataLoader (由它來自動判斷格式、對齊導程與處理 WFDB 路徑)
        data_loader = ECGDataLoader(target_fs=self.processor.target_fs)
        # 2. 讀取訊號 (限制為標準 12 導程，長度先抓 10 秒供後續轉換使用)
        try:
            signal, fs, _ = data_loader.load_data(file_path, only_standard_leads=True, duration_sec=target_sec)
        except Exception as e:
            raise RuntimeError(f"讀取單一檔案失敗 {file_path}: {e}")
                        
        # 2. 根據模式調整 Processor 狀態
        self.processor.target_sec = target_sec
        self.processor.target_len = int(self.processor.target_fs * target_sec)
        
        # 3. 預處理與預測
        input_tensor = self.processor.preprocess(signal, fs, as_3x4=as_3x4).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).cpu().item()
            
        return prob
       

    def predict_batch(self, file_paths, target_sec=10.0, as_3x4=False):
        """
        輸入路徑列表，回傳 {路徑: 機率} 的字典。
        """
        results = {}
        for path in file_paths:
            try:
                # 呼叫更新後的 predict_single，並傳遞明確的控制參數
                results[path] = self.predict_single(path, target_sec=target_sec, as_3x4=as_3x4)
            except Exception as e:
                results[path] = f"Error: {str(e)}"
        return results
    
## ----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
def plot_confusion_matrix(all_gts, all_preds, threshold=0.3, save_path="confusion_matrix.png"):
    """
    繪製混淆矩陣熱點圖，格式仿照用戶提供的圖片。
    """
    # 1. 根據門檻值產生二值化結果
    final_binary = (all_preds > threshold).astype(int)
    
    # 2. 計算矩陣數值
    # labels=[1, 0] 讓 Sick(1) 在左/上，Healthy(0) 在右/下
    # 確保順序：1 為 Sick, 0 為 Healthy
    tn, fp, fn, tp = confusion_matrix(all_gts, final_binary, labels=[0, 1]).ravel()

    P = tp + fn  # 實際陽性總數
    N = fp + tn  # 實際陰性總數
    PP = tp + fp # 預測陽性總數 (Predicted Positive Sum)
    PN = fn + tn # 預測陰性總數 (Predicted Negative Sum)
    Total = P + N
    
    eps = 1e-9
    tpr = tp / (P + eps)      # Sensitivity
    tnr = tn / (N + eps)      # Specificity
    ppv = tp / (PP + eps)     # PPV
    npv = tn / (PN + eps)     # NPV
    acc = accuracy_score(all_gts, final_binary) # Accuracy

    # 2. 設定畫布
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 5.5) 
    ax.set_ylim(-0.8, 5) # 稍微往下擴張給 Accuracy 空間
    ax.axis('off')

    # 顏色定義
    c_header = '#E0FBFC'
    c_green = '#D8F3DC'
    c_red = '#FFE5EC'
    c_yellow = '#FFF4E0'
    c_metric = '#F0EFFF'
    c_acc = '#E9ECEF' # 正確率使用淺灰色區塊

    # 3. 繪製區塊 [x, y, w, h, color, text, fontsize]
    rects = [
        # 上方預測標籤與總數 (Predicted Condition)
        (1.5, 3.4, 2.0, 0.4, c_header, "Predicted condition", 12),
        (1.5, 2.9, 1.0, 0.5, c_header, f"Predicted Positive\n {PP}", 11),
        (2.5, 2.9, 1.0, 0.5, '#BEE9E8', f"Predicted Negative\n {PN}", 11),
        
        # 左側實際標籤 (Actual Condition)
        (0.1, 0.5, 0.7, 2.4, c_yellow, "Actual\ncondition", 12),
        (0.8, 1.7, 0.7, 1.2, c_yellow, f"Real \n Positive\n  (MI)\n {P}", 10),
        (0.8, 0.5, 0.7, 1.2, c_yellow, f"Real \n Negative\n  (not MI)\n {N}", 10),
        
        # 核心矩陣 (TP, FN, FP, TN)
        (1.5, 1.7, 1.0, 1.2, c_green, f"True Positive (TP)\n{tp}\n(Hit)", 11),
        (2.5, 1.7, 1.0, 1.2, c_red, f"False Negative (FN)\n{fn}\n(Miss)", 11),
        (1.5, 0.5, 1.0, 1.2, c_red, f"False Positive (FP)\n{fp}\n(False Alarm)", 11),
        (2.5, 0.5, 1.0, 1.2, c_green, f"True Negative (TN)\n{tn}\n(Correct Rejection)", 11),
        
        # 右側率指標
        (3.5, 1.7, 1.0, 1.2, c_metric, f"Sensitivity (TPR)\n{tpr:.2%}", 11),
        (3.5, 0.5, 1.0, 1.2, c_metric, f"Specificity (TNR)\n{tnr:.2%}", 11),
        
        # 底部預測指標 (PPV / NPV)
        (1.5, 0.0, 1.0, 0.5, '#F8F9FA', f"PPV (Precision): \n{ppv:.2%}", 10),
        (2.5, 0.0, 1.0, 0.5, '#F8F9FA', f"NPV: \n{npv:.2%}", 10),
        
        # 右下角正確率區塊 (Accuracy)
        (3.5, 0.0, 1.0, 0.5, c_acc, f"Accuracy\n{acc:.2%}", 11)
    ]

    for x, y, w, h, c, txt, fs in rects:
        rect = plt.Rectangle((x, y), w, h, facecolor=c, edgecolor='#457B9D', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs, linespacing=1.6)

    plt.title(f"ECG MI Detection Performance\n(Total Samples: {Total} | Threshold: {threshold})", 
              fontsize=15, pad=00, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f">>> 專業版報表已儲存至：{save_path}")
    plt.show()    

    
