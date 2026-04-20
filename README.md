# CardiGemini — 次世代心臟缺血風險評估與成像技術

12 導程 ECG 上傳 → 訊號品質評估 → 波形平均 → 深度學習 MI 篩檢 → SRC 缺血定位 → 3D 心臟模型 + Bull's-Eye Polar Map

## 快速開始

### 1. 安裝 Python 依賴

建議使用 Python 3.11。

```bash
pip install -r requirements.txt
```

### 2. 下載模型權重

```bash
python download_models.py
```

執行後會在 `models/` 目錄下建立：
- `MI_nonMI_model_10sec.pth`（117 MB，用於 .mat / .edf / .dat 格式）
- `MI_nonMI_model_2.5sec.pth`（117 MB，用於 .pdf 格式）

### 3. 啟動應用程式

```bash
streamlit run app.py
```

瀏覽器自動開啟 `http://localhost:8501`

---

## 支援的 ECG 格式

| 格式 | 副檔名 | 備注 |
|------|--------|------|
| WFDB | `.dat` + `.hea` | PhysioNet 標準格式 |
| MATLAB | `.mat` | 常見科研格式 |
| EDF/BDF | `.edf`, `.bdf` | 臨床長程 ECG |
| PDF | `.pdf` | 向量圖 ECG（自動提取波形）|

---

## 系統需求

- Python 3.11+
- CUDA GPU（可選，無 GPU 自動使用 CPU）
- 記憶體：建議 8 GB RAM 以上

---

## 專案結構

```
cardigemini-ecg-viz/
├── app.py                  # Streamlit 主程式（入口）
├── step2_quality.py        # 訊號品質評估
├── mesh3DIschemia.py       # 3D 心臟 PLY 網格視覺化
├── ecg_loader.py           # ECG 檔案載入（多格式）
├── ecg_engine.py           # 深度學習推論引擎
├── ECGDataLoader.py        # 多格式統一資料載入器
├── dataset.py              # ECGProcessor 訊號預處理
├── net1d.py                # Net1D 模型架構
├── label_decoder.py        # SRC 標籤解碼（26 區域 × 5 嚴重度）
├── predict_function.py     # SRC 稀疏表示分類
├── download_models.py      # 模型下載腳本
├── requirements.txt
├── data/
│   ├── src_trainset.npz    # SRC 訓練集（2.4 MB）
│   ├── ventrical.ply       # 心室 3D 網格
│   ├── EpicPos_C.mat       # 心外膜定位映射
│   └── bullseye/           # 131 張 Bull's-Eye PNG
├── models/                 # 模型權重（執行 download_models.py 後生成）
│   ├── MI_nonMI_model_10sec.pth
│   └── MI_nonMI_model_2.5sec.pth
└── assets/                 # 冠狀動脈解剖示意圖
```
