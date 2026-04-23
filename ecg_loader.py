# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:45:38 2025
ecg_loader.py

Single entry-point ECG loader that auto-detects file type by extension and returns:
    Fs, data, label

Supported:
- WFDB: .hea/.dat or record name (no extension)
- MATLAB: .mat
- EDF/BDF: .edf/.bdf

Output:
- data: numpy.ndarray with shape (n_samples, n_channels)

Dependencies (install as needed):
- numpy (required)
- wfdb (for WFDB)   print(wfdb.__version__)
- scipy (for .mat)
- mne or pyedflib (for EDF/BDF)

Example:
    from ecg_loader import load_ecg

    Fs, data, label = load_ecg("record001.hea")
    Fs, data, label = load_ecg("ecg.mat")
    Fs, data, label = load_ecg("sleep.edf")
@author: BOX
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Union, Any, Dict
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["load_ecg", "DEFAULT_12LEAD"]
DEFAULT_12LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _as_list_label(label: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if label is None:
        return None
    if isinstance(label, str):
        return [label]
    return list(label)


def _clean_ch_name(name: str) -> str:
    s = str(name).strip()
    # 常見前綴/符號清掉
    s = s.replace("ECG", "").replace("ecg", "")
    s = s.replace("Lead", "").replace("lead", "")
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    s = s.upper()
    # AVR/AVL/AVF 一律用 aVR/aVL/aVF 形式（最後再轉回）
    if s in {"AVR", "A V R"}:
        return "AVR"
    if s in {"AVL", "A V L"}:
        return "AVL"
    if s in {"AVF", "A V F"}:
        return "AVF"
    return s


def _normalize_labels(labels: Optional[Sequence[str]], n_ch: int) -> List[str]:
    """
    1) 若 labels 缺或長度不符 -> 用預設命名（12導優先）
    2) 若 labels 存在 -> 盡量把常見同義名正規化成標準 12 導名稱
    """
    if not labels or len(labels) != n_ch:
        if n_ch == 12:
            return DEFAULT_12LEAD.copy()
        return [f"CH{idx+1}" for idx in range(n_ch)]

    cleaned = [_clean_ch_name(x) for x in labels]

    # 同義映射（可再自行擴充）
    # 例如：I/II/III、AVR/AVL/AVF、V1~V6
    mapped: List[str] = []
    for s in cleaned:
        if s in {"I", "1"}:
            mapped.append("I")
        elif s in {"II", "2"}:
            mapped.append("II")
        elif s in {"III", "3"}:
            mapped.append("III")
        elif s == "AVR":
            mapped.append("aVR")
        elif s == "AVL":
            mapped.append("aVL")
        elif s == "AVF":
            mapped.append("aVF")
        elif s.startswith("V") and len(s) <= 3:
            # V1..V6
            mapped.append(s)
        else:
            mapped.append(s)

    # 若剛好是 12 導但名稱亂七八糟，也不要強迫覆蓋；保持檔內資訊
    return mapped


def _ensure_2d_samples_channels(x: np.ndarray) -> np.ndarray:
    """
    讓輸出固定為 (n_samples, n_channels)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"data 維度必須是 1D 或 2D，目前是 {x.ndim}D，shape={x.shape}")

    r, c = x.shape
    # 常見情況：MAT/WFDB/EDF 可能是 (n_channels, n_samples)
    # 若 rows 很小且像 channel 數 -> 轉置
    if r <= 68 and c > r:
        # 這裡偏向把 (channels, samples) 轉成 (samples, channels)
        # 但若其實就是 (samples, channels) 且 samples<=68 會不轉置
        # -> 對 ECG 幾乎都合理
        return x.T
    return x

def _unit_to_mv_factor(unit: str) -> float:
    """
    Return multiplicative factor to convert a value expressed in `unit` into mV.

    Examples:
        V  -> factor 1000
        mV -> factor 1
        uV/µV -> factor 0.001
    """
    if unit is None:
        return 1.0
    u = str(unit).strip()
    # normalize micro sign variants
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace(" ", "")
    u_up = u.upper()

    if u_up in {"MV"}:
        return 1.0
    if u_up in {"V", "VOLTS", "VOLT"}:
        return 1000.0
    if u_up in {"UV", "MICROVOLT", "MICROVOLTS"}:
        return 0.001
    if u_up in {"NV", "NANOVOLT", "NANOVOLTS"}:
        return 1e-6

    # Unknown -> assume already mV
    return 1.0

def _convert_channels_to_mv(data_samples_channels: np.ndarray, units: Optional[Sequence[str]]) -> np.ndarray:
    """
    Convert per-channel units to mV (returns a new array).
    data shape: (n_samples, n_channels)
    units: list of unit strings per channel (len==n_channels) or None.
    """
    data = np.asarray(data_samples_channels)
    if units is None:
        return data
    if len(units) != data.shape[1]:
        return data

    out = data.astype(np.float32, copy=True)
    for i, u in enumerate(units):
        out[:, i] *= _unit_to_mv_factor(u)
    return out




def _load_wfdb(filename: str) -> Tuple[Optional[float], np.ndarray, Optional[List[str]]]:
    """
    讀 WFDB：filename 可傳 .hea/.dat 檔，或 record 名稱（不含副檔名）
    需要套件：wfdb
    """
    try:
        import wfdb  # type: ignore
    except Exception as e:
        raise ImportError("需要安裝 wfdb：pip install wfdb") from e
    
    p = Path(filename)
    physical=True
    #physical=True -> data = p_signal (物理單位)
    #physical=False -> data = d_signal (ADC counts)
    # wfdb.rdrecord 要的是 record name（不含 .hea/.dat）
    record_name = str(p.with_suffix("")) if p.suffix.lower() in {".hea", ".dat"} else str(p)

    header = wfdb.rdheader(str(record_name))
    base_dir = p.parent

    keep_idx = []
    for i, fn in enumerate(header.file_name):
        if not fn:
            continue
        if not fn.lower().endswith(".dat"):
            continue

        fpath = Path(fn)
        if not fpath.is_absolute():
            fpath = base_dir / fpath

        if fpath.exists():
            keep_idx.append(i)

    if not keep_idx:
        raise FileNotFoundError(f"找不到任何存在的 .dat channels。header.file_name={header.file_name}")

    rec = wfdb.rdrecord(str(record_name), channels=keep_idx)

    fs = rec.fs
    data = rec.p_signal if physical else rec.d_signal
    data = _ensure_2d_samples_channels(data) # wfdb p_signal 通常是 (n_samples, n_channels)；這裡保險處理
    label = rec.sig_name  # 導程名稱，例如 ['i','ii',...]
    return fs, data, label




def _load_edf_bdf(filename: str) -> Tuple[Optional[float], np.ndarray, Optional[List[str]]]:
    """
    Read EDF/BDF and convert signal units to **mV**.

    Notes on units:
    - With MNE, raw.get_data() is typically in Volts, and raw._orig_units (if available)
      may indicate the original channel units (e.g., 'uV', 'mV', 'V').
    - With pyedflib, readSignal() returns values in the physical dimension specified in the file header,
      accessible via getPhysicalDimension(i) (e.g., 'uV', 'mV', 'V').

    If units cannot be determined, MNE branch assumes Volts and converts to mV; pyedflib branch leaves as-is.
    """
    # 1) mne
    try:
        import mne  # type: ignore

        raw = mne.io.read_raw_edf(filename, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"]) if raw.info.get("sfreq") is not None else None

        labels = list(raw.ch_names) if raw.ch_names else None

        # (n_channels, n_samples) -> (n_samples, n_channels)
        data = np.asarray(raw.get_data()).T

        # Try to infer original units per channel
        units = None
        try:
            orig_units = getattr(raw, "_orig_units", None)  # dict: {ch_name: unit_str}
            if isinstance(orig_units, dict) and labels is not None:
                units = [orig_units.get(ch, None) for ch in labels]
        except Exception:
            units = None

        # MNE often outputs Volts even if original units were uV.
        if units is not None and any(u is not None for u in units):
            data = _convert_channels_to_mv(data, units)
        else:
            # Assume Volts -> mV
            data = (data * 1000.0).astype(np.float32, copy=False)

        return fs, data, labels
    except Exception:
        pass

    # 2) pyedflib
    try:
        import pyedflib  # type: ignore
    except Exception as e:
        raise ImportError("Install mne or pyedflib for EDF/BDF: pip install mne pyedflib") from e

    f = pyedflib.EdfReader(filename)
    try:
        n_ch = f.signals_in_file
        labels = [f.getLabel(i) for i in range(n_ch)]
        fs_list = [f.getSampleFrequency(i) for i in range(n_ch)]
        fs = float(fs_list[0]) if fs_list else None

        units = []
        for i in range(n_ch):
            try:
                units.append(f.getPhysicalDimension(i))
            except Exception:
                units.append(None)

        # readSignal -> values in physical dimension given by getPhysicalDimension(i)
        sigs = np.vstack([f.readSignal(i) for i in range(n_ch)])  # (n_ch, n_samples)
        data = sigs.T  # (n_samples, n_ch)

        data = _convert_channels_to_mv(data, units)
        return fs, data, labels
    finally:
        f.close()


def _pick_mat_data(mat: Dict[str, Any]) -> Tuple[np.ndarray, Optional[float], Optional[List[str]]]:
    """
    從 .mat dict 內盡量猜出 data / Fs / label
    """
    # Fs key 候選
    fs_keys = ["Fs", "fs", "FS", "sampFreq", "sampling_rate", "samplingRate", "sfreq", "SR"]
    fs: Optional[float] = None
    for k in fs_keys:
        if k in mat:
            try:
                v = mat[k]
                # squeeze 後可能是 array/scalar
                fs = float(np.asarray(v).squeeze())
                break
            except Exception:
                continue

    # label key 候選
    label_keys = ["lead_name", "label", "labels", "ch_names", "chan_names", "channel_names", "sig_name", "lead_names"]
    labels: Optional[List[str]] = None
    for k in label_keys:
        if k in mat:
            v = mat[k]
            try:
                arr = np.asarray(v).squeeze()
                # 可能是 object array / char array
                if arr.dtype.kind in {"U", "S"}:
                    labels = [str(x) for x in arr.tolist()] if arr.ndim > 0 else [str(arr)]
                    break
                if arr.dtype == object:
                    labels = [str(x).strip() for x in arr.tolist()] if arr.ndim > 0 else [str(arr)]
                    break
            except Exception:
                continue

    # data key 候選（先試常見）
    data_keys = ["ecg_signal", "data", "ecg", "ECG", "signal", "signals", "sig", "val", "x", "X"]
    for k in data_keys:
        if k in mat:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.size > 0 and v.ndim in (1, 2):
                data = _ensure_2d_samples_channels(v)
                return data, fs, labels

    # 若沒找到常見 key，就挑「最大的數值 1D/2D 陣列」
    best = None
    best_size = -1
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.size > 0 and v.ndim in (1, 2):
            if np.issubdtype(v.dtype, np.number):
                if v.size > best_size:
                    best = v
                    best_size = v.size

    if best is None:
        raise ValueError("MAT 檔內找不到可用的數值 data（1D/2D ndarray）")

    data = _ensure_2d_samples_channels(best)
    return data, fs, labels


def _load_mat(filename: str) -> Tuple[Optional[float], np.ndarray, Optional[List[str]]]:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as e:
        raise ImportError("需要安裝 scipy：pip install scipy") from e

    mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
    data, fs_in_mat, labels = _pick_mat_data(mat)
    return fs_in_mat, data, labels


def load_ecg(
    filename: str,
    Fs: Optional[float] = None,
    label: Optional[Union[str, Sequence[str]]] = None,
    default_fs: Optional[float] = None,
    prefer_float32: bool = True,
) -> Tuple[Optional[float], np.ndarray, List[str]]:
    """
    依 filename 自動讀取 WFDB / MAT / EDF / BDF，輸出 (Fs, data, label)

    參數
    - filename: 檔名或路徑；WFDB 可傳 .hea/.dat 或 record 名稱（不含副檔名）
    - Fs: 可空；若提供則優先使用（覆蓋檔案內 Fs）
    - label: 可空；若提供則優先使用；否則盡量從檔案內找
    - default_fs: 若檔案內也找不到 Fs，則回傳 default_fs（可為 None）
    - prefer_float32: True 則 data 轉 float32（省記憶體/加速）

    回傳
    - Fs: 取樣率（可能為 None）
    - data: (n_samples, n_channels)
    - label: 長度 = n_channels
    """
    p = Path(filename)
    ext = p.suffix.lower()

    # 讀檔：先拿到 (fs_from_file, data, labels_from_file)
    if ext in {".edf", ".bdf"}:
        fs_file, data, labels_file = _load_edf_bdf(str(p))
    elif ext == ".mat":
        fs_file, data, labels_file = _load_mat(str(p))
    elif ext in {".hea", ".dat"} or ext == "":
        # ext == ""：可能是 wfdb record name
        # 也可能是其他無副檔名檔案；這裡以 WFDB 為優先，失敗再提示
        try:
            fs_file, data, labels_file = _load_wfdb(str(p))
        except Exception as e:
            raise ValueError(
                f"無法以 WFDB 讀取：{filename}。若是 EDF/BDF/MAT 請確認副檔名；"
                f"若是 WFDB 請提供 record 名稱或 .hea/.dat。原始錯誤：{e}"
            ) from e
    else:
        raise ValueError(f"不支援的副檔名：{ext}（支援 .mat/.edf/.bdf/.hea/.dat 或 WFDB record name）")

    fs_out: Optional[int] = None
    ext = Path(filename).suffix.lower()
    
    if ext in {".dat", ".edf", ".bdf"}:
        # 優先使用檔案內 Fs
        fs_tmp = fs_file if fs_file is not None and not (isinstance(fs_file, float) and np.isnan(fs_file)) else None
    else:
        # 其他情況 (mat/txt) 用外部或預設
        if Fs is not None:
            fs_tmp = float(Fs)
        elif default_fs is not None:
            fs_tmp = float(default_fs)
        else:
            fs_tmp = fs_file if fs_file is not None and not (isinstance(fs_file, float) and np.isnan(fs_file)) else None
    
    # 最後轉為整數
    fs_out = None if fs_tmp is None else int(round(fs_tmp))
  
    # label：外部參數優先，其次檔案內；都沒有則用預設（12導優先）
    label_in = _as_list_label(label) if label is not None else (labels_file if labels_file is not None else None)
    labels_out = _normalize_labels(label_in, n_ch=data.shape[1])

    # data dtype
    data = np.asarray(data, dtype=np.float32 if prefer_float32 else np.float64)

    return fs_out, data, labels_out


def plot_12lead_ecg(avgECG,Fs=None, Tpeak=None, Jpos=None):
    """
    avgECG: (12, n_samples) or (n_samples, 12)
    Tpeak, Jpos: array-like length 12, each element is sample index (int) or None
    Fs: sampling rate (int) optional, for x-axis in seconds
    """
    ecg = np.asarray(avgECG)

    # 轉成 (12, n_samples)
    if ecg.ndim != 2:
        raise ValueError(f"avgECG must be 2D, got shape={ecg.shape}")
    # 轉成 (n_channels, n_samples)
    # 常見：輸入是 (n_samples, n_channels)
    if ecg.shape[0] < ecg.shape[1] and ecg.shape[0] <= 68:
        # 可能是 (n_channels, n_samples) 或 (n_samples, n_channels)
        # 如果第二維看起來像channel數（<=64）就轉置
        if ecg.shape[1] <= 68:
            ecg = ecg.T

    n_ch, n = ecg.shape
    if n_ch < 12:
        raise ValueError(f"Need at least 12 channels, got {n_ch}")

    # 只取前 12 導
    ecg = ecg[:12, :]
     # 你指定的排列順序（subplot位置從左到右、上到下）
    order = [0, 3, 6, 9,
             1, 4, 7, 10,
             2, 5, 8, 11]
    n = ecg.shape[1]
    labels = DEFAULT_12LEAD

    # x 軸：樣本或秒
    if Fs is None:
        x = np.arange(n)
        xlab = "Samples"
    else:
        x = np.arange(n) / float(Fs)
        xlab = "Time (s)"

    # Tpeak/Jpos：若給了也只取前 12
    Tpeak = None if Tpeak is None else np.asarray(Tpeak, dtype=int)[:12]
    Jpos  = None if Jpos  is None else np.asarray(Jpos,  dtype=int)[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for ax_i, lead_i in enumerate(order):
        ax = axes[ax_i]
        y = ecg[lead_i, :]

        ax.plot(x, y, linewidth=1)

        if Tpeak is not None:
            t = int(Tpeak[lead_i])
            if 0 <= t < n:
                ax.plot(x[t], y[t], "o")

        if Jpos is not None:
            j = int(Jpos[lead_i])
            if 0 <= j < n:
                ax.plot(x[j], y[j], "o")

        ax.set_title(labels[lead_i])
        ax.grid(True, alpha=0.3)

    # 空白格（理論上不會用到）
    for k in range(12, len(axes)):
        axes[k].axis("off")

    fig.supxlabel(xlab)
    fig.supylabel("Amplitude")
    fig.tight_layout()
    plt.show()