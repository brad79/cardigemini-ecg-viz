"""
download_models.py
從 GitHub Release 下載 MI 深度學習模型權重檔。

使用方式:
    python download_models.py
"""
import os
import urllib.request

REPO = "brad79/cardigemini-ecg-viz"
TAG  = "v1.0.0"
MODELS = [
    "MI_nonMI_model_10sec.pth",
    "MI_nonMI_model_2.5sec.pth",
]

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(DEST_DIR, exist_ok=True)

BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

for name in MODELS:
    dest = os.path.join(DEST_DIR, name)
    if os.path.exists(dest):
        print(f"[skip] {name} already exists")
        continue
    url = f"{BASE_URL}/{name}"
    print(f"Downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  Saved to models/{name}  ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        print(f"  Please download manually from:\n  {url}")

print("Done.")
