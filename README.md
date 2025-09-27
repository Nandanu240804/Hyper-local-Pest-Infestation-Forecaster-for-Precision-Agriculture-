# PestEye — Hyper-local Pest Infestation Forecaster for Precision Agriculture

**PestEye** is a precision-agriculture project that detects pest species from crop images and forecasts short-term (24–72 hr) spatial spread (hotspots). It combines a computer-vision pest classifier (EfficientNet backbone), a Flask backend serving predictions, and a Leaflet-based frontend for visualization and interactive hotspot generation.

---

## 🔗 Project demo / repo structure

```
pest-detection-system/
├── backend/                   # Flask API + backend utilities
├── frontend/                  # Static frontend (index.html + assets)
├── ip102_v1.1/                # (not in repo) IP102 dataset (images + csv)
├── scr/                       # training, model, data loader, evaluation scripts
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── train_model.py
│   ├── inference.py
│   └── evaluate_dump.py
├── scripts/                   # helpful scripts (download, generate labels, etc.)
├── requirements.txt
└── README.md
```

> **Note:** large binary files (dataset, model weights) are intentionally not included in the repo. See Dataset & Model sections below for where to put them.

---

## 🔎 Dataset (Kaggle)

This project uses the **IP102** dataset. The dataset is large, so the repo does **not** include it directly.

Kaggle dataset used in this repo:
**[https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset)**

### Download via Kaggle CLI (recommended)

1. Install Kaggle CLI:

```bash
pip install kaggle
```

2. Place your Kaggle API token:

* Download `kaggle.json` from [https://www.kaggle.com/](https://www.kaggle.com/) → Account → API → Create New API Token.
* Save `kaggle.json` to:

  * `~/.kaggle/kaggle.json` (Linux/Mac)
  * `C:\Users\<USERNAME>\.kaggle\kaggle.json` (Windows)
* Set permissions (Linux/Mac):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

3. Download and extract:

```bash
# from project root
kaggle datasets download -d rt lmhjbn/ip02-dataset   # use exact dataset id shown on Kaggle page
# unzip (example)
unzip ip02-dataset.zip -d ip102_v1.1
```

> If the dataset archive name differs, adjust commands accordingly. After extraction you should have `./ip102_v1.1/images/` and `./ip102_v1.1/train.csv`, etc.

### Manual download

If you prefer, download the dataset archive from Kaggle via browser, extract it to the project root as `ip102_v1.1/`.

---

## ⚙️ Environment & Installation

Recommended: Python 3.8+ in a virtual environment.

```bash
# create venv (Windows)
python -m venv venv
venv\Scripts\activate

# mac/linux
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

**Notes:**

* If you plan to use the GPU make sure to install a PyTorch build that matches your CUDA version (see [https://pytorch.org/](https://pytorch.org/)).
* If `opencv-python` (cv2) is missing, install:

  ```bash
  pip install opencv-python
  ```

---

## 🧠 Model weights & where to place them

Trained model weights (if you have them) should be placed in:

```
scr/models/trained_models/best_pest_model.pth
scr/models/trained_models/final_pest_model.pth
```

If you do not have weights, either:

* Download from the external storage you used (Google Drive/Hugging Face), or
* Train the model using `train_model.py` (see `scr/train_model.py`) — requires dataset and GPU recommended.

---

## 🚀 Run the backend (Flask API)

From project root:

```bash
cd backend
# optional: set environment variables:
# Windows (PowerShell)
# $env:FLASK_APP="app.py"; $env:FLASK_ENV="development"
python app.py
```

Default server: `http://localhost:5000/`

### Important: Geolocation & HTTPS

* Browsers restrict `navigator.geolocation` on **insecure** contexts when not on `localhost`. To use "Use Current Location" in the frontend:

  * Open the frontend at `http://localhost:5000/` (served by Flask) OR
  * Serve via HTTPS (self-signed cert or a tunnel like `ngrok`) OR
  * Enter coordinates manually.

---

## 🧩 Run the frontend

The frontend is in `frontend/` (or `backend` serves `frontend` as template). With backend running:

* Open: `http://localhost:5000/` in your browser.

If you want to open the static HTML file directly, open `frontend/index.html` in a browser (hotspots requiring backend prediction won't work).

---

## 🔧 Generate `labels.json` (map numeric → pest names)

If you want readable pest names (instead of `class_54`) generate `labels.json` from your `train.csv`. Save this script as `scripts/generate_labels.py`:

```python
# scripts/generate_labels.py
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv = ROOT / "ip102_v1.1" / "train.csv"
out = ROOT / "frontend" / "labels.json"

df = pd.read_csv(csv, header=None)  # adjust header if your CSV has headers
# assume second column is label id or label text; adapt if data format differs
# If labels are numeric: map index -> human name (you may need a manual mapping)
labels = df.iloc[:,1].unique().tolist()
labels_map = {str(i): f"class_{i}" for i in range(len(labels))}
# If train.csv stores class names in col 2 you can build direct map:
# labels_map = { str(i): name for i,name in enumerate(sorted(labels)) }

with open(out, "w", encoding="utf-8") as f:
    json.dump(labels_map, f, ensure_ascii=False, indent=2)

print("Wrote", out)
```

Run:

```bash
python scripts/generate_labels.py
```

Then restart backend so it can load `frontend/labels.json`.

---

## 🔬 Evaluate model / reproduce evaluation

To run the provided evaluation:

```bash
python scr/evaluate_dump.py
```

This will compute precision/recall/F1 and write misclassified images to `scr/debug/misclassified/`.

---

## 🛠 Common troubleshooting

* **`ModuleNotFoundError: No module named 'cv2'`** → `pip install opencv-python`.
* **CUDA OOM** → reduce `batch_size`, reduce image size (`img_size`), or use `num_workers=0`.
* **Windows: DataLoader worker spawn errors** → use `num_workers=0`.
* **Browser geolocation blocked** → serve frontend via `localhost` or HTTPS (ngrok).
* **Git push failed due to large files** → remove dataset/model from history and push minimal repo. Use Git LFS or external hosting for heavy files.

---

## 🧾 API endpoints (backend)

* `GET /` → serve frontend `index.html`

* `GET /labels.json` → label mapping for frontend

* `POST /predict` → form upload, fields:

  * `file` (image) required
  * optional `lat`, `lon` to get hotspot circles
  * Response (JSON): `pest_name`, `confidence`, `detection_location`, `predicted_hotspots`, `top_predictions`

* `POST /hotspot` (if implemented) → compute hotspot forecasts given `lat, lon, class_id, confidence, hours, scale`.

---

## ✅ Suggestions for a strong demo / next steps

1. **Use Grad-CAM** to visualize model attention on images (helps explain wrong predictions).
2. **Temperature scaling or per-class thresholds** to reduce false positives for demo.
3. **Add a “healthy / no pest” threshold** so the model reports "no pest" when confidence is low.
4. **Hotspot forecasting**: implement a simple diffusion model (or ConvLSTM for spatio-temporal forecasts) to produce 24/48/72-hour maps.
5. **Polish frontend** (UX improvements, Grad-CAM toggle, forecast sliders).

---

## 📄 License & attribution

This project is released under the **MIT License** — include a `LICENSE` file in the repo if you want to make that official.

The dataset `IP102` is provided by its authors — follow their license and citation instructions from the Kaggle page.

---

