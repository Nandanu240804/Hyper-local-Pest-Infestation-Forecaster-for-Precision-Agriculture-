# backend/app.py
import os
import sys
import io
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Ensure scr package is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
SCR_DIR = ROOT_DIR / "scr"
if str(SCR_DIR) not in sys.path:
    sys.path.insert(0, str(SCR_DIR))

from model import create_model
from model_utils import load_model, predict_from_bytes, build_label_map_if_missing
from hotspot import generate_hotspots

app = Flask(__name__, static_folder=str(ROOT_DIR / "frontend" / "static"),
            template_folder=str(ROOT_DIR / "frontend"))

# Config
BACKEND_MODEL_PATH = str(SCR_DIR / "models" / "trained_models" / "best_pest_model.pth")
LABEL_CSV_PATH = str(ROOT_DIR / "ip102_v1.1" / "train.csv")
LABELS_JSON = str(ROOT_DIR / "frontend" / "labels.json")

NUM_CLASSES = 102
DEVICE = "cuda" if (create_model and __import__("torch").cuda.is_available()) else "cpu"
BACKBONE = "b4"
PRETRAINED = False

# Build label map if not present
if not os.path.exists(LABELS_JSON):
    build_label_map_if_missing(LABEL_CSV_PATH, LABELS_JSON, num_classes=NUM_CLASSES)

# Load labels.json
with open(LABELS_JSON, "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)

print("Loading model (this may take a few seconds)...")
model = load_model(BACKEND_MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE, backbone="efficientnet_b4")
print("Model loaded.")

def idx_to_label(idx):
    """Convert numeric class idx -> readable label name"""
    key = str(idx)
    return LABEL_MAP.get(key, f"class_{idx}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/labels.json")
def labels_file():
    return send_from_directory(str(ROOT_DIR / "frontend"), "labels.json")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files["file"]
    image_bytes = io.BytesIO(file.read())

    # Optional lat/lon
    lat = request.form.get("lat", None)
    lon = request.form.get("lon", None)
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except Exception:
        lat, lon = None, None

    # Run prediction
    topk = predict_from_bytes(model, image_bytes, device=DEVICE, top_k=5)

    preds = []
    for idx, prob in topk:
        preds.append({
            "class_id": int(idx),
            "label": idx_to_label(idx),
            "confidence": float(prob)
        })

    main_pred = preds[0] if preds else {"class_id": -1, "label": "unknown", "confidence": 0.0}

    if lat is not None and lon is not None:
        detection_location = {"lat": lat, "lon": lon}
        hotspots = generate_hotspots(lat, lon, main_pred["confidence"])
    else:
        detection_location, hotspots = None, []

    resp = {
        "pest_count": 1,
        "pest_id": main_pred["class_id"],
        "pest_name": main_pred["label"],
        "confidence": main_pred["confidence"],
        "detection_location": detection_location,
        "predicted_hotspots": hotspots,
        "top_predictions": preds
    }
    return jsonify(resp)

from gradcam import generate_gradcam
import base64

@app.route("/gradcam", methods=["POST"])
def gradcam_endpoint():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    file = request.files["file"]
    b = io.BytesIO(file.read())
    # Optional: accept target class id via form
    cls = request.form.get("class_id", None)
    cls = int(cls) if cls is not None else None
    overlay_pil, heatmap = generate_gradcam(model, b, target_class=cls, device=DEVICE, layer_name=None, img_size=380)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return jsonify({"overlay_png_b64": b64})




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
