# backend/model_utils.py
import os
import io
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# We lazy-import create_model from scr/model inside functions to avoid import-time issues.
# backend/model_utils.py (patch for load_model)
def load_model(model_path, num_classes=102, device="cpu", backbone="b0"):
    """
    Load model from checkpoint. Accepts backbone values:
      'b0', 'b4', 'efficientnet_b0', 'efficientnet_b4'
    """
    # normalize backbone name to the values expected by scr.model.create_model
    if backbone is None:
        backbone = "b0"
    b = str(backbone).lower()
    if "b4" in b:
        backbone_norm = "b4"
    elif "b0" in b:
        backbone_norm = "b0"
    else:
        # fallback
        backbone_norm = "b0"

    # import create_model from scr.model
    from model import create_model
    device = torch.device(device)
    model = create_model(num_classes=num_classes, device=device, backbone=backbone_norm, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# Preprocessing: match training transforms
IMG_SIZE = 224
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_from_bytes(model, image_bytes_io, device="cpu", top_k=5):
    """
    Accepts a file-like (BytesIO) object, returns list of (index, prob) of top_k predictions.
    """
    device = torch.device(device)
    image_bytes_io.seek(0)
    img = Image.open(image_bytes_io).convert("RGB")
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        topk = torch.topk(probs, k=top_k, dim=1)
    top_probs = topk.values.cpu().numpy().tolist()[0]
    top_idxs = topk.indices.cpu().numpy().tolist()[0]
    return list(zip(top_idxs, top_probs))

def build_label_map_if_missing(train_csv_path, out_json_path, num_classes=102):
    """
    Build a simple label map from train.csv second column. If the CSV second column contains
    numeric labels (0..N-1), attempt to map to 'class_{i}'. If it contains names, use them.
    Saves to out_json_path.
    """
    if os.path.exists(out_json_path):
        return

    labels = []
    try:
        with open(train_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # assume 1st column image, 2nd column label (as in your dataset)
            for row in reader:
                if len(row) < 2:
                    continue
                labels.append(row[1].strip())
    except Exception:
        # fallback: create generic map
        mapping = {str(i): f"class_{i}" for i in range(num_classes)}
        with open(out_json_path, "w", encoding="utf-8") as of:
            json.dump(mapping, of, indent=2)
        return

    # If labels look numeric, create map using numeric -> name if names exist in dataset
    # Detect if label values are numeric ints
    unique_labels = list(dict.fromkeys(labels))  # preserve order
    # If values are names (not ints), number them 0..N-1
    mapping = {}
    if all(item.isdigit() for item in unique_labels):
        # numeric labels; map 'i' -> 'class_i' (or if you have a separate mapping file, you can adjust)
        for item in unique_labels:
            mapping[item] = f"class_{item}"
    else:
        # non-numeric labels (likely names). assign index numbers sequentially
        for idx, name in enumerate(unique_labels):
            mapping[str(idx)] = name
        # If number of mapped names < num_classes, pad with generic names
        for i in range(len(unique_labels), num_classes):
            mapping[str(i)] = f"class_{i}"

    # Save mapping
    with open(out_json_path, "w", encoding="utf-8") as of:
        json.dump(mapping, of, indent=2)
    return
