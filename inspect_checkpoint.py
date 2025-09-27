# inspect_checkpoint.py
import torch, json, os
from pathlib import Path

CKPT_PATH = Path("scr/models/trained_models/best_pest_model.pth")  # adjust if different

if not CKPT_PATH.exists():
    print("Checkpoint not found at", CKPT_PATH)
    raise SystemExit(1)

ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
# get state dict
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt

print("Loaded checkpoint:", CKPT_PATH)
print("Total keys in state_dict:", len(sd.keys()))
# print linear layers and their shapes
linear_info = []
for k, v in sd.items():
    if "linear" in k.lower() or "fc" in k.lower() or "classifier" in k.lower() or "head" in k.lower():
        try:
            s = tuple(v.shape)
            linear_info.append((k, s))
        except Exception:
            pass

print("\nDetected candidate linear layer shapes (key, shape):")
for k, s in linear_info:
    print(f"  {k:60s}  {s}")

# also print top-level parameter count
total_params = sum(p.numel() for p in sd.values() if hasattr(p, "numel"))
print("\nTotal parameter-like entries (note: rough):", total_params)

# heuristic: check for known EfficientNet feature dims
# common penultimate dims: B0 -> 1280, B4 -> 1792
found_dims = set()
for _, s in linear_info:
    if len(s) >= 2:
        found_dims.add(s[1])  # in_features usually at index 1 for weight (out,in)

print("\nFound in_features candidates:", found_dims)
if 1280 in found_dims:
    print("\nHeuristic: 1280 found -> likely EfficientNet-B0 (backbone='b0').")
if 1792 in found_dims:
    print("\nHeuristic: 1792 found -> likely EfficientNet-B4 (backbone='b4').")

# Save a tiny json with results to inspect later
out = {
    "checkpoint": str(CKPT_PATH),
    "linear_info": [(k, list(s)) for k, s in linear_info],
    "found_in_features": list(found_dims)
}
with open("inspect_checkpoint_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nWrote inspect_checkpoint_result.json")
