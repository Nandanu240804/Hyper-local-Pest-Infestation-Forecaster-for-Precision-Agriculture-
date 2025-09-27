# scr/debug_gradcam.py
import os, io, json
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# import your dataset/model helpers
from data_loader import PestDataset
from model import create_model

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "scr"
MIS_DIR = SCR / "debug" / "misclassified"
OUT_DIR = SCR / "debug" / "misclassified_gradcam"
MODEL_PATH = SCR / "models" / "trained_models" / "best_pest_model.pth"
IMG_DIR = ROOT / "ip102_v1.1" / "images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 380  # same as training

OUT_DIR.mkdir(parents=True, exist_ok=True)

# load model
num_classes = 102
model = create_model(num_classes=num_classes, device=DEVICE)
ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

# helper: find last conv layer name automatically
def find_last_conv(m):
    last = None
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            last = n
    return last

last_conv = find_last_conv(model)
print("Using conv layer:", last_conv)

def generate_gradcam_pil(model, pil_img, target_class=None, device='cpu', layer_name=None, img_size=IMG_SIZE):
    model.to(device).eval()
    img = pil_img.convert('RGB').resize((img_size, img_size))
    x = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float() / 255.0
    # normalize
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x = (x - mean) / std
    x = x.to(device)

    fmap = None
    grads = None
    def fmap_hook(m, inp, out):
        nonlocal fmap
        fmap = out.detach()
    def grad_hook(m, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    target_module = dict(model.named_modules()).get(layer_name, None)
    if target_module is None:
        raise RuntimeError("layer not found")
    fh = target_module.register_forward_hook(fmap_hook)
    gh = target_module.register_backward_hook(grad_hook)

    out = model(x)
    probs = F.softmax(out, dim=1)
    if target_class is None:
        target_class = int(torch.argmax(probs, dim=1).item())

    model.zero_grad()
    loss = out[0, target_class]
    loss.backward()

    fh.remove(); gh.remove()

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam[0,0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = (cam * 255).astype('uint8')
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1]

    orig = np.array(img).astype('uint8')
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    return Image.fromarray(overlay)

# iterate misclassified images
files = sorted([p for p in MIS_DIR.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
for f in tqdm(files):
    try:
        pil = Image.open(f).convert('RGB')
        # optionally parse target_class from filename if present: look for "_predX_"
        pred = None
        parts = f.name.split('_')
        for p in parts:
            if p.startswith('pred'):
                try:
                    pred = int(p.replace('pred',''))
                except:
                    pass
        overlay = generate_gradcam_pil(model, pil, target_class=pred, device=DEVICE, layer_name=last_conv)
        outp = OUT_DIR / (f.stem + '_gradcam.png')
        overlay.save(outp)
    except Exception as e:
        print("failed", f, e)

print("Saved Grad-CAM overlays to", OUT_DIR)
