# backend/gradcam.py  (example helper)
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2

def get_transform(img_size=380):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def generate_gradcam(model, image_bytes, target_class=None, device='cpu', layer_name=None, img_size=380):
    """
    model: nn.Module (should be in eval mode)
    image_bytes: BytesIO of image file
    target_class: integer class id (if None use top-1)
    layer_name: name of convolutional layer to hook (if None, auto-select)
    returns: (overlay_rgb_pil, heatmap_uint8) as PIL, numpy
    """
    model.to(device).eval()
    img = Image.open(image_bytes).convert('RGB')
    transform = get_transform(img_size)
    x = transform(img).unsqueeze(0).to(device)

    # Find a feature map layer if not provided (look for last 'features' conv)
    if layer_name is None:
        # attempt common names
        for n, m in reversed(list(model.named_modules())):
            if isinstance(m, torch.nn.Conv2d):
                layer_name = n
                break
    if layer_name is None:
        raise RuntimeError("Could not find conv layer to hook for Grad-CAM")

    fmap = None
    grads = None
    def fmap_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()
    def grad_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    # register hooks
    target_module = dict(model.named_modules())[layer_name]
    fh = target_module.register_forward_hook(fmap_hook)
    gh = target_module.register_backward_hook(grad_hook)

    # forward
    out = model(x)
    probs = F.softmax(out, dim=1)
    if target_class is None:
        target_class = int(torch.argmax(probs, dim=1).item())

    # backward
    model.zero_grad()
    loss = out[0, target_class]
    loss.backward()

    # remove hooks
    fh.remove(); gh.remove()

    # grads: [B,C,H,W], fmap: [B,C,H,W]
    weights = grads.mean(dim=(2,3), keepdim=True)  # global avg pool per channel
    cam = (weights * fmap).sum(dim=1, keepdim=True)  # [B,1,H,W]
    cam = torch.relu(cam)
    cam = cam[0,0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0..1
    cam = cv2.resize(cam, (img_size, img_size))
    heatmap = (cam * 255).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1]  # BGR->RGB

    # overlay on original (resized)
    orig = np.array(img.resize((img_size, img_size))).astype('uint8')
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    # to PIL
    from PIL import Image
    overlay_pil = Image.fromarray(overlay)
    return overlay_pil, heatmap
