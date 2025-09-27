# scr/run_infer.py
import torch, io
from PIL import Image
from model import create_model
from torchvision import transforms

MODEL_PATH = "models/trained_models/best_pest_model.pth"   # relative to scr/
IMG = "../ip102_v1.1/images/some_image.jpg"                # adjust image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=102, device=device, backbone='b0', pretrained=False)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(IMG).convert("RGB")
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = model(x)
    probs = torch.nn.functional.softmax(out, dim=1)
    top_p, top_i = torch.topk(probs, 5)
print(top_p.cpu().numpy(), top_i.cpu().numpy())
