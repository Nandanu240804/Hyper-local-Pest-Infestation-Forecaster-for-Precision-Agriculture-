import torch
from torchvision import transforms
from PIL import Image
import json

class PestPredictor:
    def __init__(self, model_path, class_names, device='cuda'):
        self.device = device
        self.class_names = class_names
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = PestDetectionModel(num_classes=len(self.class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def get_transform(self):
        """Get the image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=5):
        """Predict pest class for a given image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Get class names and probabilities
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            results.append({
                'rank': i + 1,
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'confidence': float(prob)
            })
        
        return results