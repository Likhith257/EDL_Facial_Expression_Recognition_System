import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms, models
from PIL import Image
import yaml

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        w_norm = nn.functional.normalize(self.fc.weight, p=2, dim=1)
        logits = torch.matmul(x_norm, w_norm.t())
        
        if labels is not None:
            theta = torch.acos(torch.clamp(logits, -1.0, 1.0))
            target_logit = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.view(-1,1), 1.0)
            logits = logits * (1 - one_hot) + target_logit * one_hot
            
        logits *= self.s
        return logits

class EmotionPredictor:
    """Real-time emotion prediction using ArcFace"""
    def __init__(self, checkpoint_path, config_path=None, device='mps'):
        self.device_type = device
        
        # Setup device
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Default classes
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        
        # Load model architecture
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.head = ArcFaceHead(num_features, len(self.class_names))
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.head.load_state_dict(checkpoint['head'])
        else:
            # Fallback if saved differently (e.g. entire model)
            # But our train.py saves as dict with 'backbone' and 'head'
            print("Warning: Unexpected checkpoint format")
        
        self.backbone.to(self.device)
        self.head.to(self.device)
        self.backbone.eval()
        self.head.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112,112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Face detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
    
    def preprocess_face(self, face_img):
        """Preprocessing for real-world images"""
        # Convert to RGB
        if face_img.mode != 'RGB':
            face_img = face_img.convert('RGB')
        
        return face_img
    
    def predict_image(self, image, apply_enhancement=True):
        """Predict emotion from PIL Image"""
        if apply_enhancement:
            image = self.preprocess_face(image)
        
        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            features = self.backbone(img_tensor)
            logits = self.head(features, labels=None)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        emotion = self.class_names[predicted.item()]
        confidence = confidence.item()
        
        return emotion, confidence

def main():
    # Simple test
    print("ArcFace Predictor Main")

if __name__ == "__main__":
    main()
