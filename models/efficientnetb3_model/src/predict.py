
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import yaml

from .model import create_model


class EmotionPredictor:
    """Real-time emotion prediction"""
    def __init__(self, checkpoint_path, config_path=None, device='mps'):
        self.device_type = device
        
        # Setup device
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        self.class_names = config.get('class_names', 
            ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'])
        self.img_size = config.get('img_size', 224)
        
        # Load model
        self.model = create_model(
            num_classes=len(self.class_names),
            pretrained=False,
            device=self.device_type
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Transforms with improved preprocessing for real-world images
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Multiple face detectors for better robustness
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
    
    def preprocess_face(self, face_img):
        """Enhanced preprocessing for real-world images"""
        # Convert to grayscale for processing
        if len(face_img.size) == 3:
            gray = face_img.convert('L')
        else:
            gray = face_img
        
        # Convert back to RGB
        if face_img.mode != 'RGB':
            face_img = face_img.convert('RGB')
        
        # Convert to numpy for enhancement
        img_array = np.array(face_img)
        
        # Enhance contrast and brightness for better recognition
        # Apply histogram equalization on luminance channel
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # Reduce noise
        img_enhanced = cv2.bilateralFilter(img_enhanced, 5, 50, 50)
        
        # Convert back to PIL
        face_pil = Image.fromarray(img_enhanced)
        
        return face_pil
    
    def predict_image(self, image, apply_enhancement=True):
        """Predict emotion from PIL Image with optional preprocessing"""
        # Apply enhanced preprocessing for real-world images
        if apply_enhancement:
            image = self.preprocess_face(image)
        
        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)
        
        emotion = self.class_names[predicted.item()]
        confidence = confidence.item()
        
        return emotion, confidence
    
    def predict_webcam(self):
        """Real-time webcam prediction"""
        cap = cv2.VideoCapture(0)
        print("📷 Starting webcam... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face
                face_img = frame[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                
                # Predict
                emotion, confidence = self.predict_image(face_pil)
                
                # Draw rectangle
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show frame
            cv2.imshow('Emotion Recognition - EfficientNet-B3', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def predict_file(self, image_path):
        """Predict from image file"""
        image = Image.open(image_path).convert('RGB')
        emotion, confidence = self.predict_image(image)
        
        print(f"Image: {image_path}")
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        
        return emotion, confidence


def main():
    """Main prediction"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints'
    CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
    
    checkpoint_path = CHECKPOINT_DIR / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        return
    
    print("🚀 Loading EfficientNet-B3 model...")
    predictor = EmotionPredictor(
        checkpoint_path=checkpoint_path,
        config_path=CONFIG_PATH,
        device='mps'
    )
    
    print("✅ Model loaded!")
    print("\nStarting webcam prediction...")
    predictor.predict_webcam()


if __name__ == "__main__":
    main()
