import cv2
import numpy as np

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for DeepFace."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def load_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return preprocess_image(img, target_size)