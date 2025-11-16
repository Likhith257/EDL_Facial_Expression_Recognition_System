from deepface import DeepFace
import cv2
import os
from tqdm import tqdm

def predict_emotions(image_path, model_path=None, output_dir='runs/predictions'):
    """
    Predict emotions on a single image or folder of images.
    Args:
        image_path (str): Path to image or folder.
        model_path (str): Path to fine-tuned model (optional, uses default if None).
        output_dir (str): Where to save predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    
    # Handle single image or folder
    if os.path.isfile(image_path):
        image_paths = [image_path]
    else:
        image_paths = []
        for root, _, files in os.walk(image_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
    
    # Predict for each image
    for img_path in tqdm(image_paths, desc="Predicting emotions"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                detector_backend='retinaface',
                model_name='SFace' if not model_path else None,
                enforce_detection=True,
                silent=True
            )
            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]
            predictions.append((img_path, emotion, confidence))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save predictions
    with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
        for path, emotion, conf in predictions:
            f.write(f"{path}: {emotion} ({conf:.2f})\n")
    
    return predictions