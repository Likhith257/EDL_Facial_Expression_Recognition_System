from deepface import DeepFace
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os

def evaluate_model(model_path, test_images, test_labels, output_dir='runs/evaluation'):
    """
    Evaluate the model on test data.
    Args:
        model_path (str): Path to fine-tuned model.
        test_images (np.array): Test images.
        test_labels (np.array): Test labels.
        output_dir (str): Where to save metrics.
    """
    # Load model
    model = DeepFace.load_model(model_path)  # Or Keras model if fine-tuned separately
    
    # Predict emotions
    predictions = []
    for img in test_images:
        result = DeepFace.analyze(
            img * 255.0,  # Denormalize for DeepFace
            actions=['emotion'],
            detector_backend='retinaface',
            enforce_detection=True,
            silent=True
        )
        predictions.append(result[0]['dominant_emotion'])
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions, labels=np.unique(test_labels))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    pd.DataFrame(cm, index=np.unique(test_labels), columns=np.unique(test_labels)).to_csv(
        os.path.join(output_dir, 'confusion_matrix.csv')
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix saved to", output_dir)