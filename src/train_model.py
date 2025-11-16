from deepface import DeepFace
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

def fine_tune_model(images, labels, output_dir='runs/train', epochs=5):
    """
    Fine-tune DeepFace SFace model for FER.
    Args:
        images (np.array): Preprocessed images.
        labels (np.array): Emotion labels.
        output_dir (str): Where to save model weights.
        epochs (int): Training epochs.
    """
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Build DeepFace model
    model = DeepFace.build_model("Emotion")  # Emotion recognition model
    
    # Prepare data for fine-tuning (DeepFace expects specific format)
    train_data = [{"image": img, "label": lbl} for img, lbl in zip(X_train, y_train)]
    val_data = [{"image": img, "label": lbl} for img, lbl in zip(X_val, y_val)]
    
    # Fine-tune
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=16,  # Small for CPU
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
    )
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.h5'))