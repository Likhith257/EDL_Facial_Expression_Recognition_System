
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .model import create_model
from .train import FacialExpressionDataset, get_transforms


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main(checkpoint_path=None):
    """Main evaluation"""
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / 'dataset'
    CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
    CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints'
    
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / 'best_model.pth'
    
    # Load config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Parameters
    num_classes = config.get('num_classes', 7)
    batch_size = config.get('batch_size', 32)
    img_size = config.get('img_size', 224)
    device_type = config.get('device', 'mps')
    class_names = config.get('class_names', ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'])
    
    # Device
    if device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\n📁 Loading test dataset from: {DATASET_PATH}")
    test_dataset = FacialExpressionDataset(
        DATASET_PATH, split='test',
        transform=get_transforms('test', img_size)
    )
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4
    )
    
    # Load model
    print(f"\n🏗️  Loading model from: {checkpoint_path}")
    model = create_model(num_classes=num_classes, pretrained=False, device=device_type)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return
    
    # Evaluate
    print("\n🎯 Evaluating model...")
    accuracy, cm, preds, labels = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Save confusion matrix
    plot_confusion_matrix(cm, class_names, CHECKPOINT_DIR / 'confusion_matrix.png')
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
