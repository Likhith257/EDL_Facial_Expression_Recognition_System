"""Minimal evaluation and visualization script."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class ModelEvaluator:
    def __init__(self, model_path, data_yaml):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained YOLO weights
            data_yaml: Path to data configuration YAML
        """
        # Select best available device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Neutral', 'Sad', 'Surprised'
        ]
        
        # Output directory
        self.output_dir = Path('runs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"🖥️  Evaluation device: {self.device}")
    
    def evaluate(self, split='test'):
        """
        Evaluate model on test/validation set.
        
        Args:
            split: Dataset split to evaluate ('test' or 'val')
        
        Returns:
            Evaluation metrics
        """
        print(f"\n🔍 Evaluating model on {split} set...")
        
        # Run validation
        metrics = self.model.val(
            data=self.data_yaml,
            split=split,
            device=self.device
        )
        
        # Print metrics
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print("-" * 60)
        for i, emotion in enumerate(self.emotion_labels):
            if i < len(metrics.box.ap50):
                print(f"{emotion:12s}: AP50={metrics.box.ap50[i]:.4f}, "
                      f"AP={metrics.box.ap[i]:.4f}")
        
        return metrics
    
    def create_confusion_matrix(self, test_images_dir, output_path=None):
        """
        Create confusion matrix for classification results.
        
        Args:
            test_images_dir: Directory containing test images
            output_path: Path to save confusion matrix plot
        """
        print("\n📊 Creating confusion matrix...")
        
        # Get predictions
        y_true = []
        y_pred = []
        
        # Process test images
        test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                     list(Path(test_images_dir).glob('*.png'))
        
        for img_path in test_images:
            # Get ground truth from filename (assuming format: emotion_id.jpg)
            gt_emotion = self._extract_emotion_from_filename(img_path.name)
            if gt_emotion is None:
                continue
            
            # Get prediction
            results = self.model.predict(str(img_path), device=self.device, verbose=False)[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                pred_emotion = int(results.boxes[0].cls[0])
                y_true.append(gt_emotion)
                y_pred.append(pred_emotion)
        
        if not y_true:
            print("❌ No predictions to evaluate")
            return
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            cbar_kws={'label': 'Normalized Count'}
        )
        plt.title('Confusion Matrix - Facial Expression Recognition', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('True Emotion', fontsize=12)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to: {output_path}")
        
        plt.close()
        
        # Print classification report
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true, 
            y_pred, 
            target_names=self.emotion_labels,
            digits=4
        ))
        
        return cm, y_true, y_pred
    
    def plot_training_curves(self, results_dir, output_path=None):
        """
        Plot training curves from results CSV.
        
        Args:
            results_dir: Directory containing training results
            output_path: Path to save plot
        """
        print("\n📈 Plotting training curves...")
        
        # Load results
        results_csv = Path(results_dir) / 'results.csv'
        if not results_csv.exists():
            print(f"❌ Results file not found: {results_csv}")
            return
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Loss plots
        metrics = [
            ('box_loss', 'Box Loss'),
            ('cls_loss', 'Classification Loss'),
            ('dfl_loss', 'DFL Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('mAP50', 'mAP@0.5')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Check for train and val columns
            train_col = f'train/{metric}'
            val_col = f'val/{metric}' if f'val/{metric}' in df.columns else metric
            
            if train_col in df.columns:
                ax.plot(df['epoch'], df[train_col], label='Train', linewidth=2)
            if val_col in df.columns:
                ax.plot(df['epoch'], df[val_col], label='Validation', 
                       linewidth=2, linestyle='--')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / 'training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training curves saved to: {output_path}")
        
        plt.close()
    
    def visualize_predictions(self, images_dir, num_samples=16, output_path=None):
        """
        Visualize model predictions on sample images.
        
        Args:
            images_dir: Directory containing images
            num_samples: Number of samples to visualize
            output_path: Path to save visualization
        """
        print(f"\n🖼️  Visualizing {num_samples} predictions...")
        
        # Get random sample of images
        images = list(Path(images_dir).glob('*.jpg')) + \
                list(Path(images_dir).glob('*.png'))
        
        if len(images) < num_samples:
            num_samples = len(images)
        
        sample_images = np.random.choice(images, num_samples, replace=False)
        
        # Create grid
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, img_path in enumerate(sample_images):
            # Read image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            results = self.model.predict(str(img_path), device=self.device, verbose=False)[0]
            
            # Annotate
            if results.boxes is not None and len(results.boxes) > 0:
                box = results.boxes[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                emotion = self.emotion_labels[cls]
                
                # Draw on image
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                title = f"{emotion} ({conf:.2f})"
            else:
                title = "No detection"
            
            # Display
            axes[idx].imshow(image_rgb)
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
        
        # Hide empty subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / 'sample_predictions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Sample predictions saved to: {output_path}")
        
        plt.close()
    
    def analyze_emotion_distribution(self, results_dir, output_path=None):
        """
        Analyze and visualize emotion distribution in dataset.
        
        Args:
            results_dir: Directory containing dataset
            output_path: Path to save visualization
        """
        print("\n📊 Analyzing emotion distribution...")
        
        # Count emotions
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        
        # Scan dataset
        for split in ['train', 'val', 'test']:
            label_dir = Path(results_dir) / 'labels' / split
            if not label_dir.exists():
                continue
            
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            emotion_counts[self.emotion_labels[cls]] += 1
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        
        bars = plt.bar(emotions, counts, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Emotion Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / 'emotion_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Emotion distribution saved to: {output_path}")
        
        plt.close()
        
        # Print statistics
        print("\nEmotion Distribution:")
        print("-" * 40)
        total = sum(counts)
        for emotion, count in emotion_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{emotion:12s}: {count:5d} ({percentage:5.2f}%)")
    
    def _extract_emotion_from_filename(self, filename):
        """Extract emotion class from filename."""
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            if emotion in filename.lower():
                return self.emotion_labels.index(emotion.capitalize())
        return None


def main():
    """Main evaluation execution."""
    print("=" * 60)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 60)
    
    # Configuration
    model_path = 'runs/train/facial_expression_yolov8n/weights/best.pt'
    data_yaml = 'dataset/data.yaml'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, data_yaml)
    
    # Evaluate model
    evaluator.evaluate(split='test')
    
    # Plot training curves
    results_dir = 'runs/train/facial_expression_yolov8n'
    if Path(results_dir).exists():
        evaluator.plot_training_curves(results_dir)
    
    # Visualize predictions
    test_images_dir = 'dataset/images/test'
    if Path(test_images_dir).exists():
        evaluator.visualize_predictions(test_images_dir, num_samples=16)
    
    # Analyze emotion distribution
    evaluator.analyze_emotion_distribution('dataset')
    
    print(f"\n✅ Evaluation complete! Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
