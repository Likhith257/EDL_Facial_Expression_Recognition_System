import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from datetime import datetime

_THIS_FILE = Path(__file__).resolve()
YOLO_ROOT = _THIS_FILE.parents[1]
PROJECT_ROOT = _THIS_FILE.parents[3]


class FacialExpressionTrainer:
    def __init__(self, data_yaml=str((PROJECT_ROOT / 'dataset' / 'data.yaml').resolve()), model_size='yolov8n'):
        """
        Initialize the YOLO trainer for facial expression recognition.
        
        Args:
            data_yaml: Path to data.yaml configuration file
            model_size: YOLO model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.model = None
        self.results = None
        
        # Create output directories
        self.output_dir = (YOLO_ROOT / 'runs' / 'train')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for GPU (CUDA for NVIDIA, MPS for Apple Silicon)
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"🖥️  Using device: {self.device}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"🖥️  Using device: {self.device} (Apple Silicon GPU)")
        else:
            self.device = 'cpu'
            print(f"🖥️  Using device: {self.device}")
    
    def load_model(self, pretrained=True):
        """
        Load YOLO model.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        print(f"\n📦 Loading {self.model_size} model...")
        
        if pretrained:
            # Load pretrained model (prefer local weights within YOLO folder)
            local_weights = YOLO_ROOT / f"{self.model_size}.pt"
            model_path = str(local_weights) if local_weights.exists() else self.model_size
            self.model = YOLO(model_path)
            print(f"✅ Loaded pretrained {self.model_size} model")
        else:
            # Load model architecture only
            arch_path = YOLO_ROOT / f"{self.model_size}.yaml"
            model_path = str(arch_path) if arch_path.exists() else f"{self.model_size}.yaml"
            self.model = YOLO(model_path)
            print(f"✅ Loaded {self.model_size} architecture")
    
    def train(self, epochs=50, batch_size=16, img_size=640, memory_profile='default', **kwargs):
        """
        Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            **kwargs: Additional training parameters
        """
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return
        
        print("\n🚀 Starting training...")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Batch size (requested): {batch_size}")
        print(f"Image size (requested): {img_size}")
        print(f"Memory profile: {memory_profile}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Training parameters
        train_params = {
            'data': self.data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'project': str(self.output_dir.parent),
            'name': f'facial_expression_{self.model_size}',
            'exist_ok': True,
            'patience': 20,  # Early stopping patience
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'verbose': True,
            'plots': True,
            
            # Optimization - Improved for accuracy
            'optimizer': 'AdamW',  # Better generalization
            'lr0': 0.002,  # Slightly higher initial LR
            'lrf': 0.001,  # Lower final LR for fine-tuning
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,  # Gradual warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,  # Cosine LR scheduler
            'label_smoothing': 0.1,  # Better generalization
            
            # Advanced Augmentation for facial expressions
            'hsv_h': 0.02,  # More color variation
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 5.0,  # Slight rotation for faces
            'translate': 0.15,  # More translation
            'scale': 0.6,  # Better scale variation
            'shear': 2.0,  # Add shear augmentation
            'perspective': 0.0003,  # Perspective transform
            'flipud': 0.0,  # No vertical flip for faces
            'fliplr': 0.5,  # Horizontal flip OK
            'mosaic': 1.0,  # Keep mosaic for diversity
            'mixup': 0.1,  # Add mixup for robustness
            'copy_paste': 0.1,  # Copy-paste augmentation
            
            # Validation - More strict
            'val': True,
            'conf': 0.3,  # Higher confidence threshold
            'iou': 0.5,  # Higher IoU for better NMS
        }

        # Adjust for memory profile
        if memory_profile == 'low':
            train_params['batch'] = min(batch_size, 8)
            train_params['imgsz'] = min(img_size, 512)
            train_params['workers'] = 2
            train_params['mosaic'] = 0.0
            train_params['mixup'] = 0.0
            train_params['cache'] = False
            train_params['amp'] = True  # mixed precision
        elif memory_profile == 'medium':
            train_params['batch'] = min(batch_size, 12)
            train_params['imgsz'] = min(img_size, 576)
            train_params['workers'] = 4
            train_params['mosaic'] = 0.2
            train_params['mixup'] = 0.0
            train_params['cache'] = False
            train_params['amp'] = True
        elif memory_profile == 'high':
            # Allow larger settings; user must have sufficient RAM/GPU
            train_params['workers'] = 8
            train_params['cache'] = 'ram'
            train_params['amp'] = True

        
        # Override with custom parameters
        train_params.update(kwargs)
        
        # Train model
        self.results = self.model.train(**train_params)
        
        print("\n✅ Training complete!")
        self.print_results()
    
    def print_results(self):
        """Print training results summary."""
        if self.results is None:
            print("❌ No training results available.")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 60)
        
        # Get best results
        results_dir = Path(self.results.save_dir)
        print(f"\n📁 Results saved to: {results_dir}")
        print(f"📊 Best weights: {results_dir / 'weights' / 'best.pt'}")
        print(f"📊 Last weights: {results_dir / 'weights' / 'last.pt'}")
    
    def validate(self, weights_path=None):
        """
        Validate the trained model.
        
        Args:
            weights_path: Path to weights file (uses best.pt if None)
        """
        if weights_path is None and self.results is not None:
            weights_path = Path(self.results.save_dir) / 'weights' / 'best.pt'
        
        if weights_path is None:
            print("❌ No weights path provided.")
            return
        
        print(f"\n🔍 Validating model: {weights_path}")
        
        # Load model with trained weights
        model = YOLO(weights_path)
        
        # Validate
        metrics = model.val(data=self.data_yaml)
        
        print("\n" + "=" * 60)
        print("VALIDATION METRICS")
        print("=" * 60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, weights_path=None, format='onnx'):
        """
        Export trained model to different formats.
        
        Args:
            weights_path: Path to weights file
            format: Export format (onnx, torchscript, coreml, etc.)
        """
        if weights_path is None and self.results is not None:
            weights_path = Path(self.results.save_dir) / 'weights' / 'best.pt'
        
        if weights_path is None:
            print("❌ No weights path provided.")
            return
        
        print(f"\n📤 Exporting model to {format} format...")
        
        model = YOLO(weights_path)
        export_path = model.export(format=format)
        
        print(f"✅ Model exported to: {export_path}")
        return export_path
    
    def resume_training(self, checkpoint_path):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (.pt file)
        """
        print(f"\n🔄 Resuming training from: {checkpoint_path}")
        
        self.model = YOLO(checkpoint_path)
        print("✅ Checkpoint loaded. Call train() to continue training.")


def main(model_size='yolov8n', memory_profile='default'):
    """Main training function.
    
    Args:
        model_size: YOLO model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    """
    print("\n🚀 Starting Facial Expression Recognition Training Pipeline")
    print("=" * 60)
    
    # Check if dataset exists (use absolute path)
    data_yaml = PROJECT_ROOT / 'dataset' / 'data.yaml'
    if not data_yaml.exists():
        print(f"\n❌ Dataset not found at {data_yaml}")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Initialize trainer
    trainer = FacialExpressionTrainer(
        data_yaml=str(data_yaml),
        model_size=model_size
    )
    
    # Load pretrained model
    trainer.load_model(pretrained=True)
    
    # Train model
    trainer.train(
        epochs=50,
        batch_size=16,
        img_size=640,
        memory_profile=memory_profile,
    )
    
    # Validate model
    trainer.validate()
    
    # Export model (optional)
    # trainer.export_model(format='onnx')
    
    print("\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
