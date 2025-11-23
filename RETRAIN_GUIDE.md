# Quick Retrain Guide

## Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate

# Verify Python and dependencies
python --version  # Should be 3.12.x
pip list | grep -E "torch|ultralytics|opencv"
```

---

## Option 1: Train All Models (Recommended)

```bash
# Train YOLO (fastest, ~2-3 hours on M4 Pro)
cd models/yolo_model
python src/train_model.py
cd ../..

# Train EfficientNet-B3 (~4-5 hours on M4 Pro)
cd models/efficientnetb3_model
python src/train.py
cd ../..

# Train ArcFace (~3-4 hours on M4 Pro)
cd models/arcface_model
python src/train.py
cd ../..
```

---

## Option 2: Train Individual Models

### YOLO (Detection + Classification)
```bash
cd models/yolo_model
python src/train_model.py

# Training will:
# - Use dataset/data.yaml configuration
# - Train for 100 epochs with early stopping (patience=20)
# - Save best model to: runs/facial_expression_yolov8n/weights/best.pt
# - Use AdamW optimizer with cosine LR scheduling
# - Apply advanced augmentation (mixup, copy-paste, shear, perspective)
```

**Expected Training Time**: 2-3 hours on Apple M4 Pro

**Key Hyperparameters**:
- Batch size: Auto-optimized based on GPU memory
- Learning rate: 0.002 → 0.0002 (cosine decay)
- Image size: 640x640
- Confidence threshold: 0.3 (validation)
- IoU threshold: 0.5 (validation)

---

### EfficientNet-B3 (Classification with CBAM)
```bash
cd models/efficientnetb3_model
python src/train.py

# Training will:
# - Use dataset/ structure (train/val/test splits)
# - Train for 120 epochs with early stopping (patience=25)
# - Save best model to: checkpoints/best_model.pth
# - Use AdamW optimizer with cosine LR + warmup
# - Apply heavy augmentation (shear, perspective, color jitter)
```

**Expected Training Time**: 4-5 hours on Apple M4 Pro

**Key Hyperparameters**:
- Batch size: 48
- Learning rate: 0.00015 → 0.0000015 (cosine decay)
- Image size: 224x224
- Warmup epochs: 5
- Label smoothing: 0.15

---

### ArcFace (Metric Learning)
```bash
cd models/arcface_model
python src/train.py

# Training will:
# - Use dataset/images/train and dataset/labels/train
# - Train for 50 epochs (no early stopping)
# - Save best model to: arcface_model.pt
# - Use AdamW optimizer with cosine LR + warmup
# - Apply augmentation (flip, rotation, color jitter)
```

**Expected Training Time**: 3-4 hours on Apple M4 Pro

**Key Hyperparameters**:
- Batch size: 64
- Learning rate: 0.0005 → 0.000005 (cosine decay)
- Image size: 112x112
- Warmup epochs: 3
- ArcFace margin: 0.5, scale: 30.0

---

## Monitoring Training

### Real-time Progress
All models print per-epoch metrics:
- Training loss and accuracy
- Validation loss and accuracy
- Current learning rate
- Best model checkpoints

### Check Training Status
```bash
# YOLO - TensorBoard
cd models/yolo_model
tensorboard --logdir runs/facial_expression_yolov8n

# EfficientNet-B3 - Training history
cat models/efficientnetb3_model/checkpoints/history.json

# ArcFace - Console output only
```

---

## After Training

### Test Trained Models
```bash
# Start web server
python main.py serve

# Open browser: http://localhost:8000
# Upload test images to verify predictions
```

### Evaluate Models
```bash
# YOLO evaluation
python main.py evaluate --framework yolo

# EfficientNet-B3 evaluation
python main.py evaluate --framework efficientnetb3

# ArcFace evaluation
python main.py evaluate --framework arcface
```

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in training scripts:
# YOLO: Edit models/yolo_model/src/train_model.py (line with 'batch:')
# EfficientNet: Edit models/efficientnetb3_model/src/train.py (line with batch_size=48)
# ArcFace: Edit models/arcface_model/src/train.py (line with batch_size=64)
```

### Slow Training
```bash
# Check GPU usage:
# For M4 Pro MPS:
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# For CUDA:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Dataset Not Found
```bash
# Verify dataset structure:
ls -R dataset/

# Should have:
# dataset/
#   data.yaml
#   images/train/
#   images/val/
#   images/test/
#   labels/train/
#   labels/val/
#   labels/test/
```

---

## Training Tips

### For Best Accuracy
1. **Train longer**: Let models train to completion (don't interrupt)
2. **Use full dataset**: Ensure all images are properly labeled
3. **Monitor validation**: Early stopping will prevent overfitting
4. **Ensemble**: Train all three models and combine predictions

### For Faster Training
1. **Smaller batch sizes**: Reduce GPU memory but slower per epoch
2. **Fewer epochs**: Set lower epoch counts (but may reduce accuracy)
3. **Use GPU**: Ensure MPS (M4 Pro) or CUDA is detected
4. **Close other apps**: Free up GPU memory

### For Production
1. **Save checkpoints**: Best models are automatically saved
2. **Version models**: Keep track of training dates and metrics
3. **Test thoroughly**: Validate on diverse test images
4. **Document performance**: Record accuracy, speed, dataset size

---

## Expected Results After Training

| Model | Validation Accuracy | Inference Speed |
|-------|---------------------|-----------------|
| **YOLO** | 75-82% | ~30-50ms per image |
| **EfficientNet-B3** | 78-85% | ~20-40ms per image |
| **ArcFace** | 70-78% | ~15-30ms per image |

*Speeds measured on Apple M4 Pro GPU (MPS)*

---

## Next Steps After Training

1. **Test predictions**: `python main.py serve` → upload images
2. **Evaluate metrics**: `python main.py evaluate --framework [model]`
3. **Deploy**: Docker containerize or deploy directly
4. **Monitor**: Track real-world performance and retrain as needed

Happy training! 🚀
