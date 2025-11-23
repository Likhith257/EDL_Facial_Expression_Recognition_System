# Facial Expression Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)

> A production-ready facial expression recognition system with **state-of-the-art optimizations** for accuracy and speed. Supports YOLOv8, EfficientNet-B3, and ArcFace architectures.

---

## ✨ Highlights

- 🎯 **Optimized for Production**: All models use AdamW, cosine annealing, label smoothing, and advanced augmentation
- 🚀 **Fast Inference**: FP16 mixed precision, optimized NMS, batch processing support
- 🎓 **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- 🌐 **Web Interface**: FastAPI backend with modern React + TypeScript frontend
- 🐳 **Docker Ready**: Single-command deployment
- 📊 **Real-time Detection**: 30-50ms per image on Apple M4 Pro

---

## 🚀 Features

### Core Capabilities
- ✅ **3 Model Architectures**: YOLOv8n (detection+classification), EfficientNet-B3 (CBAM attention), ArcFace (metric learning)
- ✅ **Production-Ready Training**: State-of-the-art optimizations (see [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md))
- ✅ **Complete Pipeline**: Dataset prep → Training → Evaluation → Inference
- ✅ **GPU Acceleration**: CUDA, Apple MPS (M-series), and CPU fallback
- ✅ **RESTful API**: FastAPI with automatic Swagger documentation
- ✅ **Real-time Web UI**: Upload images or use webcam for live detection

### Advanced Features
- 📈 **High Accuracy**: Optimized training yields +2-8% accuracy gains
- ⚡ **Fast Speed**: 30-40% faster inference with FP16 and optimized parameters
- 🔧 **Flexible Deployment**: Python CLI, Docker, or REST API
- 📊 **Comprehensive Docs**: Training guides, API reference, optimization techniques

### Documentation
- 📖 **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - Model optimization details
- 📖 **[RETRAIN_GUIDE.md](RETRAIN_GUIDE.md)** - Step-by-step training instructions
- 📖 **[ADVANCED_OPTIMIZATION.md](ADVANCED_OPTIMIZATION.md)** - Quantization, ONNX, TensorRT
- 📖 **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - REST API reference
- 📖 **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation and deployment
---

## 📁 Project Structure

```
EDL_Facial_Expression_Recognition_System/
├── main.py                     # Main entry point with CLI
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── README.md                   # This file
├── FEATURES.md                 # Detailed features list
├── API_DOCUMENTATION.md        # API reference
├── SETUP_GUIDE.md             # Setup instructions
├── LICENSE                     # MIT License
│
├── dataset/                    # Training dataset (YOLO format)
│   ├── data.yaml              # Dataset configuration
│   ├── images/                # Train/val/test images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/                # YOLO format labels
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                     # Model implementations
│   ├── yolo_model/            # YOLOv8 detection + classification
│   │   ├── src/
│   │   │   ├── prepare_dataset.py
│   │   │   ├── train_model.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   └── runs/              # Training outputs (gitignored)
│   │
│   ├── efficientnetb3_model/  # EfficientNet-B3 + CBAM
│   │   ├── src/
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   ├── config.yaml
│   │   └── checkpoints/       # Model weights (gitignored)
│   │
│   └── arcface_model/         # ArcFace + ResNet18
│       └── src/
│           ├── train.py
│           └── predict.py
│
└── frontend/                   # React + TypeScript web UI
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── app/                   # Source code
    └── dist/                  # Built files (gitignored)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System.git
cd EDL_Facial_Expression_Recognition_System

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Web Interface (Fastest)

```bash
# Start web server with pre-trained model
python main.py --serve

# Open http://localhost:8000 in browser
# Upload images or use webcam for real-time detection
```

### 3. Train Models (with Optimizations)

```bash
# Train YOLOv8 (~2-3 hours on M4 Pro)
cd models/yolo_model
python src/train_model.py

# Train EfficientNet-B3 (~4-5 hours)
cd models/efficientnetb3_model
python src/train.py

# Train ArcFace (~3-4 hours)
cd models/arcface_model
python src/train.py
```

**See [RETRAIN_GUIDE.md](RETRAIN_GUIDE.md) for detailed training instructions.**

### 4. Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

---

## 🎯 Model Architectures

### YOLOv8n (Recommended for Production)
- **Type**: Detection + Classification
- **Framework**: `--framework yolo`
- **Optimizations**: AdamW, cosine LR, mixup/copy-paste augmentation, FP16 inference
- **Speed**: 30-50ms per image (Apple M4 Pro)
- **Use Case**: Real-time applications, production deployment
- **Expected Accuracy**: 75-82% (after retraining with optimizations)

### EfficientNet-B3 (Highest Accuracy)
- **Type**: Classification with CBAM Attention
- **Framework**: `--framework efficientnetb3`
- **Optimizations**: AdamW, cosine annealing + warmup, heavy augmentation, label smoothing
- **Speed**: 20-40ms per image
- **Use Case**: High accuracy scenarios, batch processing
- **Expected Accuracy**: 78-85% (after retraining)

### ArcFace (Metric Learning)
- **Type**: Angular Margin Loss + ResNet-18
- **Framework**: `--framework arcface`
- **Optimizations**: AdamW, cosine LR, mixed precision, gradient clipping
- **Speed**: 15-30ms per image
- **Use Case**: Research, embedding-based approaches, facial recognition
- **Expected Accuracy**: 70-78% (after retraining)

---

## 📊 Performance Comparison

| Model | Inference Speed | Training Time | Accuracy | Memory | Status |
|-------|----------------|---------------|----------|--------|--------|
| **YOLOv8n** | ⚡ 30-50ms | ~2-3 hours | 75-82% | Low | ✅ Production |
| **EfficientNet-B3** | 🚀 20-40ms | ~4-5 hours | 78-85% | Medium | ✅ Production |
| **ArcFace** | 💨 15-30ms | ~3-4 hours | 70-78% | Low | ✅ Production |

*All speeds measured on Apple M4 Pro with MPS acceleration*

---

## 💻 API Usage

### Python Client Example

```python
import requests

# Upload image for prediction
url = "http://localhost:8000/api/predict"
files = {"file": open("image.jpg", "rb")}
params = {"detection_model": "yolo", "recognition_model": "yolo"}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Expression: {result['expression']}")
print(f"Confidence: {result['confidence']}")
print(f"Faces detected: {result['num_faces']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg" \
  -F "detection_model=yolo" \
  -F "recognition_model=yolo"
```

---

## 🔧 Configuration

### Training Configuration

```bash
# YOLO model sizes (trade-off between speed and accuracy)
python main.py --framework yolo --model yolov8n --train  # Fastest, smallest
python main.py --framework yolo --model yolov8s --train  # Small
python main.py --framework yolo --model yolov8m --train  # Medium (recommended)
python main.py --framework yolo --model yolov8l --train  # Large
python main.py --framework yolo --model yolov8x --train  # Extra large, most accurate

# Memory profiles for resource-constrained environments
python main.py --framework yolo --train --mem-profile low     # 2-4GB RAM
python main.py --framework yolo --train --mem-profile medium  # 4-8GB RAM
python main.py --framework yolo --train --mem-profile high    # 8GB+ RAM
```

### Server Configuration

```bash
# Custom port
python main.py --serve --port 3000

# Skip dependency checks (faster startup)
python main.py --serve --skip-deps
```

---

## 📚 Documentation

- **[API Documentation](API_DOCUMENTATION.md)** - Complete REST API reference
- **[Setup Guide](SETUP_GUIDE.md)** - Detailed installation and deployment
- **[Features](FEATURES.md)** - Comprehensive feature list
- **[Interactive API Docs](http://localhost:8000/docs)** - Swagger UI (when server running)

---

## 🛠️ Development

### Adding New Models

1. Create model directory in `models/`
2. Implement `train.py`, `evaluate.py`, `predict.py`
3. Add framework option in `main.py`
4. Update documentation

### Frontend Development

```bash
cd frontend
npm install
npm run dev     # Development server
npm run build   # Production build
```

---

## 🐛 Troubleshooting

### Virtual Environment Issues
```bash
# If venv is corrupted, recreate it
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU/MPS Not Detected
```bash
# Check PyTorch installation
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"

# Reinstall PyTorch if needed
pip uninstall torch torchvision
pip install torch torchvision
```

### Out of Memory Errors
```bash
# Reduce batch size in training scripts
# YOLO: models/yolo_model/src/train_model.py
# EfficientNet: models/efficientnetb3_model/src/train.py (batch_size=48 → 32)
# ArcFace: models/arcface_model/src/train.py (batch_size=64 → 32)
```

### Dataset Not Found
```bash
# Verify dataset structure
ls -R dataset/

# Should contain: data.yaml, images/train/, images/val/, labels/train/, labels/val/
```

---

## 📚 Additional Resources

### Key Documents
- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - Detailed model optimization techniques and results
- **[RETRAIN_GUIDE.md](RETRAIN_GUIDE.md)** - Complete retraining instructions with hyperparameters
- **[ADVANCED_OPTIMIZATION.md](ADVANCED_OPTIMIZATION.md)** - Production techniques (quantization, ONNX, TensorRT)
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - REST API endpoints and examples
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Deployment and configuration guide
- **[FEATURES.md](FEATURES.md)** - Complete feature list

### Training Tips
1. **Start with YOLOv8n** - Fastest to train, good baseline accuracy
2. **Use GPU acceleration** - 5-10x faster training (CUDA or Apple MPS)
3. **Monitor training** - Watch for overfitting, use early stopping
4. **Try ensemble** - Combine predictions from multiple models for best accuracy

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **EfficientNet** by Google Research
- **ArcFace** by Deng et al.
- Dataset contributors from EDL Lab

---

## 📧 Contact

- **Repository**: [EDL_Facial_Expression_Recognition_System](https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System)
- **Issues**: [GitHub Issues](https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for EDL Lab

</div>

