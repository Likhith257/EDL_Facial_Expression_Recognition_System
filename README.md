# Facial Expression Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)

> A production-ready facial expression recognition system supporting multiple deep learning architectures including YOLOv8, EfficientNet-B3, EfficientNetV2, and ArcFace.

---

## 🚀 Features

- ✅ **Multiple Model Architectures**: YOLOv8, EfficientNet-B3, EfficientNetV2, ArcFace
- ✅ **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- ✅ **Complete Pipeline**: Dataset preparation, training, evaluation, and inference
- ✅ **Web Interface**: FastAPI backend with modern frontend
- ✅ **Docker Ready**: Full containerization support
- ✅ **Real-time Detection**: Fast inference with optimized models
👉 **See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for API reference**
👉 **See [SETUP_GUIDE.md](SETUP_GUIDE.md) for deployment guide**

---

## Features

- **Multi-Model Support**: YOLOv8, EfficientNet-B3, and ArcFace for facial expression recognition
- **Web Interface**: Modern React + TypeScript frontend with real-time detection
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- **Multiple Input Methods**: Upload images or use webcam for live detection
- **RESTful API**: FastAPI backend with automatic documentation
- **High Accuracy**: 72.9% mAP50 with YOLOv8, 72.1% validation accuracy with EfficientNet-B3
- **GPU Support**: Automatic CUDA/MPS/CPU detection
- **Real-World Image Support**: Enhanced preprocessing for photos outside the training dataset
- **Batch Processing**: Analyze multiple images simultaneously
- **Confidence Control**: Adjustable threshold slider for detection sensitivity
- **Analytics Dashboard**: Real-time statistics and visualization
- **Model Comparison**: Compare multiple models on same image
- **Video Processing**: Analyze emotions in video files
- **Export & Reporting**: Generate PDF/CSV reports

---

## Model Status

| Model | Status | Accuracy | Features |
|-------|--------|----------|----------|
| YOLOv8 | ✅ Complete | 72.9% mAP50 | Detection + Classification |
| EfficientNet-B3 | ✅ Complete | 72.1% val acc | Classification with CBAM |
| ArcFace (ResNet-18) | ✅ Complete | - | Angular Margin Loss |
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
│   ├── arcface_model/         # ArcFace + ResNet18
│   │   └── src/
│   │       ├── train.py
│   │       └── predict.py
│   │
│   └── efficientnetv2_model/  # EfficientNetV2-S
│       └── src/
│           └── train.py
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

### 1. Web Interface (Recommended)

```bash
# Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start web server
python main.py --serve

# Open http://localhost:8000 in your browser
```

### 2. Command Line Training

```bash
# Train YOLOv8 model
python main.py --framework yolo --train

# Evaluate trained model
python main.py --framework yolo --evaluate

# Run inference
python main.py --framework yolo --predict

# Complete pipeline (prepare + train + evaluate)
python main.py --framework yolo --all
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

---

## 🎯 Available Models

### YOLOv8 (Recommended)
- **Framework**: `--framework yolo`
- **Models**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **Features**: Combined detection + classification, fastest inference
- **Use case**: Real-time applications, production deployment

### EfficientNet-B3
- **Framework**: `--framework efficientnetb3`
- **Features**: CBAM attention, strong accuracy
- **Use case**: High accuracy scenarios, batch processing

### ArcFace
- **Framework**: `--framework arcface`
- **Features**: Angular margin loss, metric learning
- **Use case**: Research, embedding-based approaches

### EfficientNetV2
- **Framework**: `--framework efficientnetv2`
- **Features**: Modern architecture, fast training
- **Use case**: Balanced speed and accuracy

---

## 📊 Model Performance

| Model | Accuracy | Speed | Memory | Status |
|-------|----------|-------|--------|--------|
| YOLOv8n | High | Fast | Low | ✅ Production Ready |
| EfficientNet-B3 | Very High | Medium | Medium | ✅ Production Ready |
| ArcFace | High | Medium | Medium | ✅ Production Ready |
| EfficientNetV2 | High | Fast | Medium | 🔄 In Development |

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

### Common Issues

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**CUDA Out of Memory**
```bash

