# Facial Expression Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)

> A comprehensive multi-model facial expression recognition system with deep learning models including YOLOv8, EfficientNet-B3, and more. Features a modern web interface for real-time emotion detection.

---

## Features

- **Multi-Model Support**: YOLOv8 and EfficientNet-B3 for facial expression recognition
- **Web Interface**: Modern React + TypeScript frontend with real-time detection
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- **Multiple Input Methods**: Upload images or use webcam for live detection
- **RESTful API**: FastAPI backend with automatic documentation
- **High Accuracy**: 72.9% mAP50 with YOLOv8, 72.1% validation accuracy with EfficientNet-B3
- **GPU Support**: Automatic CUDA/MPS/CPU detection
- **Real-World Image Support**: Enhanced preprocessing for photos outside the training dataset
- **Batch Processing**: Analyze multiple images simultaneously
- **Confidence Control**: Adjustable threshold slider for detection sensitivity

---

## Model Status

| Model | Status | Accuracy | Features |
|-------|--------|----------|----------|
| YOLOv8 | ✅ Complete | 72.9% mAP50 | Detection + Classification |
| EfficientNet-B3 | ✅ Complete | 72.1% val acc | Classification with CBAM |
| EfficientNetV2 | 🔄 In Progress | - | Training framework ready |

---

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- CUDA-compatible GPU (optional, for faster training)

### 1. Clone the Repository

```bash
git clone https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System.git
cd EDL_Facial_Expression_Recognition_System
```

### 2. Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Frontend (Optional)

```bash
cd frontend
npm install
npm run build
cd ..
```

---

## Quick Start

### Web Interface (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the web server
python main.py --serve --skip-deps
```

Then open http://localhost:8000 in your browser.

### Command Line Usage

```bash
# Train a model
python main.py --framework yolo --train

# Evaluate model
python main.py --framework yolo --evaluate

# Run inference
python main.py --framework yolo --predict

# Full pipeline
python main.py --framework yolo --all
```

### Available Frameworks

- `yolo` - YOLOv8 (default)
- `efficientnetb3` - EfficientNet-B3
- `efficientnetv2` - EfficientNetV2

---

## Project Structure

```
EDL_Facial_Expression_Recognition_System/
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── dataset/                 # Training dataset (YOLO format)
├── models/                  # Model implementations
│   ├── yolo_model/         # YOLOv8 implementation
│   ├── efficientnetb3_model/  # EfficientNet-B3
│   └── ...                 # Other models
├── frontend/               # React + TypeScript web UI
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
└── runs/                   # Training outputs
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure.

---

## Web Interface

### Features

- **Image Upload**: Drag and drop or click to upload
- **Webcam Support**: Real-time capture from camera
- **Model Selection**: Switch between YOLOv8 and EfficientNet-B3
- **Live Results**: Instant emotion detection with confidence scores
- **Multiple Faces**: Detect and analyze multiple faces simultaneously

### API Endpoints

- `GET /` - Web interface
- `POST /api/predict` - Emotion prediction
- `GET /api/health` - Health check
- `GET /docs` - API documentation (Swagger)

---

## Dataset

The system expects YOLO format dataset:

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**Emotion Classes**: angry, disgust, fear, happy, neutral, sad, surprised

---

## Configuration

### Model Selection

```bash
# YOLOv8 variants
python main.py --model yolov8n --train  # Nano (fastest)
python main.py --model yolov8s --train  # Small
python main.py --model yolov8m --train  # Medium
python main.py --model yolov8l --train  # Large
python main.py --model yolov8x --train  # Extra Large

# Memory profiles for training
python main.py --mem-profile low --train    # For 8GB RAM
python main.py --mem-profile medium --train  # For 16GB RAM
python main.py --mem-profile high --train    # For 32GB+ RAM
```

### GPU Configuration

The system automatically detects available hardware:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon (M1/M2/M3)
- CPU fallback

---

## Training

```bash
# Train YOLOv8
python main.py --framework yolo --train

# Train EfficientNet-B3
python main.py --framework efficientnetb3 --train

# Custom configuration
python main.py --framework yolo --model yolov8m --mem-profile high --train
```

Training outputs are saved to `models/{framework}/runs/` and `runs/detect/`.

---

## Evaluation

```bash
# Evaluate trained model
python main.py --framework yolo --evaluate

# Results include:
# - Confusion matrix
# - Precision, Recall, F1 scores
# - Per-class accuracy
# - Visualization plots
```

---

## Contributors

- [Likhith](https://github.com/Likhith257)
- Thanish Chinnappa KC
- Saumitra Purkayastha
- Sundareshwar S
- Tenzin Kunga

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2025
Likhith, Thanish Chinnappa KC, Saumitra Purkayastha, 
Sundareshwar S, Tenzin Kunga
```

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [React](https://reactjs.org/) - Frontend library

---

## Support

For issues or questions, please contact the contributors via their GitHub profiles.

---

**If you find this project useful, please consider giving it a star!**
