# Setup Guide

## Facial Expression Recognition System - Installation & Deployment

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Setup](#development-setup)
3. [Training Models](#training-models)
4. [Docker Deployment](#docker-deployment)
5. [Frontend Setup](#frontend-setup)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)
- CUDA GPU (optional, for faster training)

### Installation

```bash
# Clone repository
git clone https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System.git
cd EDL_Facial_Expression_Recognition_System

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start Web Server

```bash
# Start server with web interface
python main.py --serve

# Or skip dependency check for faster startup
python main.py --serve --skip-deps
```

Open browser to: **http://localhost:8000**

---

## Development Setup

### Core Dependencies

The `requirements.txt` includes all necessary packages:

```txt
# Core ML
ultralytics, torch, torchvision
opencv-python, numpy, pandas
matplotlib, seaborn, pillow
scikit-learn, pyyaml, tqdm

# Training enhancements (optional)
albumentations, tensorboard

# Web server (optional)
fastapi, uvicorn, python-multipart
```

### Install for Development

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal set for inference only
pip install ultralytics torch opencv-python numpy pillow
#   ├── images/test/
#   ├── labels/train/
#   ├── labels/val/
#   └── labels/test/
```

#### 5. Train Models

```bash
# Train YOLOv8n (nano - fastest)
python main.py --framework yolo --train --model yolov8n

# Train YOLOv8s (small - balanced)
python main.py --framework yolo --train --model yolov8s

# Train EfficientNet-B3
```

---

## Training Models

### Dataset Preparation

```bash
# Prepare YOLO format dataset
python main.py --framework yolo --prepare
```

This organizes images into:
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

### Training Commands

```bash
# Train YOLOv8 (fastest)
python main.py --framework yolo --train

# Train with specific model size
python main.py --framework yolo --model yolov8n --train  # Nano (fastest)
python main.py --framework yolo --model yolov8s --train  # Small
python main.py --framework yolo --model yolov8m --train  # Medium
python main.py --framework yolo --model yolov8l --train  # Large
python main.py --framework yolo --model yolov8x --train  # Extra Large

# Train EfficientNet-B3
python main.py --framework efficientnetb3 --train

# Train ArcFace
python main.py --framework arcface --train

# Memory-constrained training
python main.py --framework yolo --train --mem-profile low     # 4GB RAM
python main.py --framework yolo --train --mem-profile medium  # 8GB RAM
python main.py --framework yolo --train --mem-profile high    # 16GB+ RAM
```

### Evaluation

```bash
# Evaluate trained model
python main.py --framework yolo --evaluate

# Results include:
# - Confusion matrix
# - Per-class metrics
# - mAP scores
# - Visualization plots
```

### Inference

```bash
# Run predictions via CLI
python main.py --framework yolo --predict

# Or use web interface
python main.py --serve
```

---

## Docker Deployment

### Simple Docker

```bash
# Build image
docker build -t facial-recognition .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/dataset:/app/dataset \
  facial-recognition
```

### Docker Compose

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

### GPU Support (NVIDIA)

Uncomment GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires:
- NVIDIA drivers
- nvidia-docker2 installed

---

## Frontend Setup

### Development

```bash
cd frontend

# Install dependencies
npm install
# or
pnpm install

# Start dev server
npm run dev

# Runs on http://localhost:5173
# API should be on http://localhost:8000
```

### Production Build

```bash
# Build optimized bundle
npm run build

# Output: frontend/dist/

# Serve with main application
python main.py --serve
```

---

## Configuration

### Server Port

```bash
# Custom port
python main.py --serve --port 3000
```

### Model Selection

```bash
# Select framework and model
python main.py --framework yolo --model yolov8m --train
```

### Hardware Selection

Automatic detection:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU fallback

Force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python main.py --train
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 2. CUDA Out of Memory

```bash
# Use smaller model
python main.py --framework yolo --model yolov8n --train

# Or lower memory profile
python main.py --train --mem-profile low
```

#### 3. Model Not Found

```bash
# Train model first
python main.py --framework yolo --train

# Check weights exist
ls models/yolo_model/runs/*/weights/best.pt
```

#### 4. Port Already in Use

```bash
# Use different port
python main.py --serve --port 8080
```

#### 5. Frontend Not Loading

```bash
# Build frontend
cd frontend
npm install
npm run build
cd ..

# Start server
python main.py --serve
```

### Performance Issues

#### Slow Training

- Use smaller model (yolov8n)
- Reduce batch size
- Use GPU if available
- Lower image resolution

#### Slow Inference

- Use yolov8n model
- Use YOLO framework (fastest)
- Ensure GPU is being used
- Reduce image size before prediction

### Getting Help

1. Check [README.md](README.md) for overview
2. Review [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for API details
3. See [FEATURES.md](FEATURES.md) for capability list
4. Open GitHub issue for bugs

---

## Production Deployment

### Recommendations

1. **Use Docker** for consistent environment
2. **Use GPU** for better performance
3. **Monitor resources** (CPU, RAM, GPU)
4. **Set up logging** for debugging
5. **Use reverse proxy** (nginx) for production
6. **Enable HTTPS** for security

### Example nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Environment Variables

```bash
# Optional configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
```

---

## Next Steps

After setup:

1. **Prepare dataset** - Organize your images
2. **Train model** - Start with YOLOv8n for speed
3. **Evaluate** - Check model performance
4. **Deploy** - Use web interface or API
5. **Iterate** - Improve with more data/training

---

## Support

For issues or questions:
- Check documentation files
- Review error messages carefully
- Open GitHub issue with details
- Include system info (OS, Python version, GPU)

---

**Happy Training! 🚀**

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Or use Docker (included in docker-compose.yml)
```

#### 2. Create Database

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database
CREATE DATABASE facial_recognition;
CREATE USER fer_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE facial_recognition TO fer_user;
\q
```

#### 3. Set Environment Variable

```bash
export DATABASE_URL="postgresql://fer_user:secure_password@localhost:5432/facial_recognition"
```

Or create `.env` file:

```env
DATABASE_URL=postgresql://fer_user:secure_password@localhost:5432/facial_recognition
REDIS_URL=redis://localhost:6379/0
```

#### 4. Initialize Database

```bash
python -c "from backend.database import init_db; init_db()"
```

---

## Advanced Features

### Redis Setup (for Caching & Rate Limiting)

#### 1. Install Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

#### 2. Configure

```bash
export REDIS_URL="redis://localhost:6379/0"
```

### Celery Setup (for Background Tasks)

#### 1. Start Celery Worker

```bash
celery -A backend.tasks worker --loglevel=info
```

#### 2. Start Celery Beat (Scheduler)

```bash
celery -A backend.tasks beat --loglevel=info
```

### WebSocket Real-time Streaming

WebSocket endpoint is automatically available at: `ws://localhost:8000/api/ws/realtime`

Example client:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/realtime');

// Send base64-encoded image
ws.send(JSON.stringify({
  image: canvas.toDataURL('image/jpeg').split(',')[1],
  model: 'yolo'
}));

// Receive predictions
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result);
};
```

---

## Environment Variables

Create `.env` file in project root:

```env
# Database
DATABASE_URL=sqlite:///./facial_recognition.db
# Or PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/facial_recognition

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_KEY_ENABLED=false
DEFAULT_RATE_LIMIT=100

# Model Settings
DEFAULT_MODEL=yolo
DEFAULT_CONFIDENCE_THRESHOLD=0.25

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

## Testing the System

### 1. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# List available models
curl http://localhost:8000/api/models

# Single prediction
curl -X POST http://localhost:8000/api/predict \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST http://localhost:8000/api/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Model comparison
curl -X POST http://localhost:8000/api/compare \
  -F "file=@test_image.jpg"

# Get statistics
curl "http://localhost:8000/api/analytics/statistics?time_range=day"
```

### 2. Interactive API Documentation

Open in browser: **http://localhost:8000/docs**

---

## Production Deployment

### 1. Build Frontend

```bash
cd frontend
npm run build
cd ..
```

### 2. Environment Configuration

```bash
# Set production environment
export DEBUG=false
export DATABASE_URL="postgresql://..."
export REDIS_URL="redis://..."
```

### 3. Use Gunicorn (Production ASGI Server)

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 4. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /api/ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 5. Process Manager (PM2)

```bash
npm install -g pm2

pm2 start main.py --interpreter python3 --name facial-recognition
pm2 startup
pm2 save
```

---

## Performance Optimization

### 1. GPU Acceleration

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Model Optimization

```bash
# Use smaller models for faster inference
python main.py --framework yolo --train --model yolov8n

# Adjust confidence threshold
# Lower = more detections, slower
# Higher = fewer detections, faster
```

### 3. Batch Processing

Use batch API endpoint for multiple images:
- Processes images in parallel
- More efficient than multiple single requests

### 4. Caching

Enable Redis for:
- Result caching
- Rate limit tracking
- Session management

---

## Troubleshooting

### Issue: "Model not found"

```bash
# Train the model first
python main.py --framework yolo --train

# Or download pre-trained weights (if available)
# Place in: models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt
```

### Issue: "CUDA out of memory"

```bash
# Use smaller model
python main.py --framework yolo --train --model yolov8n --mem-profile low

# Or reduce batch size in training config
```

### Issue: "Frontend not loading"

```bash
# Build frontend first
cd frontend
npm install
npm run build
cd ..

# Restart server
python main.py --serve
```

### Issue: "Database connection error"

```bash
# Check DATABASE_URL
echo $DATABASE_URL

# Initialize database
python -c "from backend.database import init_db; init_db()"

# Or use SQLite (default)
unset DATABASE_URL
```

### Issue: "Port already in use"

```bash
# Use different port
python main.py --serve --port 8080

# Or kill process using port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8000    # Windows
```

---

## Development Tips

### 1. Hot Reload Development

```bash
# Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev
```

### 2. Debug Mode

```bash
export DEBUG=true
python main.py --serve
```

### 3. View Logs

```bash
# Application logs
tail -f app.log

# Docker logs
docker-compose logs -f app
```

---

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB
- OS: Linux, macOS, Windows 10+

### Recommended
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 6+ GB VRAM
- Storage: 50 GB SSD
- OS: Linux (Ubuntu 20.04+)

---

## Next Steps

1. ✅ Train your models
2. ✅ Test API endpoints
3. ✅ Customize frontend
4. ✅ Set up analytics
5. ✅ Deploy to production
6. ✅ Monitor performance
7. ✅ Scale as needed

---

## Support & Resources

- **Documentation**: See `README.md`, `API_DOCUMENTATION.md`, `FEATURES.md`
- **API Docs**: http://localhost:8000/docs
- **GitHub**: https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System
- **Issues**: Report bugs via GitHub Issues

---

**You're all set! 🎉**

Start with the Quick Start section and explore advanced features as you go!
