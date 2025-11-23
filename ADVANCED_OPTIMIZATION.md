# Advanced Optimization Techniques

This guide covers **production-level optimizations** for deploying the facial expression recognition models at scale.

---

## 1. Model Quantization (4x Speed Boost)

### What is Quantization?
Converts model weights from FP32 (32-bit float) to INT8 (8-bit integer):
- **4x smaller model size**
- **2-4x faster inference**
- **Minimal accuracy loss** (<1-2%)

### Dynamic Quantization (Easiest)

#### For EfficientNet-B3 & ArcFace
```python
import torch

# Load your trained model
model = ... # Your model
model.eval()

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized.pth')

# Use for inference (same API)
with torch.no_grad():
    output = quantized_model(input_tensor)
```

#### For YOLO
```python
from ultralytics import YOLO

# Export to ONNX with INT8 quantization
model = YOLO('models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt')
model.export(format='onnx', int8=True, imgsz=640)

# Load quantized model
quantized_yolo = YOLO('best_int8.onnx')
results = quantized_yolo.predict('image.jpg')
```

---

## 2. ONNX Export (Cross-Platform)

### Benefits
- **Platform independent** (works on CPU, GPU, mobile, edge devices)
- **Faster inference** with ONNX Runtime
- **Smaller deployment package**

### Export EfficientNet-B3 to ONNX
```python
import torch
from models.efficientnetb3_model.src.model import create_model

# Load model
model = create_model(num_classes=7, pretrained=False)
checkpoint = torch.load('models/efficientnetb3_model/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "efficientnetb3.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### Use ONNX Model for Inference
```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession("efficientnetb3.onnx")

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
predictions = outputs[0]
```

---

## 3. TensorRT Optimization (NVIDIA GPUs Only)

### Ultra-Fast Inference
- **5-10x faster** than PyTorch on NVIDIA GPUs
- Optimizes kernels and memory access patterns
- Requires NVIDIA GPU with TensorRT installed

### Export YOLO to TensorRT
```bash
# Install TensorRT (on system with NVIDIA GPU)
pip install tensorrt

# Export YOLO model
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt')
model.export(format='engine', device=0)  # Creates .engine file
"

# Use TensorRT model
python -c "
from ultralytics import YOLO
model = YOLO('best.engine')
results = model.predict('image.jpg')
"
```

### TensorRT for PyTorch Models
```python
import torch_tensorrt

# Load model
model = ...  # Your PyTorch model
model.eval()

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float16},  # Use FP16
)

# Save compiled model
torch.jit.save(trt_model, "model_trt.ts")

# Load and use
trt_model = torch.jit.load("model_trt.ts")
output = trt_model(input_tensor)
```

---

## 4. Batch Inference Optimization

### Process Multiple Images at Once
```python
import torch
from PIL import Image
import torchvision.transforms as transforms

def batch_predict(model, image_paths, batch_size=8):
    """Process images in batches for better GPU utilization"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load and preprocess batch
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            batch_images.append(transform(img))
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        results.extend(outputs.cpu().numpy())
    
    return results

# Usage
predictions = batch_predict(model, ['img1.jpg', 'img2.jpg', ...], batch_size=16)
```

---

## 5. Model Caching (API Optimization)

### Keep Models in Memory
```python
# In main.py - load models once at startup
from functools import lru_cache
import torch

class ModelCache:
    def __init__(self):
        self.yolo_model = None
        self.efficientnet_model = None
        self.arcface_model = None
    
    def get_yolo(self):
        if self.yolo_model is None:
            from ultralytics import YOLO
            self.yolo_model = YOLO('models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt')
        return self.yolo_model
    
    def get_efficientnet(self):
        if self.efficientnet_model is None:
            from models.efficientnetb3_model.src.model import create_model
            self.efficientnet_model = create_model(num_classes=7)
            checkpoint = torch.load('models/efficientnetb3_model/checkpoints/best_model.pth')
            self.efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
            self.efficientnet_model.eval()
        return self.efficientnet_model

# Global cache
model_cache = ModelCache()

# In FastAPI endpoint
@app.post("/api/predict")
async def predict(file: UploadFile):
    model = model_cache.get_yolo()  # Reuses loaded model
    results = model.predict(...)
    return results
```

---

## 6. Result Caching (Redis)

### Cache Predictions for Identical Images
```python
import hashlib
import redis
import json

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_image_hash(image_bytes):
    """Generate hash of image content"""
    return hashlib.md5(image_bytes).hexdigest()

@app.post("/api/predict")
async def predict(file: UploadFile):
    # Read image bytes
    image_bytes = await file.read()
    image_hash = get_image_hash(image_bytes)
    
    # Check cache
    cached_result = redis_client.get(f"prediction:{image_hash}")
    if cached_result:
        return json.loads(cached_result)
    
    # Run prediction
    result = model.predict(...)
    
    # Cache result (expire after 1 hour)
    redis_client.setex(
        f"prediction:{image_hash}",
        3600,  # 1 hour TTL
        json.dumps(result)
    )
    
    return result
```

---

## 7. Async Processing with Background Tasks

### Don't Block API Responses
```python
from fastapi import BackgroundTasks
import asyncio

# In-memory queue for batch processing
prediction_queue = asyncio.Queue()

async def process_prediction_queue():
    """Background worker that processes predictions in batches"""
    while True:
        batch = []
        for _ in range(8):  # Collect 8 images
            try:
                item = await asyncio.wait_for(prediction_queue.get(), timeout=0.1)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        if batch:
            # Batch predict
            results = batch_predict(model, [item['image'] for item in batch])
            for item, result in zip(batch, results):
                item['future'].set_result(result)

@app.on_event("startup")
async def startup():
    asyncio.create_task(process_prediction_queue())

@app.post("/api/predict-async")
async def predict_async(file: UploadFile):
    """Add to queue and return immediately"""
    future = asyncio.Future()
    await prediction_queue.put({'image': await file.read(), 'future': future})
    result = await future
    return result
```

---

## 8. Ensemble Predictions (Maximum Accuracy)

### Combine All Three Models
```python
def ensemble_predict(image_path):
    """Combine predictions from YOLO, EfficientNet, and ArcFace"""
    from collections import Counter
    
    # Get predictions from each model
    yolo_pred = predict_yolo(image_path)
    efficientnet_pred = predict_efficientnet(image_path)
    arcface_pred = predict_arcface(image_path)
    
    # Weighted voting (adjust weights based on model performance)
    weights = {'yolo': 0.4, 'efficientnet': 0.4, 'arcface': 0.2}
    
    # Soft voting (average probabilities)
    emotion_probs = {}
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    for emotion in emotions:
        emotion_probs[emotion] = (
            weights['yolo'] * yolo_pred['probs'][emotion] +
            weights['efficientnet'] * efficientnet_pred['probs'][emotion] +
            weights['arcface'] * arcface_pred['probs'][emotion]
        )
    
    # Return highest probability emotion
    final_emotion = max(emotion_probs, key=emotion_probs.get)
    final_confidence = emotion_probs[final_emotion]
    
    return {
        'emotion': final_emotion,
        'confidence': final_confidence,
        'individual_predictions': {
            'yolo': yolo_pred,
            'efficientnet': efficientnet_pred,
            'arcface': arcface_pred
        }
    }
```

---

## 9. Test-Time Augmentation (TTA)

### Predict on Multiple Augmented Versions
```python
def predict_with_tta(model, image, num_augmentations=5):
    """Average predictions over multiple augmented versions"""
    import torchvision.transforms as T
    
    tta_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomRotation(5)
    ])
    
    predictions = []
    for _ in range(num_augmentations):
        augmented = tta_transforms(image)
        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
        predictions.append(pred.softmax(dim=1))
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

---

## 10. Deployment Checklist

### Production-Ready Optimizations
- [ ] **Model quantization** (INT8 or FP16)
- [ ] **ONNX export** for cross-platform compatibility
- [ ] **Model caching** (load once at startup)
- [ ] **Batch inference** for multiple images
- [ ] **Result caching** with Redis/Memcached
- [ ] **Async processing** for non-blocking API
- [ ] **Health monitoring** (GPU usage, latency, throughput)
- [ ] **Auto-scaling** based on load
- [ ] **Model versioning** for rollbacks
- [ ] **A/B testing** for model improvements

### Performance Monitoring
```python
import time
from prometheus_client import Counter, Histogram

# Metrics
prediction_count = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/api/predict")
async def predict(file: UploadFile):
    start_time = time.time()
    
    # Run prediction
    result = model.predict(...)
    
    # Record metrics
    prediction_count.inc()
    prediction_latency.observe(time.time() - start_time)
    
    return result
```

---

## Expected Performance Gains

| Optimization | Speed Improvement | Accuracy Impact |
|--------------|-------------------|-----------------|
| **Quantization (INT8)** | 2-4x faster | -1% to -2% |
| **ONNX Runtime** | 1.5-2x faster | No change |
| **TensorRT (NVIDIA)** | 5-10x faster | No change |
| **Batch Inference (16)** | 3-5x throughput | No change |
| **Model Caching** | 50-100ms saved per request | No change |
| **Result Caching** | Near-instant for cached | No change |
| **Ensemble** | Slower (3x models) | +2% to +5% |
| **TTA** | Slower (5x augmentations) | +1% to +3% |

---

## Quick Start: Production Deployment

```bash
# 1. Quantize all models
python scripts/quantize_models.py

# 2. Export to ONNX
python scripts/export_onnx.py

# 3. Start optimized server with caching
python main.py serve --optimized --cache-enabled --batch-size 16

# 4. Monitor performance
python main.py monitor --metrics-port 9090
```

---

Ready for production! 🚀
