# API Documentation

## Facial Expression Recognition System API

Complete API reference for the Facial Expression Recognition System.

---

## Base URL

```
http://localhost:8000
```

When running locally with default settings.

---

## Endpoints

### Health Check

#### `GET /api/health`

Check API health status and current configuration.

**Response:**
```json
{
  "status": "ok",
  "framework": "yolo",
  "model": "yolov8n"
}
```

---

### Emotion Prediction

#### `POST /api/predict`

Detect faces and predict emotions from an uploaded image.

**Parameters:**
- `file` (form-data, required): Image file (JPG, PNG, WebP)
- `detection_model` (string, optional): Model for face detection
  - Options: `yolo` (default)
- `recognition_model` (string, optional): Model for emotion recognition
  - Options: `yolo` (default), `efficientb3`, `arcface`

**Example Request (cURL):**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@image.jpg" \
  -F "detection_model=yolo" \
  -F "recognition_model=yolo"
```

**Example Request (Python):**
```python
import requests

url = "http://localhost:8000/api/predict"
files = {"file": open("image.jpg", "rb")}
params = {
    "detection_model": "yolo",
    "recognition_model": "yolo"
}

response = requests.post(url, files=files, data=params)
result = response.json()
print(result)
```

**Success Response (200):**
```json
{
  "expression": "happy",
  "confidence": 0.952,
  "all_detections": [
    {
      "expression": "happy",
      "confidence": 0.952,
      "bbox": {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 280.2,
        "y2": 240.1
      }
    },
    {
      "expression": "neutral",
      "confidence": 0.847,
      "bbox": {
        "x1": 320.0,
        "y1": 95.2,
        "x2": 445.8,
        "y2": 220.5
      }
    }
  ],
  "num_faces": 2,
  "framework": "yolo",
  "model": "yolov8n"
}
```

**Error Response - No Faces (200):**
```json
{
  "error": "No faces detected in the image",
  "predictions": [],
  "framework": "yolo",
  "model": "yolov8n"
}
```

**Error Response - Model Not Found (404):**
```json
{
  "error": "Model weights not found. Please train the model first.",
  "path": "models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt"
}
```

---

## Response Fields

### Prediction Response

| Field | Type | Description |
|-------|------|-------------|
| `expression` | string | Primary detected emotion |
| `confidence` | float | Confidence score (0-1) for primary detection |
| `all_detections` | array | List of all face detections with emotions |
| `num_faces` | integer | Total number of faces detected |
| `framework` | string | Framework used for prediction |
| `model` | string | Model variant used |

### Detection Object

| Field | Type | Description |
|-------|------|-------------|
| `expression` | string | Detected emotion class |
| `confidence` | float | Confidence score (0-1) |
| `bbox` | object | Bounding box coordinates |
| `bbox.x1` | float | Top-left X coordinate |
| `bbox.y1` | float | Top-left Y coordinate |
| `bbox.x2` | float | Bottom-right X coordinate |
| `bbox.y2` | float | Bottom-right Y coordinate |

---

## Emotion Classes

The system recognizes 7 emotion classes:

1. **angry** - Anger
2. **disgust** - Disgust
3. **fear** - Fear
4. **happy** - Happiness
5. **neutral** - Neutral expression
6. **sad** - Sadness
7. **surprised** - Surprise

---

## Available Models

### Detection Models

- **yolo** - YOLOv8 face detection (default, recommended)

### Recognition Models

- **yolo** - YOLOv8 classification (fastest, recommended)
- **efficientb3** - EfficientNet-B3 with CBAM (high accuracy)
- **arcface** - ArcFace with ResNet-18 (metric learning)

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input (e.g., corrupted image) |
| 404 | Not Found | Model weights not found |
| 500 | Internal Server Error | Server error during processing |
| 501 | Not Implemented | Requested model not yet implemented |

### Error Response Format

```json
{
  "error": "Error message describing what went wrong",
  "framework": "yolo",
  "model": "yolov8n"
}
```

---

## Interactive Documentation

When the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

---

## Best Practices

### Image Requirements

- **Formats**: JPG, PNG, WebP
- **Size**: No strict limit, but smaller images process faster
- **Quality**: Higher quality images yield better results
- **Faces**: Should be reasonably clear and well-lit
- **Distance**: Faces should be at least 48x48 pixels

### Performance Tips

1. Use `yolo` model for fastest inference
2. Use smaller images when speed is critical
3. Ensure good lighting in images
4. For batch processing, consider using the CLI instead

### Error Handling

Always check for the `error` field in responses:

```python
response = requests.post(url, files=files, data=params)
result = response.json()

if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Detected: {result['expression']} ({result['confidence']:.2f})")
```

---

## Examples

### Complete Python Example

```python
import requests
from pathlib import Path

def predict_emotion(image_path, model="yolo"):
    """Predict emotion from image file."""
    url = "http://localhost:8000/api/predict"
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        params = {
            "detection_model": model,
            "recognition_model": model
        }
        
        response = requests.post(url, files=files, data=params)
        
    if response.status_code == 200:
        result = response.json()
        if "error" not in result:
            return result
        else:
            print(f"Error: {result['error']}")
            return None
    else:
        print(f"HTTP Error: {response.status_code}")
        return None

# Usage
result = predict_emotion("photo.jpg")
if result:
    print(f"Emotion: {result['expression']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Faces: {result['num_faces']}")
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predictEmotion(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));
  form.append('detection_model', 'yolo');
  form.append('recognition_model', 'yolo');

  try {
    const response = await axios.post(
      'http://localhost:8000/api/predict',
      form,
      { headers: form.getHeaders() }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.message);
    return null;
  }
}

// Usage
predictEmotion('photo.jpg').then(result => {
  if (result && !result.error) {
    console.log('Emotion:', result.expression);
    console.log('Confidence:', result.confidence);
  }
});
```

---

## Support

For issues or questions:
- Check the [README](README.md) for setup instructions
- Review [FEATURES](FEATURES.md) for capability details
- Open an issue on GitHub for bugs or feature requests

### Batch Processing

#### `POST /api/predict/batch`

Process multiple images in a single request.

**Parameters:**
- `files` (form-data, required): Array of image files (max 10)
- `detection_model` (string, optional): Detection model
- `recognition_model` (string, optional): Recognition model

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_images": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "file_index": 0,
      "filename": "image1.jpg",
      "expression": "happy",
      "confidence": 0.95,
      "processing_time": 0.234
    }
  ],
  "aggregate_statistics": {
    "emotion_distribution": {
      "happy": 2,
      "sad": 1
    },
    "average_confidence": 0.891
  }
}
```

---

### Video Processing

#### `POST /api/predict/video`

Process video file for emotion detection.

**Parameters:**
- `file` (form-data, required): Video file (max 100MB, 5 minutes)
- `detection_model` (string, optional): Model to use
- `frame_skip` (integer, optional): Process every Nth frame (default: 5)
- `save_annotated` (boolean, optional): Save annotated video (default: false)

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/predict/video \
  -F "file=@video.mp4" \
  -F "frame_skip=10"
```

**Response:**
```json
{
  "video_info": {
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 900,
    "duration": 30.0
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "emotion": "happy",
      "confidence": 0.89,
      "num_faces": 1
    }
  ],
  "emotion_timeline": [
    {"timestamp": 0.0, "emotion": "happy"},
    {"timestamp": 0.33, "emotion": "happy"}
  ],
  "emotion_statistics": {
    "happy": {
      "count": 45,
      "percentage": 75.0
    },
    "sad": {
      "count": 15,
      "percentage": 25.0
    }
  },
  "dominant_emotion": "happy"
}
```

---

### Model Comparison

#### `POST /api/compare`

Compare predictions from multiple models on the same image.

**Parameters:**
- `file` (form-data, required): Image file
- `models` (array, optional): List of models to compare (default: all available)

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/compare \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "comparisons": {
    "yolo": {
      "expression": "happy",
      "confidence": 0.95,
      "processing_time": 0.123,
      "framework": "yolo"
    },
    "efficientb3": {
      "expression": "happy",
      "confidence": 0.89,
      "processing_time": 0.234,
      "framework": "efficientb3"
    },
    "arcface": {
      "expression": "happy",
      "confidence": 0.92,
      "processing_time": 0.189,
      "framework": "arcface"
    }
  },
  "consensus": {
    "emotion": "happy",
    "agreement_percentage": 100.0,
    "total_models": 3
  }
}
```

---

### Ensemble Prediction

#### `POST /api/predict/ensemble`

Use ensemble of models for improved accuracy.

**Parameters:**
- `file` (form-data, required): Image file
- `method` (string, optional): Ensemble method ('weighted', 'majority', 'confidence', 'stacking')

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/predict/ensemble \
  -F "file=@image.jpg" \
  -F "method=weighted"
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.923,
  "ensemble_method": "weighted",
  "num_models": 3,
  "individual_predictions": [
    {
      "emotion": "happy",
      "confidence": 0.95,
      "weight": 1.0,
      "framework": "yolo"
    }
  ]
}
```

---

### Analytics - Statistics

#### `GET /api/analytics/statistics`

Get overall prediction statistics.

**Query Parameters:**
- `time_range` (string, optional): Filter by time ('hour', 'day', 'week', 'month')

**Example Request:**
```bash
curl "http://localhost:8000/api/analytics/statistics?time_range=day"
```

**Response:**
```json
{
  "total_predictions": 1234,
  "unique_sessions": 567,
  "emotion_distribution": {
    "happy": {
      "count": 450,
      "percentage": 36.5
    },
    "sad": {
      "count": 234,
      "percentage": 19.0
    }
  },
  "model_usage": {
    "yolo": {
      "count": 800,
      "percentage": 64.8
    },
    "efficientb3": {
      "count": 434,
      "percentage": 35.2
    }
  },
  "average_processing_time": 0.234,
  "average_confidence": 0.856,
  "time_range": "day"
}
```

---

### Analytics - Trends

#### `GET /api/analytics/trends`

Get emotion trends over time.

**Query Parameters:**
- `interval` (string, optional): Time interval ('hour', 'day')

**Example Request:**
```bash
curl "http://localhost:8000/api/analytics/trends?interval=hour"
```

**Response:**
```json
{
  "interval": "hour",
  "trends": [
    {
      "timestamp": "2025-11-23 14:00",
      "emotions": {
        "happy": 45,
        "sad": 12,
        "neutral": 23
      },
      "total": 80
    }
  ]
}
```

---

### Analytics - Session Info

#### `GET /api/analytics/session/{session_id}`

Get analytics for a specific session.

**Example Request:**
```bash
curl "http://localhost:8000/api/analytics/session/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "first_seen": "2025-11-23T14:30:00",
  "last_seen": "2025-11-23T15:45:00",
  "total_predictions": 25,
  "emotion_distribution": {
    "happy": 15,
    "sad": 7,
    "neutral": 3
  },
  "duration": 4500.0
}
```

---

### Analytics - Export

#### `GET /api/analytics/export`

Export analytics data.

**Query Parameters:**
- `format` (string, required): Export format ('json', 'csv')

**Example Request:**
```bash
curl "http://localhost:8000/api/analytics/export?format=csv" -o analytics.csv
```

---

### Model Management - List Models

#### `GET /api/models`

List all available trained models.

**Example Request:**
```bash
curl http://localhost:8000/api/models
```

**Response:**
```json
{
  "total_models": 3,
  "models": [
    {
      "framework": "yolo",
      "name": "YOLOv8-N",
      "path": "models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt",
      "size": "yolov8n",
      "status": "available"
    },
    {
      "framework": "efficientnetb3",
      "name": "EfficientNet-B3",
      "path": "models/efficientnetb3_model/checkpoints/best_model.pth",
      "status": "available"
    },
    {
      "framework": "arcface",
      "name": "ArcFace (ResNet-18)",
      "path": "models/arcface_model/arcface_model.pt",
      "status": "available"
    }
  ]
}
```

---

### Model Management - Model Info

#### `GET /api/models/{framework}/info`

Get detailed information about a specific model.

**Example Request:**
```bash
curl http://localhost:8000/api/models/yolo/info
```

**Response:**
```json
{
  "framework": "yolo",
  "name": "YOLOv8-N",
  "path": "models/yolo_model/runs/facial_expression_yolov8n/weights/best.pt",
  "size": "yolov8n",
  "status": "available",
  "file_size_mb": 6.2,
  "last_modified": "2025-11-20T10:30:00"
}
```

---

### WebSocket - Real-time Streaming

#### `WS /api/ws/realtime`

WebSocket endpoint for real-time emotion detection.

**Example JavaScript Client:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/realtime');

// Send image frame
ws.send(JSON.stringify({
  image: base64EncodedImage,
  model: 'yolo'
}));

// Receive prediction
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result);
};
```

**Message Format (Send):**
```json
{
  "image": "base64_encoded_image_data",
  "model": "yolo"
}
```

**Message Format (Receive):**
```json
{
  "expression": "happy",
  "confidence": 0.95,
  "num_faces": 1,
  "framework": "yolo"
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error description",
  "detail": "Additional details if available"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `404`: Not Found (resource doesn't exist)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

---

## Examples

### Python Example

```python
import requests

# Single image prediction
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/predict', files=files)
    result = response.json()
    print(f"Emotion: {result['expression']}")
    print(f"Confidence: {result['confidence']:.2%}")

# Batch processing
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/api/predict/batch', files=files)
result = response.json()
print(f"Processed {result['successful']} images")
```

### JavaScript Example

```javascript
// Single image prediction
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Emotion: ${result.expression}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);

// Get analytics
const analytics = await fetch('/api/analytics/statistics?time_range=day');
const stats = await analytics.json();
console.log(`Total predictions today: ${stats.total_predictions}`);
```

### cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/api/predict \
  -F "file=@image.jpg"

# Batch processing
curl -X POST http://localhost:8000/api/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"

# Model comparison
curl -X POST http://localhost:8000/api/compare \
  -F "file=@image.jpg"

# Get statistics
curl "http://localhost:8000/api/analytics/statistics?time_range=week"

# Export analytics to CSV
curl "http://localhost:8000/api/analytics/export?format=csv" -o analytics.csv
```

---

## Best Practices

1. **Image Requirements:**
   - Format: JPEG, PNG
   - Max size: 10MB
   - Recommended: Clear, well-lit faces

2. **Video Requirements:**
   - Format: MP4, AVI
   - Max size: 100MB
   - Max duration: 5 minutes
   - Use `frame_skip` to balance speed/accuracy

3. **Performance:**
   - Use batch processing for multiple images
   - Consider ensemble predictions for critical applications
   - Use WebSocket for real-time streaming

4. **Rate Limiting:**
   - Implement exponential backoff on 429 errors
   - Cache results when possible
   - Consider premium tier for high-volume applications

---

## Support

For issues or questions:
- GitHub: [github.com/Likhith257/EDL_Facial_Expression_Recognition_System](https://github.com/Likhith257/EDL_Facial_Expression_Recognition_System)
- Documentation: http://localhost:8000/docs (FastAPI auto-generated docs)
