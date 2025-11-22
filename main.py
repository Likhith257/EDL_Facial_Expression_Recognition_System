"""
Main script for facial expression recognition pipeline.
"""

import sys
import argparse
from pathlib import Path


def check_dependencies(framework='yolo'):
    """Check if required packages are installed."""
    if framework == 'yolo':
        required_packages = [
            'ultralytics', 'torch', 'cv2', 'numpy', 
            'sklearn', 'matplotlib', 'pandas'
        ]
    elif framework in ['efficientnet', 'efficientnetb3']:
        required_packages = [
            'torch', 'torchvision', 'cv2', 'numpy', 
            'sklearn', 'matplotlib', 'pandas', 'PIL'
        ]
    elif framework == 'efficientnetv2':
        required_packages = [
            'torch', 'torchvision', 'cv2', 'numpy',
            'sklearn', 'matplotlib', 'pandas', 'PIL'
        ]
    else:  # arcface, vit, swin
        required_packages = [
            'torch', 'torchvision', 'cv2', 'numpy', 
            'sklearn', 'matplotlib', 'pandas'
        ]
    
    requirements_path = "requirements.txt"
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall them using:")
        print(f"   pip install -r {requirements_path}")
        return False
    return True


def prepare_dataset(framework='yolo'):
    """Run dataset preparation."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARING DATASET")
    print("=" * 60)
    if framework == 'yolo':
        from models.yolo_model.src.prepare_dataset import main as prepare_main
        prepare_main()
    elif framework in ['efficientnet', 'efficientnetb3']:
        print("Using shared dataset (already prepared by YOLO)")
        print("EfficientNet-B3 uses the same dataset structure")
    elif framework == 'efficientnetv2':
        print("Expecting pre-cropped classification dataset (train/ val folders by class)")
        print("Use forthcoming face-crop script to build this dataset.")
    else:
        print("Dataset preparation not yet implemented for this framework")
        print("Use YOLO dataset preparation: python main.py --framework yolo --prepare")


def train_model(framework='yolo', model_size='yolov8n', memory_profile='default'):
    """Run model training."""
    print("\n" + "=" * 60)
    print(f"STEP 2: TRAINING MODEL ({framework.upper()}: {model_size})")
    print("=" * 60)
    if framework == 'yolo':
        from models.yolo_model.src.train_model import main as train_main
        train_main(model_size=model_size, memory_profile=memory_profile)
    elif framework in ['efficientnet', 'efficientnetb3']:
        from models.efficientnetb3_model.src.train import main as train_main
        train_main()
    elif framework == 'efficientnetv2':
        from models.efficientnetv2_model.src.train import main as train_main
        train_main()
    elif framework == 'arcface':
        from models.arcface_model.src.train import main as train_main
        train_main()
    else:
        print("Training not yet implemented for this framework")


def evaluate_model(framework='yolo', model_size='yolov8n'):
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print(f"STEP 3: EVALUATING MODEL ({framework.upper()}: {model_size})")
    print("=" * 60)
    if framework == 'yolo':
        from models.yolo_model.src.evaluate import main as eval_main
        eval_main(model_size=model_size)
    elif framework in ['efficientnet', 'efficientnetb3']:
        from models.efficientnetb3_model.src.evaluate import main as eval_main
        eval_main()
    elif framework == 'efficientnetv2':
        print("Evaluation script for EfficientNetV2 not yet implemented.")
    else:
        print("Evaluation not yet implemented for this framework")


def run_inference(framework='yolo', model_size='yolov8n'):
    """Run inference."""
    print("\n" + "=" * 60)
    print(f"STEP 4: RUNNING INFERENCE ({framework.upper()}: {model_size})")
    print("=" * 60)
    if framework == 'yolo':
        from models.yolo_model.src.predict import main as predict_main
        predict_main(model_size=model_size)
    elif framework in ['efficientnet', 'efficientnetb3']:
        from models.efficientnetb3_model.src.predict import main as predict_main
        predict_main()
    elif framework == 'efficientnetv2':
        print("Inference script for EfficientNetV2 not yet implemented.")
    else:
        print("Inference not yet implemented for this framework")


def serve_web_app(framework='yolo', model_size='yolov8n', port=8000):
    """Start web server with API and frontend."""
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
    except ImportError:
        print("Web server dependencies not installed.")
        print("Install them using: pip install fastapi uvicorn python-multipart")
        return
    
    print("\n" + "=" * 60)
    print(f"STARTING WEB SERVER ({framework.upper()}: {model_size})")
    print("=" * 60)
    
    app = FastAPI(title="Facial Expression Recognition API")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Serve frontend static files
    frontend_dist = Path("frontend/dist/spa")
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        # Mount assets directory
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        
        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(frontend_dist / "index.html"))
        
        @app.get("/favicon.ico")
        async def serve_favicon():
            favicon_path = frontend_dist / "favicon.ico"
            if favicon_path.exists():
                return FileResponse(str(favicon_path))
            return JSONResponse(content={"error": "Not found"}, status_code=404)
        
        # Catch-all route for client-side routing (SPA)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Only serve index.html for non-API routes
            if not full_path.startswith("api/"):
                return FileResponse(str(frontend_dist / "index.html"))
            return JSONResponse(content={"error": "Not found"}, status_code=404)
    else:
        print(f"Warning: Frontend build not found at {frontend_dist}")
        print("Build the frontend first: cd frontend && npm install && npm run build")
    
    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "framework": framework, "model": model_size}
    
    @app.post("/api/predict")
    async def predict_expression(
        file: UploadFile = File(...),
        detection_model: str = 'yolo',
        recognition_model: str = 'yolo'
    ):
        """Predict facial expression from uploaded image."""
        import tempfile
        import os
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Determine which model to use based on parameters
            selected_framework = recognition_model if recognition_model != 'yolo' else detection_model
            
            # Load the trained model
            if selected_framework == 'yolo':
                weights_path = Path(f"models/yolo_model/runs/facial_expression_{model_size}/weights/best.pt")
                if not weights_path.exists():
                    return JSONResponse(
                        content={
                            "error": "Model weights not found. Please train the model first.",
                            "path": str(weights_path)
                        },
                        status_code=404
                    )
                
                # Load model and predict
                model = YOLO(str(weights_path))
                
                # Read and process image
                image = cv2.imread(tmp_path)
                if image is None:
                    return JSONResponse(
                        content={"error": "Could not read uploaded image"},
                        status_code=400
                    )
                
                # Run inference with lower confidence for better detection
                results = model.predict(
                    source=image, 
                    conf=0.15,  # Lower confidence threshold for real-world images
                    iou=0.45,   # NMS IoU threshold
                    verbose=False
                )[0]
                
                # Emotion labels
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
                
                # Extract predictions
                predictions = []
                if len(results.boxes) > 0:
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        
                        predictions.append({
                            "expression": emotion_labels[class_id],
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": round(bbox[0], 1),
                                "y1": round(bbox[1], 1),
                                "x2": round(bbox[2], 1),
                                "y2": round(bbox[3], 1)
                            }
                        })
                
                if not predictions:
                    return JSONResponse(content={
                        "error": "No faces detected in the image",
                        "predictions": [],
                        "framework": framework,
                        "model": model_size
                    })
                
                # Return the highest confidence prediction as primary
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                result = {
                    "expression": predictions[0]["expression"],
                    "confidence": predictions[0]["confidence"],
                    "all_detections": predictions,
                    "num_faces": len(predictions),
                    "framework": "yolo",
                    "model": model_size
                }
                return JSONResponse(content=result)
            
            elif selected_framework in ['efficientb3', 'efficientnet', 'efficientnetb3']:
                # Use EfficientNet-B3 model
                import torch
                from torchvision import transforms
                from PIL import Image
                
                checkpoint_path = Path("models/efficientnetb3_model/checkpoints/best_model.pth")
                if not checkpoint_path.exists():
                    return JSONResponse(
                        content={
                            "error": "EfficientNet-B3 model weights not found. Please train the model first.",
                            "path": str(checkpoint_path)
                        },
                        status_code=404
                    )
                
                # Import predictor
                sys.path.insert(0, str(Path("models/efficientnetb3_model/src")))
                from predict import EmotionPredictor
                
                # Initialize predictor
                predictor = EmotionPredictor(
                    checkpoint_path=checkpoint_path,
                    config_path=Path("models/efficientnetb3_model/config.yaml"),
                    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                )
                
                # Read image
                image = cv2.imread(tmp_path)
                if image is None:
                    return JSONResponse(
                        content={"error": "Could not read uploaded image"},
                        status_code=400
                    )
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization for better face detection
                gray = cv2.equalizeHist(gray)
                
                # Detect faces using Haar Cascade with adjusted parameters for real-world images
                faces = predictor.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05,  # More sensitive scaling
                    minNeighbors=3,    # Lower threshold for detection
                    minSize=(48, 48),  # Slightly larger minimum size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # If no faces found, try alternative cascade
                if len(faces) == 0:
                    faces = predictor.face_cascade_alt.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(48, 48),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                
                predictions = []
                for (x, y, w, h) in faces:
                    # Add padding around face for better context (10% on each side)
                    padding = int(max(w, h) * 0.1)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    # Extract face with padding
                    face_img = image[y1:y2, x1:x2]
                    
                    # Skip if face is too small
                    if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                        continue
                    
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # Predict with enhancement enabled for real-world images
                    emotion, confidence = predictor.predict_image(face_pil, apply_enhancement=True)
                    
                    predictions.append({
                        "expression": emotion,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": float(x),
                            "y1": float(y),
                            "x2": float(x + w),
                            "y2": float(y + h)
                        }
                    })
                
                if not predictions:
                    return JSONResponse(content={
                        "error": "No faces detected in the image",
                        "predictions": [],
                        "framework": "efficientnetb3",
                        "model": "efficientnet-b3"
                    })
                
                # Return the highest confidence prediction as primary
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                result = {
                    "expression": predictions[0]["expression"],
                    "confidence": predictions[0]["confidence"],
                    "all_detections": predictions,
                    "num_faces": len(predictions),
                    "framework": "efficientnetb3",
                    "model": "efficientnet-b3"
                }
                return JSONResponse(content=result)
            
            elif selected_framework == 'arcface':
                # Use ArcFace model
                import torch
                from PIL import Image
                
                checkpoint_path = Path("models/arcface_model/arcface_model.pt")
                if not checkpoint_path.exists():
                    return JSONResponse(
                        content={
                            "error": "ArcFace model weights not found. Please train the model first.",
                            "path": str(checkpoint_path)
                        },
                        status_code=404
                    )
                
                # Import predictor
                # We need to be careful with sys.path to avoid conflicts if multiple models have 'src.predict'
                # But here we are inside a function, so imports are local scope-ish, but sys.path is global.
                # Better to use absolute imports if possible, but the structure relies on 'src'
                # Let's append temporarily
                model_src = str(Path("models/arcface_model/src"))
                if model_src not in sys.path:
                    sys.path.insert(0, model_src)
                
                try:
                    # Force reload to ensure we get the right module if names collide
                    import predict
                    import importlib
                    importlib.reload(predict)
                    from predict import EmotionPredictor
                except ImportError:
                    # Fallback
                    from models.arcface_model.src.predict import EmotionPredictor

                # Initialize predictor
                predictor = EmotionPredictor(
                    checkpoint_path=checkpoint_path,
                    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                )
                
                # Read image
                image = cv2.imread(tmp_path)
                if image is None:
                    return JSONResponse(
                        content={"error": "Could not read uploaded image"},
                        status_code=400
                    )
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Detect faces
                faces = predictor.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(48, 48),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) == 0:
                    faces = predictor.face_cascade_alt.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=3, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
                    )
                
                predictions = []
                for (x, y, w, h) in faces:
                    padding = int(max(w, h) * 0.1)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    face_img = image[y1:y2, x1:x2]
                    if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                        continue
                    
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    emotion, confidence = predictor.predict_image(face_pil, apply_enhancement=True)
                    
                    predictions.append({
                        "expression": emotion,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)
                        }
                    })
                
                if not predictions:
                    return JSONResponse(content={
                        "error": "No faces detected",
                        "predictions": [],
                        "framework": "arcface",
                        "model": "resnet18-arcface"
                    })
                
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                result = {
                    "expression": predictions[0]["expression"],
                    "confidence": predictions[0]["confidence"],
                    "all_detections": predictions,
                    "num_faces": len(predictions),
                    "framework": "arcface",
                    "model": "resnet18-arcface"
                }
                return JSONResponse(content=result)

            else:
                return JSONResponse(
                    content={
                        "error": f"Framework '{selected_framework}' not yet implemented for API. Available: yolo, efficientb3, arcface",
                        "framework": selected_framework,
                        "note": "Swin and ViT models are not yet implemented"
                    },
                    status_code=501
                )
        
        except Exception as e:
            return JSONResponse(
                content={
                    "error": f"Prediction failed: {str(e)}",
                    "framework": selected_framework if 'selected_framework' in locals() else framework
                },
                status_code=500
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    print(f"\nServer starting on http://localhost:{port}")
    print("API documentation: http://localhost:{port}/docs")
    if frontend_dist.exists():
        print(f"Frontend: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--framework',
        type=str,
        default='yolo',
        choices=['yolo', 'arcface', 'efficientnet', 'efficientnetb3', 'efficientnetv2', 'vit', 'swin', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='Choose framework: yolo, efficientnet(b3/v2), vit, swin, or yolov8[n/s/m/l/x] shorthand (default: yolo)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (prepare, train, evaluate)'
    )
    
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Prepare dataset'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run inference'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start web server with API and frontend'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for web server (default: 8000)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='YOLO model size (default: yolov8n). Only used with --framework yolo'
    )

    parser.add_argument(
        '--mem-profile',
        type=str,
        default='default',
        choices=['low', 'medium', 'high', 'default'],
        help='Memory profile for YOLO training: low, medium, high, default'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    if args.framework.startswith('yolov8'):
        args.model = args.framework
        args.framework = 'yolo'
    elif args.framework in ['efficientnetb3']:
        args.framework = 'efficientnet'
    
    if not args.skip_deps:
        if not check_dependencies(framework=args.framework):
            return
    
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION SYSTEM")
    if args.framework == 'yolo':
        print(f"Using YOLOv8 ({args.model}) for Detection and Classification")
    elif args.framework == 'efficientnet':
        print("Using EfficientNet-B3 with CBAM Attention")
    elif args.framework == 'efficientnetv2':
        print("Using EfficientNetV2-S for High-Accuracy Classification")
    elif args.framework == 'vit':
        print("Using Vision Transformer (ViT)")
    elif args.framework == 'swin':
        print("Using Swin Transformer")
    else:
        print(f"Using {args.framework.upper()} for Facial Expression Recognition")
    print("=" * 60)
    
    # Run requested operations
    if args.serve:
        serve_web_app(framework=args.framework, model_size=args.model, port=args.port)
        return
    
    if args.all:
        prepare_dataset(framework=args.framework)
        train_model(framework=args.framework, model_size=args.model, memory_profile=args.mem_profile)
        evaluate_model(framework=args.framework, model_size=args.model)
    else:
        if args.prepare:
            prepare_dataset(framework=args.framework)
        
        if args.train:
            train_model(framework=args.framework, model_size=args.model, memory_profile=args.mem_profile)
        
        if args.evaluate:
            evaluate_model(framework=args.framework, model_size=args.model)
        
        if args.predict:
            run_inference(framework=args.framework, model_size=args.model)
    
    if not any([args.all, args.prepare, args.train, args.evaluate, args.predict, args.serve]):
        parser.print_help()


if __name__ == "__main__":
    main()
