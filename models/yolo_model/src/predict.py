import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import argparse
from datetime import datetime

_THIS_FILE = Path(__file__).resolve()
YOLO_ROOT = _THIS_FILE.parents[1]
PROJECT_ROOT = _THIS_FILE.parents[3]


class FacialExpressionPredictor:
    def __init__(self, weights_path, conf_threshold=0.25):
        """
        Initialize the facial expression predictor.
        
        Args:
            weights_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold for detections
        """
        # Select best available device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        
        # Emotion labels
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Neutral', 'Sad', 'Surprised'
        ]
        
        # Color mapping for each emotion (BGR format)
        self.emotion_colors = {
            0: (0, 0, 255),      # Angry - Red
            1: (0, 165, 255),    # Disgust - Orange
            2: (128, 0, 128),    # Fear - Purple
            3: (0, 255, 0),      # Happy - Green
            4: (255, 255, 255),  # Neutral - White
            5: (255, 0, 0),      # Sad - Blue
            6: (0, 255, 255),    # Surprised - Yellow
        }
        
        print(f"✅ Model loaded from: {weights_path}")
        print(f"🖥️  Inference device: {self.device}")
    
    def predict_image(self, image_path, save_output=True):
        """
        Predict facial expression in a single image.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated output
        
        Returns:
            Annotated image and predictions
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return None, None
        
        # Perform inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Annotate image
        annotated_image = self.annotate_image(image.copy(), results)
        
        # Save output
        if save_output:
            output_dir = Path('runs/predict')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"prediction_{timestamp}.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"💾 Saved prediction to: {output_path}")
        
        return annotated_image, results
    
    def predict_video(self, video_source=0, save_output=False, display=True):
        """
        Predict facial expressions in video stream.
        
        Args:
            video_source: Video file path or camera index (0 for webcam)
            save_output: Whether to save output video
            display: Whether to display video window
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Could not open video source: {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        out = None
        if save_output:
            output_dir = Path('runs/predict')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"video_prediction_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Saving video to: {output_path}")
        
        print("🎥 Processing video... Press 'q' to quit")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform inference
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False
                )[0]
                
                # Annotate frame
                annotated_frame = self.annotate_image(frame, results)
                
                # Add frame counter
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Save frame
                if out is not None:
                    out.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Facial Expression Recognition', annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"✅ Processed {frame_count} frames")
    
    def predict_webcam(self, save_output=False):
        """
        Real-time prediction using webcam.
        
        Args:
            save_output: Whether to save output video
        """
        print("📹 Starting webcam prediction...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        self.predict_video(video_source=0, save_output=save_output, display=True)
    
    def annotate_image(self, image, results):
        """
        Annotate image with predictions.
        
        Args:
            image: Input image (BGR)
            results: YOLO prediction results
        
        Returns:
            Annotated image
        """
        # Get detections
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            # No detections
            cv2.putText(
                image,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return image
        
        # Process each detection
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get emotion label and color
            emotion = self.emotion_labels[cls]
            color = self.emotion_colors[cls]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{emotion}: {conf:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )
            
            # Draw background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
        
        return image
    
    def get_emotion_statistics(self, results):
        """
        Get statistics about detected emotions.
        
        Args:
            results: YOLO prediction results
        
        Returns:
            Dictionary of emotion counts and confidences
        """
        stats = {
            'total_faces': 0,
            'emotions': {emotion: {'count': 0, 'avg_conf': 0.0} 
                        for emotion in self.emotion_labels}
        }
        
        if results.boxes is None or len(results.boxes) == 0:
            return stats
        
        stats['total_faces'] = len(results.boxes)
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            emotion = self.emotion_labels[cls]
            
            stats['emotions'][emotion]['count'] += 1
            stats['emotions'][emotion]['avg_conf'] += conf
        
        # Calculate average confidences
        for emotion in stats['emotions']:
            count = stats['emotions'][emotion]['count']
            if count > 0:
                stats['emotions'][emotion]['avg_conf'] /= count
        
        return stats


def main(model_size='yolov8n'):
    """Main inference execution.
    
    Args:
        model_size: YOLO model size to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    """
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition using YOLO'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to trained model weights (defaults to YOLO runs/best.pt)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (0 for webcam, path to image/video file)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output'
    )
    
    args = parser.parse_args()
    
    # Determine weights path - try multiple possible locations
    possible_paths = [
        YOLO_ROOT / 'runs' / 'train' / f'facial_expression_{model_size}' / 'weights' / 'best.pt',
        YOLO_ROOT / 'runs' / f'facial_expression_{model_size}' / 'weights' / 'best.pt',
    ]
    
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = None
        for path in possible_paths:
            if path.exists():
                weights_path = path
                break
    
    # Check if weights exist
    if weights_path is None or not weights_path.exists():
        print(f"❌ Weights not found in any of these locations:")
        for path in possible_paths:
            print(f"   • {path}")
        print(f"\nPlease train the {model_size} model first using:")
        print(f"   python main.py --train --model {model_size}")
        return
    
    # Initialize predictor
    predictor = FacialExpressionPredictor(
        weights_path=str(weights_path),
        conf_threshold=args.conf
    )
    
    # Determine source type
    if args.source == '0':
        # Webcam
        predictor.predict_webcam(save_output=args.save)
    
    elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image file
        print(f"📷 Processing image: {args.source}")
        predictor.predict_image(args.source, save_output=True)
    
    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video file
        print(f"🎬 Processing video: {args.source}")
        predictor.predict_video(
            video_source=args.source,
            save_output=args.save,
            display=True
        )
    
    else:
        print(f"❌ Unsupported source: {args.source}")
        return
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
