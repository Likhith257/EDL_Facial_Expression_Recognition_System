"""
Main script to run the complete facial expression recognition pipeline.
Supports both YOLO and ArcFace models.
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
    else:  # arcface
        required_packages = [
            'torch', 'cv2', 'numpy', 
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
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install them using:")
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
    else:  # arcface
        print("⚠️  ArcFace dataset preparation not yet implemented")
        # from models.arcface_model.src.prepare_dataset import main as prepare_main
        # prepare_main()


def train_model(framework='yolo', model_size='yolov8n'):
    """Run model training."""
    print("\n" + "=" * 60)
    print(f"STEP 2: TRAINING MODEL ({framework.upper()}: {model_size})")
    print("=" * 60)
    
    if framework == 'yolo':
        from models.yolo_model.src.train_model import main as train_main
        train_main(model_size=model_size)
    else:  # arcface
        print("⚠️  ArcFace training not yet implemented")
        # from models.arcface_model.src.train_model import main as train_main
        # train_main()


def evaluate_model(framework='yolo', model_size='yolov8n'):
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print(f"STEP 3: EVALUATING MODEL ({framework.upper()}: {model_size})")
    print("=" * 60)
    
    if framework == 'yolo':
        from models.yolo_model.src.evaluate import main as eval_main
        eval_main(model_size=model_size)
    else:  # arcface
        print("⚠️  ArcFace evaluation not yet implemented")
        # from models.arcface_model.src.evaluate import main as eval_main
        # eval_main()


def run_inference(framework='yolo', model_size='yolov8n'):
    """Run inference."""
    print("\n" + "=" * 60)
    print(f"STEP 4: RUNNING INFERENCE ({framework.upper()}: {model_size})")
    print("=" * 60)
    
    if framework == 'yolo':
        from models.yolo_model.src.predict import main as predict_main
        predict_main(model_size=model_size)
    else:  # arcface
        print("⚠️  ArcFace inference not yet implemented")
        # from models.arcface_model.src.predict import main as predict_main
        # predict_main()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO Model Examples:
  python main.py --framework yolo --all
  python main.py --framework yolo --prepare
  python main.py --framework yolo --train --model yolov8s
  python main.py --framework yolo --predict --model yolov8n
  
  # ArcFace Model Examples:
  python main.py --framework arcface --all
  python main.py --framework arcface --train
  python main.py --framework arcface --predict
  
Available YOLO model sizes:
  - yolov8n: Nano (fastest, lowest accuracy)
  - yolov8s: Small
  - yolov8m: Medium (balanced)
  - yolov8l: Large
  - yolov8x: Extra Large (slowest, highest accuracy)
        """
    )
    
    parser.add_argument(
        '--framework',
        type=str,
        default='yolo',
        choices=['yolo', 'arcface', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='Choose framework: yolo, arcface, or yolov8[n/s/m/l/x] as shorthand (default: yolo)'
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
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='YOLO model size (default: yolov8n). Only used with --framework yolo'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Handle shorthand: if framework is a yolo model size, extract it
    if args.framework.startswith('yolov8'):
        args.model = args.framework
        args.framework = 'yolo'
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies(framework=args.framework):
            return
    
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION SYSTEM")
    if args.framework == 'yolo':
        print(f"Using YOLOv8 ({args.model}) for Detection and Classification")
    else:
        print("Using ArcFace for Facial Expression Recognition")
    print("=" * 60)
    
    # Run requested operations
    if args.all:
        prepare_dataset(framework=args.framework)
        train_model(framework=args.framework, model_size=args.model)
        evaluate_model(framework=args.framework, model_size=args.model)
    else:
        if args.prepare:
            prepare_dataset(framework=args.framework)
        
        if args.train:
            train_model(framework=args.framework, model_size=args.model)
        
        if args.evaluate:
            evaluate_model(framework=args.framework, model_size=args.model)
        
        if args.predict:
            run_inference(framework=args.framework, model_size=args.model)
    
    if not any([args.all, args.prepare, args.train, args.evaluate, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
