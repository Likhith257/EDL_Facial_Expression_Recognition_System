"""
Main script to run the complete facial expression recognition pipeline.
"""

import sys
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'ultralytics', 'torch', 'cv2', 'numpy', 
        'sklearn', 'matplotlib', 'pandas'
    ]
    
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
        print("   pip install -r requirements.txt")
        return False
    
    return True


def prepare_dataset():
    """Run dataset preparation."""
    print("\n" + "=" * 60)
    print("STEP 1: PREPARING DATASET")
    print("=" * 60)
    
    from src.prepare_dataset import main as prepare_main
    prepare_main()


def train_model():
    """Run model training."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODEL")
    print("=" * 60)
    
    from src.train_model import main as train_main
    train_main()


def evaluate_model():
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING MODEL")
    print("=" * 60)
    
    from src.evaluate import main as eval_main
    eval_main()


def run_inference():
    """Run inference."""
    print("\n" + "=" * 60)
    print("STEP 4: RUNNING INFERENCE")
    print("=" * 60)
    
    from src.predict import main as predict_main
    predict_main()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition System using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --all
  
  # Prepare dataset only
  python main.py --prepare
  
  # Train model only
  python main.py --train
  
  # Evaluate model only
  python main.py --evaluate
  
  # Run inference on webcam
  python main.py --predict
        """
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
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            return
    
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION SYSTEM")
    print("Using YOLOv8 for Detection and Classification")
    print("=" * 60)
    
    # Run requested operations
    if args.all:
        prepare_dataset()
        train_model()
        evaluate_model()
    else:
        if args.prepare:
            prepare_dataset()
        
        if args.train:
            train_model()
        
        if args.evaluate:
            evaluate_model()
        
        if args.predict:
            run_inference()
    
    if not any([args.all, args.prepare, args.train, args.evaluate, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
