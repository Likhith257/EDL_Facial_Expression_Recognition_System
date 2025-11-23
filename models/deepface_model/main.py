import argparse
from src.prepare_dataset import load_dataset, save_dataset
from src.train_model import fine_tune_model
from src.evaluate import evaluate_model
from src.predict import predict_emotions

def main():
    parser = argparse.ArgumentParser(description="DeepFace FER Pipeline")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], required=True, help="Pipeline mode")
    parser.add_argument('--data_dir', default=r'C:\EDL\EDL_Proj2\deepface\DATASET - To Students', help="Path to dataset folder")
    parser.add_argument('--image_path', help="Path to image or folder for prediction")
    parser.add_argument('--model_path', default='runs/train/final_model.h5', help="Path to fine-tuned model")
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Loading and preprocessing dataset...")
        images, labels = load_dataset(args.data_dir, label_depth=2)
        save_dataset(images, labels)
        print("Fine-tuning model...")
        fine_tune_model(images, labels, epochs=5)
        print("Training complete. Outputs in runs/train/")
    
    elif args.mode == 'evaluate':
        print("Loading test dataset...")
        images, labels = load_dataset(args.data_dir, label_depth=2)
        print("Evaluating model...")
        evaluate_model(args.model_path, images, labels)
        print("Evaluation complete. Metrics in runs/evaluation/")
    
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("Provide --image_path for prediction")
        print("Predicting emotions...")
        predictions = predict_emotions(args.image_path, args.model_path)
        for path, emotion, conf in predictions:
            print(f"{path}: {emotion} ({conf:.2f})")
        print("Predictions saved in runs/predictions/")

if __name__ == "__main__":
    main()