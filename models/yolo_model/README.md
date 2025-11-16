# YOLO Model - Facial Expression Recognition

This folder contains the YOLO (You Only Look Once) implementation for facial expression recognition.

## Structure
```
yolo_model/
├── main.py                              # Main pipeline script
├── config.yaml                          # YOLO configuration
├── yolov8n.pt                          # Pre-trained YOLOv8 weights
├── requirements.txt                     # Python dependencies
├── facial_expression_recognition.ipynb  # Jupyter notebook
├── src/                                # Source code
│   ├── prepare_dataset.py              # Dataset preparation
│   ├── train_model.py                  # Model training
│   ├── evaluate.py                     # Model evaluation
│   ├── predict.py                      # Inference/prediction
│   └── utils.py                        # Utility functions
└── runs/                               # Training outputs
    ├── train/                          # Training results
    ├── evaluation/                     # Evaluation metrics
    └── detect/                         # Detection results
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python main.py
   ```

3. **Or run individual steps:**
   ```bash
   # Prepare dataset
   python src/prepare_dataset.py
   
   # Train model
   python src/train_model.py
   
   # Evaluate model
   python src/evaluate.py
   
   # Make predictions
   python src/predict.py
   ```

## Model Details

- **Architecture:** YOLOv8n (nano - lightweight version)
- **Classes:** 7 emotions (angry, disgust, fear, happy, neutral, sad, surprised)
- **Input Size:** 640x640
- **Framework:** Ultralytics YOLO

## Configuration

Edit `config.yaml` to customize:
- Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Training hyperparameters (epochs, batch size, learning rate)
- Data augmentation settings
- Device selection (CPU/GPU)

## Dataset

The model expects data in the following structure:
```
../../dataset/
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

## Results

Training results and metrics are saved in the `runs/` directory:
- Training curves and metrics
- Confusion matrices
- Model weights (best and last)
- Evaluation reports
