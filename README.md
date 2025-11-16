# EDL_Facial_Expression_Recognition_System
## Project Overview
This project implements an end-to-end FER pipeline that:

Loads and preprocesses images from a nested dataset structure.
Fine-tunes DeepFace’s SFace model for emotion classification.
Evaluates performance and predicts emotions on new images.
Runs efficiently on CPU, with optional Jupyter notebook exploration.

## Getting Started
## Prerequisites

Python: 3.8 or higher.
OS: Windows (tested with PowerShell).
Dependencies: Install via:powershellpip install -r requirements.txt

If deepface installation fails, use:
pip install tensorflow-cpu
pip install deepface


## Setup Instructions

## Create Project Directory:
Make a folder: mkdir path/to/directory and copy all project files into it.

## Prepare the Dataset:
Link your dataset (admin CMD): mklink /D data "path/to/dataset"
Or copy it: xcopy "path/to/dataset" data /E /H /C /I


## Verify Setup:
Test data loading:powershellpython -c "from src.prepare_dataset import load_dataset; images, labels = load_dataset(r'C:\EDL\EDL_Proj2\deepface\DATASET - To Students', label_depth=2); print(f'Loaded {len(images)} images')"
Adjust label_depth (e.g., to 3) if emotions are deeper (e.g., .../train/happy/).


## Usage
Run the Pipeline
Use main.py with different modes via PowerShell:

## Train the Model:
Command: python main.py --mode train
Details: Fine-tunes the model on your dataset, saving weights to runs/train/. Expect 2-10 hours on CPU.

## Evaluate Performance:
Command: python main.py --mode evaluate
Details: Computes accuracy and saves metrics to runs/evaluation/. Target >90%.

## Predict Emotions:
Command (Single Image):powershellpython main.py --mode predict --image_path "C:\EDL\EDL_Proj2\deepface\DATASET - To Students\23BTRCL002 - VISWWAJITH PRADOSH K...\happy\img1.jpg"
Command (Folder):powershellpython main.py --mode predict --image_path "C:\EDL\EDL_Proj2\deepface\DATASET - To Students"
Details: Outputs predictions to runs/predictions/.

## Output Files

runs/train/: best_model.h5, final_model.h5 (trained models).
runs/evaluation/: metrics.txt (accuracy), confusion_matrix.csv.
runs/predictions/: predictions.txt (emotion predictions).
