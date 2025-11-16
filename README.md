# Facial Expression Recognition (Group Project)

Collaborative project exploring multiple models for facial expression recognition. The YOLO-based implementation is organized under `models/yolo_model/`. Add your own models under `models/` without affecting others.

## Repository Structure

- `models/yolo_model/`: Canonical YOLOv8 pipeline (code, notebook, weights placeholder, runs)
- `dataset/`: Shared dataset root (expects YOLO format; see `dataset/data.yaml`)
- `LICENSE`: Project license
- `README.md`: This file

## Quick Start (YOLO model)

1) Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

2) Install dependencies for the YOLO model:

```bash
cd models/yolo_model
pip install -r requirements.txt
```

3) Prepare data (ensure `dataset/` exists with images + labels defined in `dataset/data.yaml`). Then run:

```bash
# Full pipeline
python main.py

# Or individual steps
python src/prepare_dataset.py
python src/train_model.py
python src/evaluate.py
python src/predict.py
```

Notes:
- By default, configs expect the shared dataset at `../../dataset/` from the YOLO folder.
- Large artifacts (weights, runs, caches) are ignored via `.gitignore`.

## Adding Another Model

1) Create a folder under `models/` (e.g., `models/cnn_model/`).
2) Keep your code, configs, and requirements inside your folder.
3) Reference the shared dataset with a relative path (e.g., `../../dataset/`).
4) Document how to train/evaluate in a `README.md` inside your model folder.

## Dataset Expectations

YOLO format is expected:

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

Classes: angry, disgust, fear, happy, neutral, sad, surprised.

## GPU/Device

Ultralytics will auto-detect CUDA/MPS if available. To force CPU/GPU, set the `device` field in the YOLO `config.yaml` (e.g., `device: 0` for first GPU, `device: cpu` for CPU).

## License

See `LICENSE` for details.