- Thanish Chinappa K.C.
- Likhith
- Saumitra Purkayastha
- Sundareshwar S
- Tenzin Kunga

See the [LICENSE](LICENSE) file for full contributor and copyright details.

---

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

## Web Interface

A React + Vite frontend is available in the `frontend/` directory.

### Setup & Development

1) Navigate to the frontend directory:

```bash
cd frontend
```

2) Install dependencies (using npm or pnpm):

```bash
npm install
# or
pnpm install
```

3) Run the development server:

```bash
npm run dev
# or
pnpm dev
```

4) Build for production:

```bash
npm run build
# or
pnpm build
```

### Running with Backend

To serve the built frontend with the FastAPI backend:

1) Build the frontend (from the `frontend/` directory):

```bash
npm run build
```

2) Install web server dependencies (from the project root):

```bash
pip install fastapi uvicorn python-multipart
```

3) Start the web server (from the project root):

```bash
python main.py --serve
# Or specify a custom port:
python main.py --serve --port 8080
```

The server will:
- Serve the frontend at `http://localhost:8000`
- Provide API endpoints at `http://localhost:8000/api/`
- Show API documentation at `http://localhost:8000/docs`

## License

See `LICENSE` for details.