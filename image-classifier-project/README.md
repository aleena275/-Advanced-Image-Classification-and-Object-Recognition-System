# Advanced Image Classification and Object Recognition System

CNN-based image classification project with a separate **backend** (Python/Flask + PyTorch)
and **frontend** (HTML/CSS/JS).

## Folder structure

```
image-classifier-project/
├── backend/
│   ├── app.py            # Flask API (predict + Grad-CAM endpoints)
│   ├── model.py           # SimpleCNN (from scratch) + TransferModel (ResNet/VGG/EfficientNet)
│   ├── train.py            # Training script for CIFAR-10 (from scratch or transfer learning)
│   ├── gradcam.py          # Grad-CAM model interpretation
│   ├── utils.py            # Image preprocessing & helper functions
│   └── requirements.txt
└── frontend/
    ├── index.html          # Upload UI
    ├── style.css
    └── script.js           # Calls the Flask API and renders results
```

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API starts at `http://localhost:5000`. By default it loads a **pretrained ResNet18**
(ImageNet, 1000 classes) so it works immediately with no training required.

### Endpoints
- `GET  /api/health` – health check
- `GET  /api/models` – active model info
- `POST /api/predict` – form field `image` → returns top-5 predictions, predicted
  class, Grad-CAM heatmap overlay (base64 PNG), and inference time

### (Optional) Train your own CIFAR-10 model

```bash
cd backend
python train.py --model scratch --epochs 20          # CNN from scratch
python train.py --model transfer --backbone resnet18 --epochs 10   # transfer learning
```

This saves `best_model.pt`. Move it to `backend/checkpoints/best_model.pt` and start
the server with:

```bash
MODEL_MODE=custom python app.py
```

## 2. Frontend setup

The frontend is plain HTML/CSS/JS — no build step needed. Just serve the folder:

```bash
cd frontend
python -m http.server 8000
```

Open `http://localhost:8000` in your browser. It calls the backend at
`http://localhost:5000` (configurable in `script.js` via `API_BASE_URL`).

## Features implemented

- CNN from scratch (LeNet/AlexNet-style) with batch norm & dropout
- Transfer learning wrapper (ResNet18 / VGG16 / EfficientNet-B0) with
  feature-extraction and fine-tuning modes
- Data augmentation pipeline (crop, flip, color jitter)
- Training strategies: LR scheduling, early stopping, checkpointing, gradient clipping
- Evaluation: accuracy, classification report, confusion matrix
- Model interpretation: Grad-CAM heatmap generation & overlay
- Web app: drag-and-drop image upload, top-5 predictions with confidence,
  Grad-CAM visualization, inference time display

## Notes

- Swap `--backbone` in `train.py` to try VGG16 or EfficientNet-B0.
- For production deployment, consider converting the model to TensorFlow Lite /
  TorchScript and serving behind a WSGI server (gunicorn) instead of Flask's dev server.
