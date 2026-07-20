"""
app.py
------
Flask REST API for the Image Classification & Object Recognition system.

Endpoints:
  GET  /api/health            -> health check
  GET  /api/models             -> list available models
  POST /api/predict            -> classify an uploaded image, returns
                                   top-k predictions + Grad-CAM overlay

By default this loads a pretrained ResNet18 (ImageNet, 1000 classes) so
the API works out-of-the-box with no training required. If a checkpoint
trained via train.py (CIFAR-10, 10 classes) is present at
backend/checkpoints/best_model.pt, set MODEL_MODE=custom to use it instead.
"""

import os
import time

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision.models import ResNet18_Weights, resnet18

from gradcam import GradCAM, overlay_heatmap
from model import build_model
from utils import (
    get_topk_predictions,
    load_image_from_bytes,
    ndarray_to_base64,
    pil_to_bgr_ndarray,
    preprocess_image,
)

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately) to call this API

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MODE = os.environ.get("MODEL_MODE", "pretrained")  # "pretrained" | "custom"
CUSTOM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_model():
    if MODEL_MODE == "custom" and os.path.exists(CUSTOM_CHECKPOINT):
        model = build_model("scratch", num_classes=10)
        model.load_state_dict(torch.load(CUSTOM_CHECKPOINT, map_location=DEVICE))
        model.to(DEVICE).eval()
        target_layer = model.get_last_conv_layer()
        class_names = CIFAR10_CLASSES
        print("Loaded custom CIFAR-10 checkpoint.")
        return model, target_layer, class_names

    # Default: pretrained ImageNet ResNet18 (works with no training step)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(DEVICE).eval()
    target_layer = model.layer4[-1]
    class_names = weights.meta["categories"]
    print("Loaded pretrained ImageNet ResNet18.")
    return model, target_layer, class_names


MODEL, TARGET_LAYER, CLASS_NAMES = load_model()
CAM = GradCAM(MODEL, TARGET_LAYER)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "mode": MODEL_MODE})


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({
        "active_mode": MODEL_MODE,
        "num_classes": len(CLASS_NAMES),
        "available_modes": ["pretrained", "custom"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        start = time.time()
        image_bytes = file.read()
        pil_img = load_image_from_bytes(image_bytes)

        input_tensor = preprocess_image(pil_img).to(DEVICE)
        input_tensor.requires_grad_(True)

        # Top-k predictions
        with torch.no_grad():
            logits = MODEL(input_tensor)
        topk = get_topk_predictions(logits, CLASS_NAMES, k=5)

        # Grad-CAM (needs a fresh forward/backward pass with grad enabled)
        cam, class_idx = CAM.generate(input_tensor)
        original_bgr = pil_to_bgr_ndarray(pil_img)
        overlay = overlay_heatmap(original_bgr, cam)
        overlay_b64 = ndarray_to_base64(overlay)

        inference_ms = round((time.time() - start) * 1000, 1)

        return jsonify({
            "predictions": topk,
            "predicted_class": CLASS_NAMES[class_idx],
            "gradcam_overlay": f"data:image/png;base64,{overlay_b64}",
            "inference_time_ms": inference_ms,
        })

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
