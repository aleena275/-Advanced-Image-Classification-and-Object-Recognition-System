"""
utils.py
--------
Helper functions shared across the backend: image preprocessing,
top-k prediction formatting, and base64 image encoding for the API
response (used to send the Grad-CAM overlay back to the frontend).
"""

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load an uploaded file's bytes into a PIL Image (RGB)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalized (1, C, H, W) tensor."""
    tensor = preprocess(img).unsqueeze(0)
    return tensor


def pil_to_bgr_ndarray(img: Image.Image, size=(224, 224)) -> np.ndarray:
    """Convert a PIL RGB image into a resized BGR numpy array for OpenCV."""
    img_resized = img.resize(size)
    arr = np.array(img_resized)[:, :, ::-1]  # RGB -> BGR
    return arr.astype(np.uint8)


def ndarray_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image as a base64 PNG string for JSON responses."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")


def get_topk_predictions(logits: torch.Tensor, class_names: list, k: int = 5):
    """
    logits: raw model output, shape (1, num_classes)
    Returns a list of dicts: [{"label": ..., "confidence": ...}, ...]
    sorted by confidence, descending.
    """
    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k=min(k, probs.shape[0]))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        results.append({
            "label": class_names[idx] if idx < len(class_names) else str(idx),
            "confidence": round(prob * 100, 2),
        })
    return results
