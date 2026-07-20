"""
gradcam.py
----------
Model interpretation utilities (Phase 9 of the project workflow):
  - Grad-CAM heatmap generation
  - Overlaying the heatmap on the original image

Works with either SimpleCNN (from scratch) or TransferModel (resnet18).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: shape (1, C, H, W), already normalized
        class_idx: target class index. If None, uses the predicted class.
        Returns: heatmap as a numpy array in range [0, 1], shape (H, W)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx


def overlay_heatmap(original_img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """
    original_img_bgr: original image as a numpy array (H, W, 3), BGR, uint8
    cam: heatmap in range [0, 1], shape (H, W)
    Returns: overlaid image as numpy array (H, W, 3), BGR, uint8
    """
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay
