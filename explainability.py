"""Grad-CAM utilities for classifier explainability."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Generate Grad-CAM heatmaps for image classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._capture_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _capture_gradient(self, _module: torch.nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image."""

    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 1.0 - alpha, colored, alpha, 0)
