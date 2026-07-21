"""
ComfyUI-LocalAssetFactory · Image Conversion Utilities
Conversions between PIL, NumPy (ComfyUI tensor format), and file paths.
"""

from __future__ import annotations

import os
from typing import Optional, Any

import numpy as np
from PIL import Image

try:
    from .logging_utils import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

log = get_logger(__name__)


import torch

def pil_to_comfy_tensor(image: Image.Image) -> "torch.Tensor":
    """Convert a PIL Image to the ComfyUI tensor format.

    ComfyUI uses ``[B, H, W, C]`` float32 tensors in ``[0, 1]``.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension → (1, H, W, 3)
    return torch.from_numpy(arr[np.newaxis, ...])


def comfy_tensor_to_pil(tensor: Any) -> Image.Image:
    """Convert a ComfyUI ``[B, H, W, C]`` tensor to a PIL Image.

    Only the first image in the batch is returned.
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]  # take first in batch
    arr = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def load_image_as_comfy(path: str) -> "np.ndarray":
    """Load an image from disk and return as ComfyUI tensor."""
    img = Image.open(path)
    return pil_to_comfy_tensor(img)


def save_comfy_image(tensor: "np.ndarray", path: str) -> str:
    """Save a ComfyUI tensor to disk as PNG.  Returns the saved path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = comfy_tensor_to_pil(tensor)
    img.save(path, "PNG")
    log.info("Saved image: %s", path)
    return path


def save_pil_image(image: Image.Image, path: str) -> str:
    """Save a PIL image to disk.  Returns the saved path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    log.info("Saved image: %s", path)
    return path
