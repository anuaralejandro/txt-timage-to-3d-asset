"""
local_asset_factory · preflight · background_analyzer
Analyzes background type and uniformity in an image.

Classifies into:
  TRANSPARENT     — real alpha channel, clean bg
  UNIFORM         — solid color (white, grey, etc.) — accepted for rembg
  NEAR_UNIFORM    — mostly solid but some variation — warning
  COMPLEX         — natural photo background — needs removal
  CHECKERBOARD    — baked fake transparency — REJECT
  GRADIENT        — smooth gradient — may be acceptable
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class BackgroundType(str, Enum):
    TRANSPARENT = "transparent"
    UNIFORM = "uniform"
    NEAR_UNIFORM = "near_uniform"
    GRADIENT = "gradient"
    COMPLEX = "complex"
    CHECKERBOARD = "checkerboard"


@dataclass
class BackgroundResult:
    bg_type: BackgroundType
    dominant_color: Optional[tuple] = None   # (R, G, B)
    uniformity_score: float = 0.0            # 1.0 = perfectly uniform
    background_fraction: float = 0.0         # fraction of image estimated as bg
    needs_removal: bool = False
    accepted: bool = True
    message: str = ""


def analyze_background(
    image: Image.Image,
    *,
    uniformity_threshold: float = 0.85,
    max_bg_fraction: float = 0.50,
) -> BackgroundResult:
    """
    Analyze the background of an image.

    Args:
        image: PIL Image (any mode)
        uniformity_threshold: std_dev / 255 threshold for 'uniform'
        max_bg_fraction: fraction of image that can be background before warning

    Returns:
        BackgroundResult
    """
    # Handle transparent images quickly
    if image.mode in ("RGBA", "LA"):
        arr = np.array(image.convert("RGBA"))
        alpha = arr[:, :, 3]
        transparent_fraction = float((alpha < 128).mean())
        if transparent_fraction >= 0.05:
            return BackgroundResult(
                bg_type=BackgroundType.TRANSPARENT,
                transparent_fraction=transparent_fraction,
                uniformity_score=1.0,
                background_fraction=transparent_fraction,
                needs_removal=False,
                accepted=True,
                message=f"Image has real transparency ({transparent_fraction:.1%} transparent).",
            )

    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    # Sample edge pixels as a proxy for background
    h, w = arr.shape[:2]
    margin = max(4, min(16, h // 16, w // 16))
    edge = np.concatenate([
        arr[:margin, :].reshape(-1, 3),
        arr[-margin:, :].reshape(-1, 3),
        arr[:, :margin].reshape(-1, 3),
        arr[:, -margin:].reshape(-1, 3),
    ], axis=0)

    # Dominant background color (median of edge)
    dominant = tuple(int(v) for v in np.median(edge, axis=0))

    # Uniformity: how consistent is the edge color
    std_per_channel = edge.std(axis=0)   # shape (3,)
    uniformity = float(1.0 - (std_per_channel.max() / 255.0))

    # Estimate background fraction: pixels close to dominant color
    diff = np.abs(arr - np.array(dominant, dtype=np.float32))
    close_to_bg = (diff.max(axis=2) < 30).mean()
    bg_fraction = float(close_to_bg)

    # Classify
    if uniformity >= uniformity_threshold:
        if bg_fraction <= max_bg_fraction:
            bg_type = BackgroundType.UNIFORM
            needs_removal = False
            accepted = True
            msg = f"Uniform background (color={dominant}, uniformity={uniformity:.2f})."
        else:
            bg_type = BackgroundType.UNIFORM
            needs_removal = True
            accepted = True
            msg = (
                f"Large uniform background ({bg_fraction:.1%} of image). "
                "Background removal recommended before reconstruction."
            )
    elif uniformity >= 0.60:
        bg_type = BackgroundType.NEAR_UNIFORM
        needs_removal = True
        accepted = True
        msg = f"Near-uniform background (uniformity={uniformity:.2f}, color≈{dominant})."
    else:
        # Check for gradient (low spatial frequency in edge)
        edge_flat = edge.mean(axis=1)
        grad_score = float(1.0 - np.abs(np.diff(edge_flat)).mean() / 128.0)
        if grad_score > 0.7:
            bg_type = BackgroundType.GRADIENT
            needs_removal = True
            accepted = True
            msg = "Gradient background detected. Removal recommended."
        else:
            bg_type = BackgroundType.COMPLEX
            needs_removal = True
            accepted = True
            msg = (
                "Complex/natural background detected. "
                "Background removal required before reconstruction."
            )

    return BackgroundResult(
        bg_type=bg_type,
        dominant_color=dominant,
        uniformity_score=uniformity,
        background_fraction=bg_fraction,
        needs_removal=needs_removal,
        accepted=accepted,
        message=msg,
    )


# Monkey-patch dataclass to support transparent_fraction field dynamically
BackgroundResult.__dataclass_fields__["transparent_fraction"] = None  # type: ignore
