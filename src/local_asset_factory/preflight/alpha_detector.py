"""
local_asset_factory · preflight · alpha_detector
Detects images that have an alpha channel but with no true transparency.

A common failure mode: character images exported as RGBA where every
pixel has alpha=255 (fully opaque). The checkerboard pattern is baked
into RGB, so there is no actual transparency.

Also detects: partially transparent images where alpha is insufficient
to cleanly segment the subject.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import numpy as np

log = logging.getLogger(__name__)

# Thresholds
TRUE_ALPHA_THRESHOLD = 128          # pixels below this alpha are "transparent"
MIN_TRANSPARENT_FRACTION = 0.02     # at least 2% of pixels must be transparent


@dataclass
class AlphaResult:
    has_alpha_channel: bool
    true_alpha: bool                # alpha channel encodes real transparency
    fully_opaque: bool              # all alpha == 255
    transparent_fraction: float     # fraction of pixels with alpha < threshold
    min_alpha: int
    max_alpha: int
    mean_alpha: float
    message: str


def detect_alpha(image: Image.Image) -> AlphaResult:
    """
    Analyze the alpha channel of an image.

    Returns:
        AlphaResult describing whether the image has real transparency.
    """
    has_alpha = image.mode in ("RGBA", "LA", "PA")

    if not has_alpha:
        return AlphaResult(
            has_alpha_channel=False,
            true_alpha=False,
            fully_opaque=True,
            transparent_fraction=0.0,
            min_alpha=255,
            max_alpha=255,
            mean_alpha=255.0,
            message="Image has no alpha channel (RGB or grayscale).",
        )

    # Extract alpha
    if image.mode == "RGBA":
        alpha_arr = np.array(image)[:, :, 3]
    elif image.mode == "LA":
        alpha_arr = np.array(image)[:, :, 1]
    elif image.mode == "PA":
        # Convert via RGBA
        alpha_arr = np.array(image.convert("RGBA"))[:, :, 3]
    else:
        alpha_arr = np.full((image.height, image.width), 255, dtype=np.uint8)

    min_alpha = int(alpha_arr.min())
    max_alpha = int(alpha_arr.max())
    mean_alpha = float(alpha_arr.mean())
    fully_opaque = min_alpha >= 255
    transparent_fraction = float((alpha_arr < TRUE_ALPHA_THRESHOLD).mean())
    true_alpha = (not fully_opaque) and (transparent_fraction >= MIN_TRANSPARENT_FRACTION)

    if fully_opaque:
        msg = (
            "Image has RGBA channel but ALL pixels are fully opaque (alpha=255). "
            "This is NOT true transparency. Background must be removed manually or "
            "via SAM 3.1 before reconstruction."
        )
    elif not true_alpha:
        msg = (
            f"Image has alpha channel but only {transparent_fraction:.1%} of pixels "
            f"are below threshold {TRUE_ALPHA_THRESHOLD}. Alpha is present but may "
            f"not cleanly separate the subject from background."
        )
    else:
        msg = (
            f"Image has real transparency ({transparent_fraction:.1%} transparent pixels, "
            f"alpha range {min_alpha}–{max_alpha})."
        )

    log.debug("AlphaDetector: %s", msg)
    return AlphaResult(
        has_alpha_channel=True,
        true_alpha=true_alpha,
        fully_opaque=fully_opaque,
        transparent_fraction=transparent_fraction,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        mean_alpha=mean_alpha,
        message=msg,
    )
