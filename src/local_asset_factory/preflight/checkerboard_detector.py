"""
local_asset_factory · preflight · checkerboard_detector
Detects baked checkerboard backgrounds (fake transparency in PNG).

A checkerboard pattern horneado en RGB es la señal más común de que
una imagen fue exportada con canal alfa no real. Debe ser rechazada
antes de entrar al pipeline de Hunyuan.

Algorithm:
  1. Convert to grayscale
  2. Apply Sobel edge detection
  3. Analyze frequency of alternating light/dark blocks
  4. Check for regular grid pattern characteristic of a checkerboard
  5. Estimate what fraction of the image is checkerboard background
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

from PIL import Image
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CheckerboardResult:
    detected: bool
    confidence: float           # 0.0–1.0
    pattern_frequency_px: int   # Estimated tile size in pixels
    affected_fraction: float    # Fraction of image covered by pattern
    message: str


def detect_checkerboard(
    image: Image.Image,
    *,
    tile_size_range: Tuple[int, int] = (8, 64),
    confidence_threshold: float = 0.6,
    min_affected_fraction: float = 0.08,
) -> CheckerboardResult:
    """
    Detect a baked checkerboard background in an image.

    Args:
        image: PIL Image (RGB or RGBA)
        tile_size_range: (min, max) tile sizes to check in pixels
        confidence_threshold: minimum pattern regularity to trigger detection
        min_affected_fraction: minimum image fraction that must be checkerboard

    Returns:
        CheckerboardResult with detection outcome.
    """
    # Convert to numpy RGB (ignore alpha channel for pattern detection)
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # --- Step 1: Convert to luminance ---
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    best_confidence = 0.0
    best_tile_size = 0
    best_fraction = 0.0

    h, w = lum.shape
    min_tile, max_tile = tile_size_range

    # --- Step 2: Test candidate tile sizes ---
    for tile in range(min_tile, max_tile + 1, 4):
        confidence, fraction = _score_checkerboard_tile(lum, tile)
        if confidence > best_confidence:
            best_confidence = confidence
            best_tile_size = tile
            best_fraction = fraction

    detected = (
        best_confidence >= confidence_threshold
        and best_fraction >= min_affected_fraction
    )

    if detected:
        msg = (
            f"Baked checkerboard background detected "
            f"(confidence={best_confidence:.2f}, tile~{best_tile_size}px, "
            f"coverage={best_fraction:.1%}). "
            f"Image appears to have fake transparency. Pipeline rejected."
        )
    else:
        msg = (
            f"No checkerboard background detected "
            f"(max_confidence={best_confidence:.2f})"
        )

    log.debug("Checkerboard: %s", msg)
    return CheckerboardResult(
        detected=detected,
        confidence=best_confidence,
        pattern_frequency_px=best_tile_size,
        affected_fraction=best_fraction,
        message=msg,
    )


def _score_checkerboard_tile(
    luminance: np.ndarray, tile_size: int
) -> Tuple[float, float]:
    """
    Score how well the image matches a checkerboard with given tile size.
    Returns (confidence, fraction_covered).
    """
    h, w = luminance.shape
    if tile_size <= 0 or h < tile_size * 2 or w < tile_size * 2:
        return 0.0, 0.0

    # Build expected checkerboard mask (0 = dark, 1 = light)
    rows = np.arange(h) // tile_size
    cols = np.arange(w) // tile_size
    expected = ((rows[:, None] + cols[None, :]) % 2).astype(np.float32)

    # Normalize luminance to 0-1
    lum_min, lum_max = luminance.min(), luminance.max()
    if lum_max - lum_min < 10:
        return 0.0, 0.0   # flat image — no pattern
    lum_norm = (luminance - lum_min) / (lum_max - lum_min)

    # Threshold to binary
    lum_bin = (lum_norm > 0.5).astype(np.float32)

    # Agreement score: how many pixels agree with expected pattern
    agree = (lum_bin == expected).astype(np.float32)
    # Check inverted pattern too
    agree_inv = (lum_bin == (1 - expected)).astype(np.float32)
    agreement = max(agree.mean(), agree_inv.mean())

    # fraction: use edge-based local variance to estimate coverage
    # High-variance regions tend to be the subject, low-variance = background
    from scipy.ndimage import uniform_filter
    try:
        lum2 = uniform_filter(lum_norm ** 2, size=tile_size)
        lum1 = uniform_filter(lum_norm, size=tile_size)
        variance_map = np.clip(lum2 - lum1 ** 2, 0, None)
        # Background pixels: variance consistent with checkerboard pattern
        bg_mask = variance_map < 0.08
        fraction = float(bg_mask.mean())
    except ImportError:
        # scipy not available — estimate from agreement only
        fraction = max(0.0, (agreement - 0.5) * 2)

    return float(agreement), fraction
