"""
local_asset_factory · multiview · canonical_views
Canonical view normalization for Hunyuan3D-2mv input.

Produces a CanonicalViewSet from raw input images:
  1. Detect and crop subject bounding box
  2. Apply symmetric padding (subject centered)
  3. Resize to target (1024×1024)
  4. Ensure RGBA with real or cleaned background
  5. Register camera metadata per view
  6. Record SHA-256 hash per view

Output structure:
  canonical_views/
    front.png   — checkpoint input + QA
    left.png    — checkpoint input + QA
    back.png    — checkpoint input + QA
    right.png   — QA only (not fed to Hunyuan3D-2mv checkpoint)
    meta.json   — camera params, keypoints, bboxes
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..domain.enums import CameraType, ViewOrientation
from ..domain.models import BoundingBox2D, CanonicalView, CanonicalViewSet, sha256_file
from .image_alignment import clean_hidden_rgb, calculate_joint_registration

log = logging.getLogger(__name__)

# Default camera yaw angles per orientation (degrees, Y-up, Z-forward)
CAMERA_YAW: Dict[str, float] = {
    "front": 0.0,
    "left": 90.0,
    "back": 180.0,
    "right": 270.0,
}

TARGET_SIZE = (1024, 1024)
PAD_FRACTION = 0.05
BG_COLOR = (255, 255, 255, 0)   # fully transparent


@dataclass
class NormalizeResult:
    orientation: ViewOrientation
    output_path: str
    width: int
    height: int
    subject_bbox: Optional[BoundingBox2D] = None
    sha256: str = ""
    warnings: list = field(default_factory=list)


def _detect_subject_bbox(img_rgba: Image.Image) -> Optional[BoundingBox2D]:
    """
    Detect subject bounding box from alpha channel.
    Returns None if no alpha or subject is the full image.
    """
    if img_rgba.mode != "RGBA":
        return None
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return BoundingBox2D(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0))


def normalize_view(
    input_path: str | Path,
    orientation: str,
    output_dir: str | Path,
    *,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_fraction: float = PAD_FRACTION,
    producer: str = "normalizer",
) -> NormalizeResult:
    """
    Normalize a single input view to canonical format.

    Steps:
      1. Load + convert to RGBA
      2. Detect subject via alpha (or use full image if no alpha)
      3. Crop + symmetric pad
      4. Resize to target_size
      5. Save as PNG

    Args:
        input_path: source image
        orientation: "front" | "left" | "back" | "right"
        output_dir: directory for output PNG
        target_size: (W, H) target resolution
        pad_fraction: padding fraction around subject
        producer: metadata field

    Returns:
        NormalizeResult with path, dimensions, bbox, hash
    """
    src = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{orientation}.png"
    warnings = []

    img = Image.open(src).convert("RGBA")
    img = clean_hidden_rgb(img)
    orig_w, orig_h = img.size

    # Detect subject
    bbox = _detect_subject_bbox(img)
    if bbox is None:
        log.warning("No alpha-based subject detected in %s — using full image", src.name)
        bbox = BoundingBox2D(x=0, y=0, w=float(orig_w), h=float(orig_h))
        warnings.append("no_alpha_subject_detection")

    # Crop to subject + symmetric padding
    sx, sy = int(bbox.x), int(bbox.y)
    sw, sh = int(bbox.w), int(bbox.h)

    # Symmetric pad (equal on all sides)
    pad_x = int(sw * pad_fraction)
    pad_y = int(sh * pad_fraction)

    x0 = max(0, sx - pad_x)
    y0 = max(0, sy - pad_y)
    x1 = min(orig_w, sx + sw + pad_x)
    y1 = min(orig_h, sy + sh + pad_y)

    cropped = img.crop((x0, y0, x1, y1))

    # Make square canvas (center subject)
    cw, ch = cropped.size
    side = max(cw, ch)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    paste_x = (side - cw) // 2
    paste_y = (side - ch) // 2
    square.paste(cropped, (paste_x, paste_y))

    # Resize to target
    resized = square.resize(target_size, Image.LANCZOS)
    resized.save(str(out_path), "PNG")

    # Hash
    sha = sha256_file(out_path)

    log.info(
        "Normalized %s: %dx%d → %dx%d (bbox=%s)",
        orientation, orig_w, orig_h, target_size[0], target_size[1], bbox
    )

    return NormalizeResult(
        orientation=ViewOrientation(orientation),
        output_path=str(out_path),
        width=target_size[0],
        height=target_size[1],
        subject_bbox=bbox,
        sha256=sha,
        warnings=warnings,
    )


def build_canonical_view_set(
    views: Dict[str, str],
    output_dir: str | Path,
    job_id: str,
    *,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_fraction: float = PAD_FRACTION,
) -> CanonicalViewSet:
    """
    Build a CanonicalViewSet from a dict of {orientation: image_path}.

    Required: front, left, back
    Optional: right (QA-only, not fed to Hunyuan3D-2mv checkpoint)

    Returns CanonicalViewSet ready for pipeline use.
    """
    out_dir = Path(output_dir)
    canonical: Dict[str, CanonicalView] = {}

    for orientation in ("front", "left", "back", "right"):
        if orientation not in views:
            if orientation in ("front", "left", "back"):
                raise ValueError(f"Missing required view: {orientation}")
            continue

        result = normalize_view(
            input_path=views[orientation],
            orientation=orientation,
            output_dir=out_dir,
            target_size=target_size,
            pad_fraction=pad_fraction,
        )

        view = CanonicalView(
            orientation=ViewOrientation(orientation),
            relative_path=str(Path(result.output_path).relative_to(out_dir.parent)),
            width=result.width,
            height=result.height,
            subject_bbox=result.subject_bbox,
            camera_type=CameraType.ORTHOGRAPHIC,
            camera_yaw_deg=CAMERA_YAW.get(orientation, 0.0),
            camera_pitch_deg=0.0,
            producer="normalizer",
            sha256=result.sha256,
            for_checkpoint=orientation in ("front", "left", "back"),
            for_qa=True,
            metadata={"warnings": result.warnings},
        )
        canonical[orientation] = view

    # Save meta.json
    meta_path = out_dir / "meta.json"
    meta = {
        "job_id": job_id,
        "target_size": list(target_size),
        "views": {
            k: {
                "path": v.relative_path,
                "sha256": v.sha256,
                "camera_yaw_deg": v.camera_yaw_deg,
                "for_checkpoint": v.for_checkpoint,
            }
            for k, v in canonical.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vs = CanonicalViewSet(
        front=canonical["front"],
        left=canonical["left"],
        back=canonical["back"],
        right=canonical.get("right"),
    )
    log.info(
        "CanonicalViewSet built: %d views, checkpoint_views=%s",
        len(canonical), list(vs.checkpoint_views().keys())
    )
    return vs
