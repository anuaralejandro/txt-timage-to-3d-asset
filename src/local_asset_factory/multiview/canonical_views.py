"""
local_asset_factory · multiview · canonical_views
Canonical view normalization for Hunyuan3D-2mv input.
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
from .image_alignment import clean_hidden_rgb, calculate_joint_registration, apply_joint_registration

log = logging.getLogger(__name__)

CAMERA_YAW: Dict[str, float] = {
    "front": 0.0,
    "left": 90.0,
    "back": 180.0,
    "right": 270.0,
}

TARGET_SIZE = (1024, 1024)
PAD_FRACTION = 0.05

def build_canonical_view_set(
    views: Dict[str, str],
    output_dir: str | Path,
    job_id: str,
    *,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_fraction: float = PAD_FRACTION,
) -> CanonicalViewSet:
    """
    Build a CanonicalViewSet using joint registration.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical: Dict[str, CanonicalView] = {}
    
    loaded_images = {}
    for orientation, path in views.items():
        if Path(path).exists():
            img = Image.open(path).convert("RGBA")
            loaded_images[orientation] = clean_hidden_rgb(img)
            
    if not loaded_images:
        raise ValueError("No valid images found for canonical views")

    # Calculate joint registration
    registration = calculate_joint_registration(loaded_images, pad_fraction=pad_fraction)
    
    alignment_report = {}

    for orientation, img in loaded_images.items():
        reg = registration.get(orientation)
        warnings = []
        if reg is None or not reg["valid"]:
            warnings.append("invalid_registration_fallback")
            # Fallback to independent if completely failed
            w, h = img.size
            side = max(w, h)
            reg = {
                "crop_box": (0, 0, side, side),
                "padded_dim": side,
                "t_pose_valid": False,
                "span_ratio": 0
            }
        
        aligned_img = apply_joint_registration(img, reg, target_size)
        out_path = out_dir / f"{orientation}.png"
        aligned_img.save(str(out_path), "PNG")
        
        sha = sha256_file(str(out_path))
        
        alignment_report[orientation] = {
            "t_pose_valid": reg["t_pose_valid"],
            "span_ratio": reg["span_ratio"],
            "crop_box": reg["crop_box"]
        }
        
        view = CanonicalView(
            orientation=ViewOrientation(orientation),
            relative_path=str(out_path.relative_to(out_dir.parent)),
            width=target_size[0],
            height=target_size[1],
            subject_bbox=None, # Already centered and padded
            camera_type=CameraType.ORTHOGRAPHIC,
            camera_yaw_deg=CAMERA_YAW.get(orientation, 0.0),
            camera_pitch_deg=0.0,
            producer="joint_normalizer",
            sha256=sha,
            for_checkpoint=orientation in ("front", "left", "back"),
            for_qa=True,
            metadata={"warnings": warnings, "registration": alignment_report[orientation]},
        )
        canonical[orientation] = view

    # Save meta and alignment report
    meta_path = out_dir / "meta.json"
    meta = {
        "job_id": job_id,
        "target_size": list(target_size),
        "views": {
            k: {
                "path": v.relative_path,
                "sha256": v.sha256,
                "camera_yaw_deg": v.camera_yaw_deg,
            }
            for k, v in canonical.items()
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    report_path = out_dir / "alignment_report.json"
    report_path.write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    vs = CanonicalViewSet(
        front=canonical.get("front"),
        left=canonical.get("left"),
        back=canonical.get("back"),
        right=canonical.get("right"),
    )
    return vs
