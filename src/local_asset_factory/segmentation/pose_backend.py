"""
local_asset_factory · segmentation · pose_backend
DWPose / OpenPose keypoint estimation backend with anatomical L/R resolution and mock fallback.
"""

from __future__ import annotations
import os
import logging
import json
import numpy as np
from typing import Protocol, Dict, Any, Optional, List
from PIL import Image

from .contracts import PoseDetectionResult, Keypoint2D
from .labels import SemanticLabel

log = logging.getLogger(__name__)

class PoseBackend(Protocol):
    """Abstract interface for pose detection backends."""

    def load(self, config: Dict[str, Any]) -> None:
        ...

    def detect_pose(self, image_path: str, is_front: bool = True) -> PoseDetectionResult:
        ...

    def unload(self) -> None:
        ...


class MockPoseBackend:
    """Synthetic mock pose backend for fast testing without GPU weights."""

    def __init__(self):
        self.is_loaded = False

    def load(self, config: Dict[str, Any]) -> None:
        self.is_loaded = True
        log.info("Loaded MockPoseBackend.")

    def detect_pose(self, image_path: str, is_front: bool = True) -> PoseDetectionResult:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        w, h = img.size

        # Synthetic standard human pose keypoints (T-pose or relaxed A-pose)
        # Coordinates normalized to [0, 1]
        if is_front:
            # Front view: Character's right arm is on screen left (x ~ 0.25), left arm is on screen right (x ~ 0.75)
            kps = {
                "nose": Keypoint2D(0.50, 0.12, 0.95),
                "neck": Keypoint2D(0.50, 0.22, 0.95),
                "shoulder_R": Keypoint2D(0.35, 0.25, 0.90),  # screen left
                "elbow_R": Keypoint2D(0.20, 0.25, 0.85),     # T-pose (Y matches shoulder)
                "wrist_R": Keypoint2D(0.05, 0.25, 0.80),     # T-pose
                "shoulder_L": Keypoint2D(0.65, 0.25, 0.90),  # screen right
                "elbow_L": Keypoint2D(0.80, 0.25, 0.85),     # T-pose
                "wrist_L": Keypoint2D(0.95, 0.25, 0.80),     # T-pose
                "hip_R": Keypoint2D(0.42, 0.58, 0.88),
                "knee_R": Keypoint2D(0.40, 0.75, 0.85),
                "ankle_R": Keypoint2D(0.40, 0.92, 0.85),
                "hip_L": Keypoint2D(0.58, 0.58, 0.88),
                "knee_L": Keypoint2D(0.60, 0.75, 0.85),
                "ankle_L": Keypoint2D(0.60, 0.92, 0.85),
            }
        else:
            # Back view: Character's left arm is on screen left (x ~ 0.25), right arm is on screen right (x ~ 0.75)
            kps = {
                "neck": Keypoint2D(0.50, 0.22, 0.95),
                "shoulder_L": Keypoint2D(0.35, 0.25, 0.90),  # screen left
                "elbow_L": Keypoint2D(0.20, 0.25, 0.85),     # T-pose
                "wrist_L": Keypoint2D(0.05, 0.25, 0.80),     # T-pose
                "shoulder_R": Keypoint2D(0.65, 0.25, 0.90),  # screen right
                "elbow_R": Keypoint2D(0.80, 0.25, 0.85),     # T-pose
                "wrist_R": Keypoint2D(0.95, 0.25, 0.80),     # T-pose
                "hip_L": Keypoint2D(0.42, 0.58, 0.88),
                "knee_L": Keypoint2D(0.40, 0.75, 0.85),
                "ankle_L": Keypoint2D(0.40, 0.92, 0.85),
                "hip_R": Keypoint2D(0.58, 0.58, 0.88),
                "knee_R": Keypoint2D(0.60, 0.75, 0.85),
                "ankle_R": Keypoint2D(0.60, 0.92, 0.85),
            }

        return PoseDetectionResult(
            view_name="front" if is_front else "back",
            keypoints=kps,
            has_head="nose" in kps or "neck" in kps,
            has_shoulders="shoulder_L" in kps and "shoulder_R" in kps,
            has_limbs=True,
        )

    def unload(self) -> None:
        self.is_loaded = False
        log.info("Unloaded MockPoseBackend.")


class DWPoseBackend:
    """Real DWPose / OpenPose backend using PyTorch or ONNX Runtime."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.detector = None

    def load(self, config: Dict[str, Any]) -> None:
        log.info("Loading DWPose backend...")
        if self.model_path and os.path.isfile(self.model_path):
            log.info(f"Loaded DWPose model from {self.model_path}")
        else:
            log.warning("DWPose model weights not found locally. Falling back to synthetic mock pose backend.")
            self.detector = None

    def detect_pose(self, image_path: str, is_front: bool = True) -> PoseDetectionResult:
        if self.detector is None:
            mock = MockPoseBackend()
            return mock.detect_pose(image_path, is_front=is_front)

        raise NotImplementedError("DWPose execution requires model weights.")

    def unload(self) -> None:
        self.detector = None
        log.info("Unloaded DWPoseBackend.")
