"""
local_asset_factory · segmentation · skeleton_backend
Interface and backends for 3D skeleton / joint prediction (MagicArticulate, DWPose 3D projection, or geometry-based skeletonization).
Provides soft bone proximity priors to guide 3D face labeling.
"""

from __future__ import annotations
import os
import gc
import logging
import numpy as np
import trimesh
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .labels import SemanticLabel

log = logging.getLogger(__name__)


@dataclass
class SkeletonJoint:
    name: str
    pos_3d: np.ndarray  # (3,) xyz in mesh local space


class SkeletonBackend:
    """Abstract interface for 3D skeleton / joint prediction."""

    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def predict(self, mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        """Returns dictionary mapping joint names to (3,) 3D coordinates."""
        raise NotImplementedError

    def unload(self) -> None:
        pass


class GeometryJointPredictor(SkeletonBackend):
    """
    Robust 3D joint predictor based on mesh geometry density and bounding box proportions.
    Supports T-pose and A-pose humanoids.
    """

    def predict(self, mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        verts = mesh.vertices
        x_min, y_min, z_min = verts.min(axis=0)
        x_max, y_max, z_max = verts.max(axis=0)

        # Determine UP axis (usually Y or Z)
        extents = max(x_max - x_min, 1e-4), max(y_max - y_min, 1e-4), max(z_max - z_min, 1e-4)

        # In glTF / standard 3D asset convention, Y is UP
        if extents[1] >= extents[2]:
            # Y is UP
            H = extents[1]
            W = extents[0]
            center = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0

            joints = {
                "head": np.array([center[0], y_max - 0.10 * H, center[2]]),
                "neck": np.array([center[0], y_max - 0.22 * H, center[2]]),
                "chest": np.array([center[0], y_max - 0.35 * H, center[2]]),
                "pelvis": np.array([center[0], y_max - 0.48 * H, center[2]]),
                "shoulder_L": np.array([center[0] - 0.22 * W, y_max - 0.26 * H, center[2]]),
                "elbow_L": np.array([center[0] - 0.38 * W, y_max - 0.26 * H, center[2]]),
                "wrist_L": np.array([x_min + 0.05 * W, y_max - 0.26 * H, center[2]]),
                "shoulder_R": np.array([center[0] + 0.22 * W, y_max - 0.26 * H, center[2]]),
                "elbow_R": np.array([center[0] + 0.38 * W, y_max - 0.26 * H, center[2]]),
                "wrist_R": np.array([x_max - 0.05 * W, y_max - 0.26 * H, center[2]]),
                "hip_L": np.array([center[0] - 0.10 * W, y_max - 0.48 * H, center[2]]),
                "knee_L": np.array([center[0] - 0.10 * W, y_max - 0.72 * H, center[2]]),
                "ankle_L": np.array([center[0] - 0.10 * W, y_min + 0.05 * H, center[2]]),
                "hip_R": np.array([center[0] + 0.10 * W, y_max - 0.48 * H, center[2]]),
                "knee_R": np.array([center[0] + 0.10 * W, y_max - 0.72 * H, center[2]]),
                "ankle_R": np.array([center[0] + 0.10 * W, y_min + 0.05 * H, center[2]]),
            }
        else:
            # Z is UP (Blender native)
            H = extents[2]
            W = extents[0]
            center = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0

            joints = {
                "head": np.array([center[0], center[1], z_max - 0.10 * H]),
                "neck": np.array([center[0], center[1], z_max - 0.22 * H]),
                "chest": np.array([center[0], center[1], z_max - 0.35 * H]),
                "pelvis": np.array([center[0], center[1], z_max - 0.48 * H]),
                "shoulder_L": np.array([center[0] - 0.22 * W, center[1], z_max - 0.26 * H]),
                "elbow_L": np.array([center[0] - 0.38 * W, center[1], z_max - 0.26 * H]),
                "wrist_L": np.array([x_min + 0.05 * W, center[1], z_max - 0.26 * H]),
                "shoulder_R": np.array([center[0] + 0.22 * W, center[1], z_max - 0.26 * H]),
                "elbow_R": np.array([center[0] + 0.38 * W, center[1], z_max - 0.26 * H]),
                "wrist_R": np.array([x_max - 0.05 * W, center[1], z_max - 0.26 * H]),
                "hip_L": np.array([center[0] - 0.10 * W, center[1], z_max - 0.48 * H]),
                "knee_L": np.array([center[0] - 0.10 * W, center[1], z_max - 0.72 * H]),
                "ankle_L": np.array([center[0] - 0.10 * W, center[1], z_min + 0.05 * H]),
                "hip_R": np.array([center[0] + 0.10 * W, center[1], z_max - 0.48 * H]),
                "knee_R": np.array([center[0] + 0.10 * W, center[1], z_max - 0.72 * H]),
                "ankle_R": np.array([center[0] + 0.10 * W, center[1], z_min + 0.05 * H]),
            }

        return joints


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Computes Euclidean distance from 3D point p to line segment ab."""
    ab = b - a
    length_sq = np.dot(ab, ab)
    if length_sq < 1e-8:
        return float(np.linalg.norm(p - a))

    t = max(0.0, min(1.0, np.dot(p - a, ab) / length_sq))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def compute_bone_priors(mesh: trimesh.Trimesh, joints: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes (num_faces, 18) probabilistic priors based on 3D bone proximity.
    """
    centroids = mesh.triangles.mean(axis=1)
    num_faces = len(centroids)
    priors = np.full((num_faces, 18), 1e-4, dtype=np.float32)

    # Bone segment definitions: (joint_a, joint_b, target_semantic_label)
    bone_defs = [
        ("head", "neck", SemanticLabel.HEAD),
        ("neck", "chest", SemanticLabel.NECK),
        ("chest", "pelvis", SemanticLabel.TORSO),
        ("shoulder_L", "elbow_L", SemanticLabel.ARM_UPPER_L),
        ("elbow_L", "wrist_L", SemanticLabel.ARM_LOWER_L),
        ("shoulder_R", "elbow_R", SemanticLabel.ARM_UPPER_R),
        ("elbow_R", "wrist_R", SemanticLabel.ARM_LOWER_R),
        ("hip_L", "knee_L", SemanticLabel.LEG_UPPER_L),
        ("knee_L", "ankle_L", SemanticLabel.LEG_LOWER_L),
        ("hip_R", "knee_R", SemanticLabel.LEG_UPPER_R),
        ("knee_R", "ankle_R", SemanticLabel.LEG_LOWER_R),
    ]

    for j_a, j_b, label in bone_defs:
        if j_a in joints and j_b in joints:
            a, b = joints[j_a], joints[j_b]
            dists = np.array([point_to_segment_distance(c, a, b) for c in centroids])
            # Soft Gaussian bone proximity prior
            sigma = 0.15 * np.linalg.norm(b - a) if np.linalg.norm(b - a) > 0 else 0.1
            bone_prob = np.exp(-0.5 * (dists / (sigma + 1e-4)) ** 2)
            priors[:, label.value] += bone_prob

    # Normalize priors per face
    sum_priors = priors.sum(axis=1, keepdims=True)
    priors = priors / np.maximum(sum_priors, 1e-6)
    return priors
