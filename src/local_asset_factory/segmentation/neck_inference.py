"""
local_asset_factory · segmentation · neck_inference
Geometric and pose-guided neck region inference engine.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

from .labels import SemanticLabel
from .contracts import PoseDetectionResult

log = logging.getLogger(__name__)

class NeckInferencer:
    """Infers neck region between head bottom and shoulder line."""

    def __init__(self, vertical_band_margin: float = 0.05):
        self.vertical_band_margin = vertical_band_margin

    def infer_neck(
        self,
        glb_path: str,
        face_labels: np.ndarray,
        pose_result: Optional[PoseDetectionResult] = None,
    ) -> np.ndarray:
        """
        Derives neck label (SemanticLabel.NECK = 2) on mesh faces.
        Returns: updated_face_labels (np.ndarray of shape (num_faces,))
        """
        if not TRIMESH_AVAILABLE:
            log.warning("Trimesh not available. Skipping geometric neck inference.")
            return face_labels

        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        num_faces = len(mesh.faces)
        if len(face_labels) != num_faces:
            face_labels = np.resize(face_labels, num_faces)

        updated = face_labels.copy()

        # Get centroids of faces
        centroids = mesh.triangles.mean(axis=1)  # (N, 3)

        head_indices = np.where(updated == SemanticLabel.HEAD)[0]
        torso_indices = np.where(updated == SemanticLabel.TORSO)[0]

        if not np.any(head_indices) or not np.any(torso_indices):
            log.warning("Missing head or torso faces. Neck inference using geometric proportions.")
            # Fallback based on mesh height proportions (top 20% to 25% height)
            y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
            h = y_max - y_min
            neck_y_bottom = y_max - 0.25 * h
            neck_y_top = y_max - 0.18 * h

            # Central width constraint
            x_center = centroids[:, 0].mean()
            x_span = np.ptp(centroids[:, 0])
            central_mask = np.abs(centroids[:, 0] - x_center) < (x_span * 0.15)

            neck_candidates = (centroids[:, 1] >= neck_y_bottom) & (centroids[:, 1] <= neck_y_top) & central_mask
            updated[neck_candidates & (updated == SemanticLabel.TORSO)] = SemanticLabel.NECK
            return updated

        # Calculate head bottom boundary Y and shoulder top boundary Y
        head_y_min = centroids[head_indices, 1].min()
        torso_y_max = centroids[torso_indices, 1].max()

        neck_y_top = head_y_min + (head_y_min * 0.02)
        neck_y_bottom = min(head_y_min - 0.05, torso_y_max - 0.05)

        # Central cylinder constraint
        head_x_center = centroids[head_indices, 0].mean()
        head_x_span = np.ptp(centroids[head_indices, 0])
        center_width = max(head_x_span * 0.45, 0.05)

        central_x_mask = np.abs(centroids[:, 0] - head_x_center) < center_width
        height_mask = (centroids[:, 1] >= neck_y_bottom) & (centroids[:, 1] <= neck_y_top)

        candidate_faces = height_mask & central_x_mask

        # Exclude hair, arms, and legs
        forbidden = (
            (updated == SemanticLabel.HAIR) |
            (updated == SemanticLabel.ARM_UPPER_L) | (updated == SemanticLabel.ARM_LOWER_L) |
            (updated == SemanticLabel.ARM_UPPER_R) | (updated == SemanticLabel.ARM_LOWER_R) |
            (updated == SemanticLabel.LEG_UPPER_L) | (updated == SemanticLabel.LEG_LOWER_L) |
            (updated == SemanticLabel.LEG_UPPER_R) | (updated == SemanticLabel.LEG_LOWER_R)
        )

        neck_faces = candidate_faces & ~forbidden
        updated[neck_faces] = SemanticLabel.NECK

        log.info(f"Inferred {np.sum(neck_faces)} neck faces between head Y={head_y_min:.2f} and shoulder line.")
        return updated
