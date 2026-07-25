"""
local_asset_factory · segmentation · fusion
Multi-view 2D->3D score accumulation engine per mesh face ID.
"""

from __future__ import annotations
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .labels import SemanticLabel
from .contracts import ViewRenderInfo
from .human_parser_backend import ParserResult
from .sam2_backend import SAM2RefinementResult
from .p3sam_backend import P3SAMResult

log = logging.getLogger(__name__)

class SemanticFusionEngine:
    """Accumulates multi-view 2D predictions onto 3D mesh face IDs."""

    def __init__(self, num_classes: int = 13):
        self.num_classes = num_classes  # Labels 0..12

    def fuse_views(
        self,
        face_count: int,
        views_info: List[ViewRenderInfo],
        parser_results: List[ParserResult],
        sam_results: Optional[List[Dict[SemanticLabel, SAM2RefinementResult]]] = None,
        p3sam_result: Optional[P3SAMResult] = None,
        pose_results: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Accumulates evidence for each face across all views.
        Returns:
            face_labels: np.ndarray shape (face_count,) dtype int32
            face_confidences: np.ndarray shape (face_count,) dtype float32
            confidence_summary: Dict[str, float]
        """
        # Helper for distance to segment squared
        def dist_to_segment_sq(px, py, ax, ay, bx, by):
            l2 = (bx - ax)**2 + (by - ay)**2
            if l2 == 0:
                return (px - ax)**2 + (py - ay)**2
            t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2
            t = np.clip(t, 0.0, 1.0)
            proj_x = ax + t * (bx - ax)
            proj_y = ay + t * (by - ay)
            return (px - proj_x)**2 + (py - proj_y)**2

        # Score matrix shape: (num_faces, num_classes)
        # Note: face IDs in renders are 1-based (0 is background), so face_id - 1 = face_index
        face_scores = np.zeros((face_count, self.num_classes), dtype=np.float32)
        face_obs_counts = np.zeros(face_count, dtype=np.int32)

        for v_idx, view_info in enumerate(views_info):
            if not os.path.isfile(view_info.face_id_path):
                log.warning(f"Face ID map missing for view {view_info.view_name}. Skipping.")
                continue

            face_id_map = np.load(view_info.face_id_path)  # (H, W)
            depth_map = np.load(view_info.depth_path) if os.path.isfile(view_info.depth_path) else None
            normals_map = np.load(view_info.normals_path) if os.path.isfile(view_info.normals_path) else None
            
            parser_res = parser_results[v_idx]
            parser_labels = parser_res.labels  # (H, W)
            parser_conf = parser_res.confidence  # (H, W)

            sam_dict = sam_results[v_idx] if sam_results and v_idx < len(sam_results) else None
            view_pose = pose_results[v_idx] if pose_results and v_idx < len(pose_results) else None

            # Get visible face IDs in this view
            valid_pixel_mask = (face_id_map > 0) & (face_id_map <= face_count)
            if not np.any(valid_pixel_mask):
                continue

            # Limb splitting logic using pose
            if view_pose and view_pose.keypoints:
                H, W = face_id_map.shape
                Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                norm_y = Y[valid_pixel_mask] / float(H)
                norm_x = X[valid_pixel_mask] / float(W)
                kps = view_pose.keypoints

                def split_limb(upper_lbl, lower_lbl, ja, jb, jc):
                    if ja in kps and jb in kps and jc in kps:
                        a, b, c = kps[ja], kps[jb], kps[jc]
                        # Apply to parser labels
                        mask = parser_labels[valid_pixel_mask] == upper_lbl
                        if np.any(mask):
                            px, py = norm_x[mask], norm_y[mask]
                            d_up = dist_to_segment_sq(px, py, a.x, a.y, b.x, b.y)
                            d_dn = dist_to_segment_sq(px, py, b.x, b.y, c.x, c.y)
                            # Update parser labels in-place
                            parser_labels[valid_pixel_mask] = np.where(
                                (parser_labels[valid_pixel_mask] == upper_lbl) & mask & (d_dn < d_up),
                                lower_lbl,
                                parser_labels[valid_pixel_mask]
                            )

                split_limb(SemanticLabel.ARM_UPPER_L, SemanticLabel.ARM_LOWER_L, "shoulder_L", "elbow_L", "wrist_L")
                split_limb(SemanticLabel.ARM_UPPER_R, SemanticLabel.ARM_LOWER_R, "shoulder_R", "elbow_R", "wrist_R")
                split_limb(SemanticLabel.LEG_UPPER_L, SemanticLabel.LEG_LOWER_L, "hip_L", "knee_L", "ankle_L")
                split_limb(SemanticLabel.LEG_UPPER_R, SemanticLabel.LEG_LOWER_R, "hip_R", "knee_R", "ankle_R")

            visible_face_ids = face_id_map[valid_pixel_mask]
            visible_face_indices = visible_face_ids - 1

            view_parser_labels = parser_labels[valid_pixel_mask]
            view_parser_conf = parser_conf[valid_pixel_mask]

            # Vectorized score accumulation
            for c in range(1, self.num_classes):
                label_mask = (view_parser_labels == c)
                if not np.any(label_mask):
                    continue

                target_face_indices = visible_face_indices[label_mask]
                confs = view_parser_conf[label_mask]

                # Weight factor calculation
                weights = confs
                if sam_dict and SemanticLabel(c) in sam_dict:
                    sam_ref = sam_dict[SemanticLabel(c)]
                    sam_mask_flat = sam_ref.mask[valid_pixel_mask][label_mask]
                    weights = weights * (0.5 + 0.5 * sam_mask_flat.astype(np.float32))

                # Accumulate score for this class and face
                np.add.at(face_scores[:, c], target_face_indices, weights)
                np.add.at(face_obs_counts, target_face_indices, 1)

        # Apply P3-SAM geometric prior if available
        if p3sam_result is not None:
            region_ids = p3sam_result.face_region_ids
            # Super-region majority smoothing boost
            for r_id in np.unique(region_ids):
                r_mask = (region_ids == r_id)
                if np.any(r_mask):
                    region_class_sums = face_scores[r_mask].sum(axis=0)
                    dominant_class = np.argmax(region_class_sums[1:]) + 1
                    face_scores[r_mask, dominant_class] += 0.25 * p3sam_result.confidence[r_mask]

        # Resolve final class per face via argmax
        face_labels = np.zeros(face_count, dtype=np.int32)
        face_confidences = np.zeros(face_count, dtype=np.float32)

        total_scores = face_scores.sum(axis=1)
        scored_faces_mask = total_scores > 0

        if np.any(scored_faces_mask):
            face_labels[scored_faces_mask] = np.argmax(face_scores[scored_faces_mask], axis=1)
            
            # Confidence = max_score / (total_score + eps)
            max_scores = np.max(face_scores[scored_faces_mask], axis=1)
            face_confidences[scored_faces_mask] = max_scores / (total_scores[scored_faces_mask] + 1e-6)

        summary = {
            "total_faces": float(face_count),
            "labeled_faces": float(np.sum(face_labels > 0)),
            "unassigned_faces": float(np.sum(face_labels == 0)),
            "mean_confidence": float(np.mean(face_confidences[scored_faces_mask])) if np.any(scored_faces_mask) else 0.0,
        }

        log.info(f"Fusion complete: {summary['labeled_faces']}/{face_count} faces assigned labels.")
        return face_labels, face_confidences, summary
