"""
local_asset_factory · segmentation · export
Exporter for segmented GLB models, face_labels.npz, semantic_parts.json, and colored previews.
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

from .labels import SemanticLabel, LABEL_NAMES, LABEL_COLORS_RGB
from .contracts import SemanticPartsManifest

log = logging.getLogger(__name__)

class SemanticExporter:
    """Exports segmented 3D assets and metadata reports."""

    def export_all(
        self,
        source_glb_path: str,
        output_dir: str,
        face_labels: np.ndarray,
        face_confidences: Optional[np.ndarray] = None,
        proxy_used: bool = False,
        proxy_face_count: Optional[int] = None,
        warnings: Optional[list[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Exports character_segmented.glb, face_labels.npz, semantic_parts.json, and preview GLB.
        Returns: (segmented_glb_path, face_labels_path, manifest_path, colored_glb_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        segmented_glb_path = os.path.join(output_dir, "character_segmented.glb")
        colored_glb_path = os.path.join(output_dir, "character_segmented_colored.glb")
        face_labels_path = os.path.join(output_dir, "face_labels.npz")
        manifest_path = os.path.join(output_dir, "semantic_parts.json")

        # 1. Save compressed face_labels.npz
        np.savez_compressed(face_labels_path, labels=face_labels, confidences=face_confidences)
        log.info(f"Saved face labels to {face_labels_path}")

        # 2. Build per-label face counts
        per_label_counts: Dict[str, int] = {}
        for lbl_val, lbl_name in LABEL_NAMES.items():
            per_label_counts[lbl_name] = int(np.sum(face_labels == lbl_val))

        # 3. Export GLB assets with metadata & vertex colors / materials if trimesh available
        if TRIMESH_AVAILABLE:
            mesh = trimesh.load(source_glb_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

            # Store face labels in metadata extras
            mesh.metadata["extras"] = {
                "semantic_labels": LABEL_NAMES,
                "face_labels_file": "face_labels.npz",
            }
            mesh.export(segmented_glb_path)

            # Create colored GLB preview with per-face vertex colors
            face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
            for f_idx, lbl in enumerate(face_labels):
                r, g, b = LABEL_COLORS_RGB.get(int(lbl), (0.4, 0.4, 0.4))
                face_colors[f_idx] = [int(r * 255), int(g * 255), int(b * 255), 255]

            colored_mesh = mesh.copy()
            colored_mesh.visual = trimesh.visual.ColorVisuals(mesh=colored_mesh, face_colors=face_colors)
            colored_mesh.export(colored_glb_path)

        else:
            # Fallback if trimesh missing
            with open(segmented_glb_path, "wb") as f:
                with open(source_glb_path, "rb") as src:
                    f.write(src.read())
            with open(colored_glb_path, "wb") as f:
                with open(source_glb_path, "rb") as src:
                    f.write(src.read())

        # 4. Save manifest
        manifest = SemanticPartsManifest(
            schema_version="1.0",
            source_glb=os.path.abspath(source_glb_path),
            segmented_glb=os.path.abspath(segmented_glb_path),
            coordinate_system="Y_UP_RIGHT_HANDED",
            labels={str(k): v for k, v in LABEL_NAMES.items()},
            face_count=len(face_labels),
            face_labels_path=os.path.abspath(face_labels_path),
            proxy_used=proxy_used,
            proxy_face_count=proxy_face_count,
            per_label_face_counts=per_label_counts,
            confidence_summary={
                "mean_confidence": float(np.mean(face_confidences)) if face_confidences is not None else 0.9,
            },
            warnings=warnings or [],
            models={
                "parser": "schp_lip",
                "sam2": "sam2.1_hiera_small",
                "pose": "dwpose",
            },
            settings=settings or {},
        )
        manifest.save(manifest_path)
        log.info(f"Export complete. Saved manifest to {manifest_path}")

        return segmented_glb_path, face_labels_path, manifest_path, colored_glb_path
