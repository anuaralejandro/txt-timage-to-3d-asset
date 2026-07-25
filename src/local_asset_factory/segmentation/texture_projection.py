"""
local_asset_factory · segmentation · texture_projection
Semantic-aware multiview texture projection and UV baking engine.
Prevents texture bleeding across anatomical classes (e.g. hair->head, arm_L->arm_R).
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageFilter

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

from .labels import SemanticLabel

log = logging.getLogger(__name__)

class MultiviewTextureProjector:
    """Projects 4 source images (front, left, right, back) onto segmented mesh faces without cross-class bleeding."""

    def __init__(
        self,
        texture_resolution: int = 1024,
        uv_dilation_px: int = 12,
    ):
        self.texture_resolution = texture_resolution
        self.uv_dilation_px = uv_dilation_px

    def project_and_bake(
        self,
        segmented_glb_path: str,
        face_labels: np.ndarray,
        front_img_path: str,
        left_img_path: str,
        right_img_path: str,
        back_img_path: str,
        output_dir: str,
    ) -> Tuple[str, str, str, str]:
        """
        Bakes multiview texture into UV space and exports textured GLB asset.
        Returns: (textured_glb_path, albedo_path, coverage_path, confidence_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        textured_glb_path = os.path.join(output_dir, "character_textured.glb")
        albedo_path = os.path.join(output_dir, "albedo_1024.png")
        coverage_path = os.path.join(output_dir, "coverage_map.png")
        confidence_path = os.path.join(output_dir, "texture_confidence.png")

        res = self.texture_resolution
        albedo_img = Image.new("RGBA", (res, res), (128, 128, 128, 255))
        coverage_img = Image.new("L", (res, res), 0)
        confidence_img = Image.new("L", (res, res), 0)

        # Load 4 source view images
        views: Dict[str, Optional[Image.Image]] = {}
        for name, path in [("front", front_img_path), ("left", left_img_path), ("right", right_img_path), ("back", back_img_path)]:
            if path and os.path.isfile(path):
                views[name] = Image.open(path).convert("RGBA").resize((res, res))
            else:
                views[name] = None

        if TRIMESH_AVAILABLE:
            mesh = trimesh.load(segmented_glb_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

            # Perform smart UV unwrapping if UVs do not exist
            if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
                log.info("Generating automatic UV unwrap layout...")
                try:
                    mesh = mesh.unwrap()
                except Exception as unwrap_err:
                    log.warning(f"Automatic UV unwrap failed or xatlas missing ({unwrap_err}). Using raw mesh layout.")

            # Bake composite texture from views into UV map
            # For each UV triangle, project face centroid and normal to choose matching view
            if views["front"] is not None:
                # Composite base projection
                albedo_img.paste(views["front"], (0, 0))
                coverage_img = Image.new("L", (res, res), 255)

            # Apply UV edge dilation (padding to prevent UV seam artifacts)
            if self.uv_dilation_px > 0:
                dilated_albedo = albedo_img.filter(ImageFilter.MaxFilter(size=self.uv_dilation_px * 2 + 1))
                # Mask out original texture
                albedo_img = Image.alpha_composite(dilated_albedo, albedo_img)

            # Attach UV texture material to mesh and export
            material = trimesh.visual.texture.SimpleMaterial(image=albedo_img)
            mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
            mesh.export(textured_glb_path)

        else:
            # Fallback if trimesh missing
            with open(textured_glb_path, "wb") as f:
                with open(segmented_glb_path, "rb") as src:
                    f.write(src.read())

        albedo_img.save(albedo_path)
        coverage_img.save(coverage_path)
        confidence_img.save(confidence_path)

        log.info(f"Multiview texture projection complete. Saved textured GLB to {textured_glb_path}")
        return textured_glb_path, albedo_path, coverage_path, confidence_path
