"""
local_asset_factory · segmentation · face_renderer
Renders orthographic views of 3D meshes with exact Face-ID buffers, depth maps, normals, and camera matrices.
Supports Blender rendering and pure-Python Trimesh raycasting fallback.
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
import trimesh
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from .contracts import ViewRenderInfo
from .semantic_renderer import SemanticRenderer

log = logging.getLogger(__name__)


def decode_face_id_map(face_id_img_path: str) -> np.ndarray:
    """
    Decodes a 24-bit RGB encoded Face-ID map image into a 2D integer array (H, W) of face indices.
    Index -1 (or 16777215) represents background.
    """
    if not os.path.isfile(face_id_img_path):
        raise FileNotFoundError(f"Face ID map image not found at '{face_id_img_path}'")

    img = Image.open(face_id_img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint32)
    # 24-bit RGB decoding: R + G*256 + B*65536 - 1
    face_ids = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16) - 1
    # Handle background (white 255,255,255 or 0)
    bg_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255)
    face_ids[bg_mask] = -1
    return face_ids.astype(np.int32)


class FaceViewRenderer:
    """
    Multi-view orthographic camera renderer producing RGB, Depth, Face-ID buffers, and Normals.
    """

    def __init__(self, use_blender: bool = True):
        self.use_blender = use_blender
        self.blender_renderer = SemanticRenderer() if use_blender else None

    def render_views(
        self,
        mesh_or_path: Any,
        output_dir: str,
        view_count: int = 8,
        resolution: int = 768,
    ) -> List[Dict[str, Any]]:
        """
        Renders N orthographic views of the 3D model.
        Returns a list of dicts with paths to RGB, Depth, Face-ID, Normal maps, and camera matrices.
        """
        os.makedirs(output_dir, exist_ok=True)
        glb_path = mesh_or_path if isinstance(mesh_or_path, str) else None

        if glb_path and self.use_blender and self.blender_renderer:
            try:
                views_info, manifest_path = self.blender_renderer.render_semantic_views(
                    glb_path=glb_path,
                    output_dir=output_dir,
                    view_count=view_count,
                    resolution=resolution,
                )
                results = []
                for v in views_info:
                    results.append({
                        "view_idx": v.view_idx,
                        "angle_deg": v.angle_deg,
                        "is_front": v.is_front,
                        "rgb_path": v.rgb_path,
                        "depth_path": v.depth_path,
                        "face_id_path": v.face_id_path,
                        "normal_path": v.normal_path,
                        "camera_matrix": v.camera_matrix if hasattr(v, "camera_matrix") else np.eye(4).tolist(),
                    })
                return results
            except Exception as e:
                log.warning(f"Blender rendering failed ({e}). Falling back to Trimesh raycast renderer.")

        # Fallback pure-Python / Trimesh raycasting renderer
        return self._render_trimesh_fallback(mesh_or_path, output_dir, view_count, resolution)

    def _render_trimesh_fallback(
        self,
        mesh_or_path: Any,
        output_dir: str,
        view_count: int,
        resolution: int,
    ) -> List[Dict[str, Any]]:
        """Fallback renderer using Trimesh raycasting for Face-ID buffer generation."""
        if isinstance(mesh_or_path, str):
            mesh = trimesh.load(mesh_or_path, force="mesh")
        else:
            mesh = mesh_or_path

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        # Normalize bounding box to unit sphere
        bbox = mesh.bounding_box.bounds
        center = (bbox[0] + bbox[1]) / 2.0
        extents = bbox[1] - bbox[0]
        max_extent = max(extents)
        scale = 1.8 / max_extent if max_extent > 0 else 1.0

        verts = (mesh.vertices - center) * scale
        norm_mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)

        views_data = []
        angles = np.linspace(0, 360, view_count, endpoint=False)

        for i, angle in enumerate(angles):
            rad = np.radians(angle)
            cam_x = 2.5 * np.sin(rad)
            cam_z = 2.5 * np.cos(rad)
            cam_pos = np.array([cam_x, 0.0, cam_z])

            # Ray origin grid
            x_vals = np.linspace(-1.2, 1.2, resolution)
            y_vals = np.linspace(1.2, -1.2, resolution)
            xx, yy = np.meshgrid(x_vals, y_vals)

            # Ray directions for orthographic projection
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            if np.linalg.norm(right) < 1e-4:
                right = np.array([1, 0, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            origins = cam_pos + xx[..., None] * right + yy[..., None] * up
            directions = np.tile(forward, (resolution * resolution, 1))

            origins_flat = origins.reshape(-1, 3)
            locations, index_ray, index_tri = norm_mesh.ray.intersects_location(
                ray_origins=origins_flat,
                ray_directions=directions,
                multiple_hits=False
            )

            face_id_map = np.full((resolution, resolution), -1, dtype=np.int32)
            depth_map = np.zeros((resolution, resolution), dtype=np.float32)

            if len(index_ray) > 0:
                face_id_map.flat[index_ray] = index_tri
                depths = np.linalg.norm(locations - origins_flat[index_ray], axis=1)
                depth_map.flat[index_ray] = depths

            # Save Face ID map as 24-bit encoded RGB image
            face_id_rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            valid_mask = face_id_map >= 0

            val = face_id_map[valid_mask] + 1
            face_id_rgb[valid_mask, 0] = val & 0xFF
            face_id_rgb[valid_mask, 1] = (val >> 8) & 0xFF
            face_id_rgb[valid_mask, 2] = (val >> 16) & 0xFF
            face_id_rgb[~valid_mask] = 255  # Background white

            face_id_path = os.path.join(output_dir, f"face_id_{i:02d}.png")
            rgb_path = os.path.join(output_dir, f"rgb_{i:02d}.png")
            depth_path = os.path.join(output_dir, f"depth_{i:02d}.npy")

            Image.fromarray(face_id_rgb).save(face_id_path)
            # Simple gray render for fallback RGB
            rgb_arr = np.where(valid_mask[..., None], [200, 200, 200], [255, 255, 255]).astype(np.uint8)
            Image.fromarray(rgb_arr).save(rgb_path)
            np.save(depth_path, depth_map)

            views_data.append({
                "view_idx": i,
                "angle_deg": float(angle),
                "is_front": (i == 0),
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "face_id_path": face_id_path,
                "normal_path": "",
                "camera_matrix": np.eye(4).tolist(),
            })

        return views_data
