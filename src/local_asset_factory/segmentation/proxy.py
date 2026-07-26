"""
local_asset_factory · segmentation · proxy
Proxy mesh decimation and BVH nearest-surface label transfer back to high-poly original mesh.
"""

import os
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

log = logging.getLogger(__name__)

class ProxyManager:
    """Manages high-poly mesh detection, proxy decimation, and label transfer."""

    def __init__(
        self,
        triangle_threshold: int = 250000,
        target_triangles: int = 150000,
    ):
        self.triangle_threshold = triangle_threshold
        self.target_triangles = target_triangles

    def prepare_segmentation_mesh(
        self,
        input_glb_path: str,
        output_dir: str,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Inspects input GLB. If triangle count exceeds threshold, creates proxy.
        Returns: (segmentation_mesh_path, proxy_used, metadata)
        """
        if not TRIMESH_AVAILABLE:
            log.warning("Trimesh not available. Skipping proxy creation.")
            return input_glb_path, False, {"error": "trimesh not installed"}

        scene_or_mesh = trimesh.load(input_glb_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(scene_or_mesh.dump())
        else:
            mesh = scene_or_mesh

        num_faces = len(mesh.faces)
        log.info(f"Loaded mesh '{input_glb_path}' with {num_faces} faces.")

        if num_faces <= self.triangle_threshold:
            log.info(f"Face count {num_faces} <= threshold {self.triangle_threshold}. No proxy needed.")
            return input_glb_path, False, {"face_count": num_faces, "proxy_used": False}

        # Decimate to create proxy
        log.info(f"Face count {num_faces} exceeds threshold {self.triangle_threshold}. Decimating to ~{self.target_triangles} faces.")
        target_ratio = min(1.0, float(self.target_triangles) / float(num_faces))
        try:
            proxy_mesh = mesh.simplify_quadric_decimation(int(num_faces * target_ratio))
        except Exception:
            try:
                proxy_mesh = mesh.simplify_quadratic_decimation(int(num_faces * target_ratio))
            except Exception:
                proxy_mesh = mesh
        
        os.makedirs(output_dir, exist_ok=True)
        proxy_path = os.path.join(output_dir, "segmentation_proxy.glb")
        proxy_mesh.export(proxy_path)
        
        proxy_faces = len(proxy_mesh.faces)
        log.info(f"Saved proxy mesh to '{proxy_path}' with {proxy_faces} faces.")
        
        metadata = {
            "original_face_count": num_faces,
            "proxy_face_count": proxy_faces,
            "proxy_used": True,
            "proxy_path": proxy_path,
        }
        return proxy_path, True, metadata

    def transfer_labels_to_highpoly(
        self,
        original_glb_path: str,
        proxy_glb_path: str,
        proxy_face_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Transfers face labels from proxy mesh back to original high-poly mesh using BVH nearest surface query.
        Returns: highpoly_face_labels (numpy array of shape (num_original_faces,))
        """
        if not TRIMESH_AVAILABLE:
            raise RuntimeError("Trimesh is required for proxy label transfer.")

        orig_mesh = trimesh.load(original_glb_path, force="mesh")
        proxy_mesh = trimesh.load(proxy_glb_path, force="mesh")

        if isinstance(orig_mesh, trimesh.Scene):
            orig_mesh = trimesh.util.concatenate(orig_mesh.dump())
        if isinstance(proxy_mesh, trimesh.Scene):
            proxy_mesh = trimesh.util.concatenate(proxy_mesh.dump())

        # Compute centroids of original highpoly faces
        orig_centroids = orig_mesh.triangles.mean(axis=1)  # shape (N_orig, 3)

        # Query nearest points on proxy surface using BVH proximity
        closest_points, distances, proxy_face_indices = proxy_mesh.nearest.on_surface(orig_centroids)

        # Map labels from proxy face indices to highpoly faces
        highpoly_labels = proxy_face_labels[proxy_face_indices]

        # Verify that all original faces received valid labels
        unassigned_count = np.sum(highpoly_labels == 0)
        log.info(
            f"Transferred labels from proxy ({len(proxy_face_labels)} faces) to highpoly ({len(highpoly_labels)} faces). "
            f"Unassigned faces: {unassigned_count}"
        )

        # If any highpoly face is unassigned (0), fall back to nearest neighbor among non-zero proxy faces
        if unassigned_count > 0:
            valid_proxy_mask = proxy_face_labels > 0
            if np.any(valid_proxy_mask):
                valid_proxy_indices = np.where(valid_proxy_mask)[0]
                valid_proxy_centroids = proxy_mesh.triangles[valid_proxy_indices].mean(axis=1)
                
                unassigned_indices = np.where(highpoly_labels == 0)[0]
                unassigned_centroids = orig_centroids[unassigned_indices]
                
                # KNN search for unassigned faces
                from scipy.spatial import KDTree
                tree = KDTree(valid_proxy_centroids)
                _, nearest_valid_k = tree.query(unassigned_centroids)
                
                highpoly_labels[unassigned_indices] = proxy_face_labels[valid_proxy_indices[nearest_valid_k]]

        return highpoly_labels
