"""
local_asset_factory · segmentation · p3sam_backend
P3-SAM / Sonata geometric 3D mesh/point-cloud region proposal backend with lazy loading and bypass support.
"""

from __future__ import annotations
import os
import logging
import numpy as np
from typing import Protocol, Dict, Any, Optional, List

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

log = logging.getLogger(__name__)

class P3SAMResult:
    """Stores geometric region proposals per face on a 3D mesh."""
    def __init__(
        self,
        face_region_ids: np.ndarray,  # shape (num_faces,) integer region cluster ID
        num_regions: int,
        confidence: np.ndarray,        # shape (num_faces,) float
    ):
        self.face_region_ids = face_region_ids
        self.num_regions = num_regions
        self.confidence = confidence


class P3SAMBackend(Protocol):
    """Abstract interface for 3D geometric region segmentation backends."""

    def load(self, config: Dict[str, Any]) -> None:
        ...

    def segment_mesh_regions(self, glb_path: str, point_count: int = 50000) -> P3SAMResult:
        ...

    def unload(self) -> None:
        ...


class MockP3SAMBackend:
    """Synthetic mock P3-SAM backend for fast testing without GPU weights."""

    def __init__(self):
        self.is_loaded = False

    def load(self, config: Dict[str, Any]) -> None:
        self.is_loaded = True
        log.info("Loaded MockP3SAMBackend.")

    def segment_mesh_regions(self, glb_path: str, point_count: int = 50000) -> P3SAMResult:
        if not TRIMESH_AVAILABLE:
            raise RuntimeError("Trimesh required for P3-SAM region segmentation.")

        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        num_faces = len(mesh.faces)
        centroids = mesh.triangles.mean(axis=1)  # (N, 3)

        # Cluster mesh faces along height (Y-axis) and spatial clusters to simulate geometric super-regions
        y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
        height = max(1e-5, y_max - y_min)
        
        # 6 geometric height bands: Feet, Legs, Lower Torso, Upper Torso, Neck, Head/Hair
        normalized_y = (centroids[:, 1] - y_min) / height
        region_ids = (normalized_y * 6).astype(np.int32)
        region_ids = np.clip(region_ids, 0, 5)

        num_regions = len(np.unique(region_ids))
        confidence = np.full(num_faces, fill_value=0.90, dtype=np.float32)

        log.info(f"P3-SAM Mock segmented mesh with {num_faces} faces into {num_regions} super-regions.")
        return P3SAMResult(face_region_ids=region_ids, num_regions=num_regions, confidence=confidence)

    def unload(self) -> None:
        self.is_loaded = False
        log.info("Unloaded MockP3SAMBackend.")


class P3SAMSonataBackend:
    """Real P3-SAM / Sonata 3D point cloud segmentation backend using native PyTorch weights."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.predictor = None

    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        log.info("Loading P3-SAM / Sonata neural segmentation model...")
        try:
            import sys
            xpart_path = os.path.abspath(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\src\Hunyuan3D-Part\XPart")
            p3sam_path = os.path.abspath(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\src\Hunyuan3D-Part\P3-SAM")
            for p in [xpart_path, p3sam_path]:
                if p not in sys.path:
                    sys.path.insert(0, p)

            from partgen.bbox_estimator.auto_mask_api import AutoMask
            ckpt_path = os.path.abspath(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\src\Hunyuan3D-Part\XPart\checkpoints\p3sam.ckpt")
            if not os.path.isfile(ckpt_path):
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(repo_id="tencent/Hunyuan3D-Part", filename="p3sam/p3sam.safetensors")

            self.predictor = AutoMask(ckpt_path=ckpt_path)
            log.info("Successfully loaded P3-SAM AutoMask model.")
        except Exception as e:
            log.warning(f"Could not load native P3-SAM predictor: {e}. Falling back to super-region estimation.")
            self.predictor = None

    def segment_mesh_regions(self, glb_path: str, point_count: int = 50000) -> P3SAMResult:
        if self.predictor is not None:
            try:
                log.info(f"Running native P3-SAM 3D neural segmentation on {glb_path} (15-second GPU pass)...")
                mesh_obj = trimesh.load(glb_path, force="mesh")
                if isinstance(mesh_obj, trimesh.Scene):
                    mesh_obj = trimesh.util.concatenate(mesh_obj.dump())
                
                aabb, face_ids, mesh_res = self.predictor.predict_aabb(mesh_obj, post_process=True)
                if face_ids is not None:
                    num_regions = int(len(np.unique(face_ids)))
                    conf = np.full(len(face_ids), fill_value=0.95, dtype=np.float32)
                    log.info(f"P3-SAM GPU Neural Segmentation finished: detected {num_regions} 3D part regions across {len(face_ids)} faces.")
                    return P3SAMResult(face_region_ids=face_ids, num_regions=num_regions, confidence=conf)
            except Exception as e:
                log.warning(f"Native P3-SAM execution error: {e}. Using geometric fallback.")

        mock = MockP3SAMBackend()
        return mock.segment_mesh_regions(glb_path, point_count)

    def unload(self) -> None:
        self.predictor = None
        log.info("Unloaded P3SAMSonataBackend.")

