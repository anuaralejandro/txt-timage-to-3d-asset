"""
local_asset_factory · topology · postprocessor
Safe geometric post-processing.
Cleans up, decimates, and validates generated meshes.
"""
import logging
from typing import Dict, Any, Optional
import trimesh
import numpy as np

log = logging.getLogger(__name__)

class PostProcessConfig:
    def __init__(
        self,
        remove_degenerate: bool = True,
        remove_unreferenced: bool = True,
        min_component_ratio: float = 0.05,
        target_faces_preview: int = 20000,
    ):
        self.remove_degenerate = remove_degenerate
        self.remove_unreferenced = remove_unreferenced
        self.min_component_ratio = min_component_ratio
        self.target_faces_preview = target_faces_preview

def safe_cleanup_mesh(mesh: trimesh.Trimesh, config: PostProcessConfig) -> trimesh.Trimesh:
    """Applies safe topological cleanups without destroying the shape."""
    log.info("Starting mesh cleanup. Original vertices: %d, faces: %d", len(mesh.vertices), len(mesh.faces))
    
    if config.remove_degenerate:
        mesh.remove_degenerate_faces()
        
    if config.remove_unreferenced:
        mesh.remove_unreferenced_vertices()
        
    # Remove tiny disconnected components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        total_faces = sum(len(c.faces) for c in components)
        valid_components = []
        for c in components:
            if len(c.faces) / total_faces >= config.min_component_ratio:
                valid_components.append(c)
            else:
                log.info("Removed component with %d faces", len(c.faces))
                
        if valid_components:
            mesh = trimesh.util.concatenate(valid_components)
            
    # Fix normals
    mesh.fix_normals()
    
    log.info("Finished mesh cleanup. Final vertices: %d, faces: %d", len(mesh.vertices), len(mesh.faces))
    return mesh

def validate_glb(glb_path: str) -> Dict[str, Any]:
    """Re-opens a GLB to ensure it's not empty and is valid."""
    result = {
        "is_valid": False,
        "vertices": 0,
        "faces": 0,
        "bounds_finite": False,
        "error": None
    }
    
    try:
        scene = trimesh.load(glb_path, force='scene')
        if not scene.geometry:
            result["error"] = "No geometry found in GLB"
            return result
            
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
        result["vertices"] = len(mesh.vertices)
        result["faces"] = len(mesh.faces)
        
        if result["vertices"] == 0 or result["faces"] == 0:
            result["error"] = "Geometry has 0 vertices or faces"
            return result
            
        result["bounds_finite"] = np.isfinite(mesh.bounds).all()
        if not result["bounds_finite"]:
            result["error"] = "Mesh bounds are not finite (contains NaN/Inf)"
            return result
            
        result["is_valid"] = True
    except Exception as e:
        result["error"] = f"Failed to parse GLB: {str(e)}"
        
    return result
