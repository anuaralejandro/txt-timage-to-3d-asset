"""
local_asset_factory · scoring · mesh_scoring
Hard gates and quality metrics for generated meshes.
"""
import logging
from typing import Dict, List, Optional
import trimesh
import numpy as np

log = logging.getLogger(__name__)

class QualityGateFailed(Exception):
    pass

def calculate_silhouette_iou(rendered_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Calculates Intersection over Union for the silhouette."""
    intersection = np.logical_and(rendered_mask, target_mask)
    union = np.logical_or(rendered_mask, target_mask)
    if not union.any():
        return 0.0
    return float(np.sum(intersection)) / float(np.sum(union))

def render_mesh_views(mesh: trimesh.Trimesh, resolution=(512, 512)) -> Dict[str, np.ndarray]:
    """Renders the mesh from 4 orthogonal angles using pyrender."""
    renders = {}
    try:
        import pyrender
    except ImportError:
        log.warning("pyrender not installed, cannot compute real silhouette IoU. Returning empty masks.")
        return {v: np.zeros(resolution, dtype=bool) for v in ["front", "back", "left", "right"]}
        
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)
    
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    cam_node = scene.add(camera, pose=np.eye(4))
    
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    
    poses = {
        "front": np.array([[1,0,0,0], [0,1,0,0], [0,0,1,2], [0,0,0,1]]),
        "back": np.array([[-1,0,0,0], [0,1,0,0], [0,0,-1,-2], [0,0,0,1]]),
        "left": np.array([[0,0,-1,-2], [0,1,0,0], [1,0,0,0], [0,0,0,1]]),
        "right": np.array([[0,0,1,2], [0,1,0,0], [-1,0,0,0], [0,0,0,1]]),
    }
    
    for view, pose in poses.items():
        scene.set_pose(cam_node, pose=pose)
        color, depth = r.render(scene)
        # Silhouette is anywhere depth is not 0
        renders[view] = depth > 0
        
    r.delete()
    return renders

def extract_mesh_metrics(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Extracts genuine geometric properties from a trimesh."""
    num_verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
    num_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
    
    metrics = {
        "vertices": float(num_verts),
        "faces": float(num_faces),
        "surface_area": float(mesh.area) if (hasattr(mesh, 'area') and num_verts > 0) else 0.0,
        "is_watertight": float(mesh.is_watertight) if (hasattr(mesh, 'is_watertight') and num_verts > 0) else 0.0,
        "degenerate_faces": float(num_faces - len(mesh.nondegenerate_faces())) if (hasattr(mesh, 'nondegenerate_faces') and num_faces > 0) else 0.0,
    }
    
    # Calculate finite vertices ratio
    if num_verts > 0:
        finite_verts = np.sum(np.isfinite(mesh.vertices).all(axis=1))
        metrics["finite_vertices_ratio"] = float(finite_verts) / num_verts
    else:
        metrics["finite_vertices_ratio"] = 0.0

    # Calculate main component ratio
    if num_verts > 0:
        components = mesh.split(only_watertight=False)
        metrics["connected_components"] = float(len(components))
        
        if len(components) > 0:
            main_comp = max(components, key=lambda c: len(c.faces))
            metrics["main_component_face_ratio"] = float(len(main_comp.faces)) / num_faces if num_faces > 0 else 0.0
        else:
            metrics["main_component_face_ratio"] = 0.0
    else:
        metrics["connected_components"] = 0.0
        metrics["main_component_face_ratio"] = 0.0
        
    # Extents
    if num_verts > 0 and hasattr(mesh, 'extents') and mesh.extents is not None:
        extents = mesh.extents
    else:
        extents = np.array([0.0, 0.0, 0.0])

    metrics["extent_x"] = float(extents[0])
    metrics["extent_y"] = float(extents[1])
    metrics["extent_z"] = float(extents[2])
    
    # Bounding box aspect ratio approximation
    metrics["span_ratio"] = metrics["extent_x"] / max(metrics["extent_y"], 1e-5)
    
    return metrics

def check_hard_gates(metrics: Dict[str, float]) -> List[str]:
    """
    Checks the generated metrics against the minimum required thresholds.
    Supports both low-level geometric mesh metrics and high-level semantic quality gates.
    """
    failed = []
    
    # Geometric mesh gates (only checked if vertices/faces key present)
    if "vertices" in metrics and metrics.get("vertices", 0) < 1000:
        failed.append("insufficient_vertices")
        
    if "faces" in metrics and metrics.get("faces", 0) < 1000:
        failed.append("insufficient_faces")
        
    if "finite_vertices_ratio" in metrics and metrics.get("finite_vertices_ratio", 1.0) < 1.0:
        failed.append("contains_nan_or_inf")
        
    if "main_component_face_ratio" in metrics and metrics.get("main_component_face_ratio", 1.0) < 0.85:
        failed.append("highly_fragmented_mesh")
        
    if "span_ratio" in metrics and metrics.get("span_ratio", 1.0) < 0.2:
        # 3D bounding box is too narrow, likely collapsed mesh
        failed.append("collapsed_geometry_x")
        
    if "extent_z" in metrics and metrics.get("extent_z", 1.0) < 0.1 * max(metrics.get("extent_y", 1), metrics.get("extent_x", 1)):
        # 3D bounding box is completely flat
        failed.append("collapsed_geometry_z")

    # Semantic quality gates
    if "silhouette_iou" in metrics and metrics.get("silhouette_iou", 1.0) < 0.70:
        failed.append("silhouette_iou_too_low")

    if "has_head" in metrics and metrics.get("has_head", 1) < 1:
        failed.append("head_missing")

    if "arms_count" in metrics and metrics.get("arms_count", 2) != 2:
        failed.append("invalid_arms_count")

    if "legs_count" in metrics and metrics.get("legs_count", 2) != 2:
        failed.append("invalid_legs_count")

    if "severe_cavities" in metrics and metrics.get("severe_cavities", 0) > 0:
        failed.append("severe_cavities_detected")
        
    return failed

def score_candidate(mesh_path: str, canonical_views: Dict) -> Dict:
    """
    Analyzes a candidate mesh and produces a score report.
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # If it's a scene, extract the largest mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("Empty Scene")
            geom = list(mesh.geometry.values())
            mesh = max(geom, key=lambda m: len(m.faces))

        metrics = extract_mesh_metrics(mesh)
        
        # Real render to reference IoU
        renders = render_mesh_views(mesh)
        
        # To compute actual IoU, we need the reference masks from canonical_views.
        # If reference masks were passed in canonical_views dict, we compare them:
        iou_scores = []
        for view, ref_path in canonical_views.items():
            if view in renders:
                try:
                    from PIL import Image
                    ref_img = Image.open(ref_path).convert("RGBA")
                    ref_img = ref_img.resize((512, 512))
                    ref_mask = np.array(ref_img)[:, :, 3] > 10
                    iou = calculate_silhouette_iou(renders[view], ref_mask)
                    iou_scores.append(iou)
                    if view == "front":
                        metrics["silhouette_iou_front"] = iou
                except Exception as e:
                    log.warning("Could not compute IoU for view %s: %s", view, e)
                    
        metrics["silhouette_iou_mean"] = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

        failed_gates = check_hard_gates(metrics)
        
        return {
            "mesh_path": mesh_path,
            "metrics": metrics,
            "passed": len(failed_gates) == 0,
            "failed_gates": failed_gates
        }
    except Exception as e:
        log.error("Failed to score mesh %s: %s", mesh_path, e)
        return {
            "mesh_path": mesh_path,
            "metrics": {},
            "passed": False,
            "failed_gates": ["mesh_loading_failed"]
        }
