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
    metrics = {
        "vertices": float(len(mesh.vertices)),
        "faces": float(len(mesh.faces)),
        "surface_area": float(mesh.area) if hasattr(mesh, 'area') else 0.0,
        "is_watertight": float(mesh.is_watertight),
        "degenerate_faces": float(len(mesh.faces) - len(mesh.nondegenerate_faces())),
    }
    
    # Calculate finite vertices ratio
    if len(mesh.vertices) > 0:
        finite_verts = np.sum(np.isfinite(mesh.vertices).all(axis=1))
        metrics["finite_vertices_ratio"] = float(finite_verts) / len(mesh.vertices)
    else:
        metrics["finite_vertices_ratio"] = 0.0

    # Calculate main component ratio
    components = mesh.split(only_watertight=False)
    metrics["connected_components"] = float(len(components))
    
    if len(components) > 0:
        main_comp = max(components, key=lambda c: len(c.faces))
        metrics["main_component_face_ratio"] = float(len(main_comp.faces)) / len(mesh.faces)
    else:
        metrics["main_component_face_ratio"] = 0.0
        
    # Extents
    extents = mesh.extents
    metrics["extent_x"] = float(extents[0])
    metrics["extent_y"] = float(extents[1])
    metrics["extent_z"] = float(extents[2])
    
    # Bounding box aspect ratio approximation
    metrics["span_ratio"] = metrics["extent_x"] / max(metrics["extent_y"], 1e-5)
    
    return metrics

def check_hard_gates(metrics: Dict[str, float]) -> List[str]:
    """
    Checks the generated metrics against the minimum required thresholds.
    """
    failed = []
    
    if metrics.get("vertices", 0) < 1000:
        failed.append("insufficient_vertices")
        
    if metrics.get("faces", 0) < 1000:
        failed.append("insufficient_faces")
        
    if metrics.get("finite_vertices_ratio", 1.0) < 1.0:
        failed.append("contains_nan_or_inf")
        
    if metrics.get("main_component_face_ratio", 0) < 0.85:
        failed.append("highly_fragmented_mesh")
        
    if metrics.get("span_ratio", 0) < 0.2:
        # 3D bounding box is too narrow, likely collapsed mesh
        failed.append("collapsed_geometry_x")
        
    if metrics.get("extent_z", 0) < 0.1 * max(metrics.get("extent_y", 1), metrics.get("extent_x", 1)):
        # 3D bounding box is completely flat
        failed.append("collapsed_geometry_z")
        
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
