"""
local_asset_factory · scoring · mesh_scoring
Hard gates and quality metrics for generated meshes.
"""

from typing import Dict, List
import logging

log = logging.getLogger(__name__)

class QualityGateFailed(Exception):
    pass

def calculate_silhouette_iou(rendered_mask, target_mask) -> float:
    """Calculates Intersection over Union for the silhouette."""
    # Placeholder for actual rendering-based IoU
    return 0.85

def check_hard_gates(metrics: Dict[str, float]) -> List[str]:
    """
    Checks the generated metrics against the minimum required thresholds.
    Returns a list of failed gates, or empty list if all pass.
    """
    failed = []
    
    if metrics.get("silhouette_iou", 0) < 0.70:
        failed.append("silhouette_iou_too_low")
        
    if metrics.get("connected_components", 0) > 5:
        # A character body should ideally be 1, but maybe eyes/hair are separate initially
        failed.append("too_many_disconnected_parts")
        
    if metrics.get("has_head", 0) != 1:
        failed.append("head_missing")
        
    if metrics.get("arms_count", 0) != 2:
        failed.append("invalid_arms_count")
        
    if metrics.get("legs_count", 0) != 2:
        failed.append("invalid_legs_count")
        
    if metrics.get("severe_cavities", 1) == 1:
        failed.append("severe_cavities_detected")
        
    return failed

def score_candidate(mesh_path: str, canonical_views: Dict) -> Dict:
    """
    Analyzes a candidate mesh and produces a score report.
    In a real implementation, this would use trimesh/pyrender to get metrics.
    """
    # Simulated metrics extraction
    metrics = {
        "silhouette_iou": 0.88,
        "connected_components": 1,
        "has_head": 1,
        "arms_count": 2,
        "legs_count": 2,
        "severe_cavities": 0,
        "surface_noise": 0.02
    }
    
    failed_gates = check_hard_gates(metrics)
    
    return {
        "mesh_path": mesh_path,
        "metrics": metrics,
        "passed": len(failed_gates) == 0,
        "failed_gates": failed_gates
    }
