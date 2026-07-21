import pytest
from src.local_asset_factory.scoring.mesh_scoring import check_hard_gates

def test_hard_gates_pass():
    metrics = {
        "silhouette_iou": 0.80,
        "connected_components": 1,
        "has_head": 1,
        "arms_count": 2,
        "legs_count": 2,
        "severe_cavities": 0
    }
    assert len(check_hard_gates(metrics)) == 0

def test_hard_gates_fail_missing_limb():
    metrics = {
        "silhouette_iou": 0.80,
        "connected_components": 1,
        "has_head": 1,
        "arms_count": 1, # missing arm
        "legs_count": 2,
        "severe_cavities": 0
    }
    failed = check_hard_gates(metrics)
    assert "invalid_arms_count" in failed
    
def test_hard_gates_fail_iou():
    metrics = {
        "silhouette_iou": 0.50, # low iou
        "connected_components": 1,
        "has_head": 1,
        "arms_count": 2,
        "legs_count": 2,
        "severe_cavities": 0
    }
    failed = check_hard_gates(metrics)
    assert "silhouette_iou_too_low" in failed
