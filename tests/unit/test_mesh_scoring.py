import pytest
import numpy as np

trimesh = pytest.importorskip("trimesh")

from local_asset_factory.scoring.mesh_scoring import extract_mesh_metrics, check_hard_gates

def test_extract_mesh_metrics_empty():
    mesh = trimesh.Trimesh()
    metrics = extract_mesh_metrics(mesh)
    assert metrics["vertices"] == 0
    assert metrics["faces"] == 0
    assert metrics["finite_vertices_ratio"] == 0.0

def test_extract_mesh_metrics_nan_inf():
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, np.nan],
        [np.inf, 1, 1]
    ])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    metrics = extract_mesh_metrics(mesh)
    assert metrics["finite_vertices_ratio"] == 0.5 # 2 out of 4 are finite

def test_extract_mesh_metrics_components():
    # Two disconnected triangles
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [10, 10, 10], [11, 10, 10], [10, 11, 10]
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    metrics = extract_mesh_metrics(mesh)
    assert metrics["connected_components"] == 2
    assert metrics["main_component_face_ratio"] == 0.5

def test_check_hard_gates():
    # Good metrics
    good = {
        "vertices": 2000,
        "faces": 4000,
        "finite_vertices_ratio": 1.0,
        "main_component_face_ratio": 0.9,
        "span_ratio": 0.5,
        "extent_y": 2.0,
        "extent_x": 1.0,
        "extent_z": 0.5
    }
    assert len(check_hard_gates(good)) == 0
    
    # Bad metrics
    bad = {
        "vertices": 500, # fails
        "faces": 4000,
        "finite_vertices_ratio": 0.9, # fails
        "main_component_face_ratio": 0.5, # fails
        "span_ratio": 0.1, # fails
        "extent_y": 2.0,
        "extent_x": 0.2,
        "extent_z": 0.05 # fails
    }
    failed = check_hard_gates(bad)
    assert "insufficient_vertices" in failed
    assert "contains_nan_or_inf" in failed
    assert "highly_fragmented_mesh" in failed
    assert "collapsed_geometry_x" in failed
    assert "collapsed_geometry_z" in failed
