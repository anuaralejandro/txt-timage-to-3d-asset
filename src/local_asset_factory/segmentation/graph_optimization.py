"""
local_asset_factory · segmentation · graph_optimization
Face-adjacency graph construction, multi-signal fusion (Sapiens2 + P3-SAM + Skeleton),
MRF/Graph Cut smoothing, and proxy-to-full mesh label transfer.
"""

from __future__ import annotations
import logging
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from typing import Dict, Any, List, Tuple, Optional

from .labels import SemanticLabel

log = logging.getLogger(__name__)


def build_face_adjacency_graph(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns face adjacency edges (num_edges, 2) and dihedral angles between adjacent faces.
    """
    adj = mesh.face_adjacency
    normals = mesh.face_normals
    # Calculate angle between adjacent face normals
    n0 = normals[adj[:, 0]]
    n1 = normals[adj[:, 1]]
    dot = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
    dihedral_angles = np.arccos(dot)
    return adj, dihedral_angles


def fuse_anatomical_signals(
    sapiens_probs: np.ndarray,      # (num_faces, 18)
    p3sam_regions: np.ndarray,      # (num_faces,) region IDs
    bone_priors: np.ndarray,         # (num_faces, 18)
    w_sapiens: float = 0.50,
    w_p3sam: float = 0.25,
    w_skeleton: float = 0.25,
) -> np.ndarray:
    """
    Fuses 2D multi-view Sapiens probabilities, P3-SAM geometric clusters, and 3D skeleton bone priors.
    Returns unary cost matrix (num_faces, 18).
    """
    num_faces = len(sapiens_probs)
    num_classes = 18

    # Unary log-likelihoods
    eps = 1e-6
    sapiens_cost = -np.log(np.clip(sapiens_probs, eps, 1.0))
    skeleton_cost = -np.log(np.clip(bone_priors, eps, 1.0))

    # P3-SAM geometric cluster agreement prior
    p3sam_cost = np.zeros((num_faces, num_classes), dtype=np.float32)
    unique_regions = np.unique(p3sam_regions)

    for reg_id in unique_regions:
        if reg_id < 0:
            continue
        mask = (p3sam_regions == reg_id)
        if not np.any(mask):
            continue
        # Mean Sapiens probability within this geometric region
        region_mean_prob = sapiens_probs[mask].mean(axis=0)
        p3sam_cost[mask] = -np.log(np.clip(region_mean_prob, eps, 1.0))

    fused_cost = w_sapiens * sapiens_cost + w_p3sam * p3sam_cost + w_skeleton * skeleton_cost
    return fused_cost


def refine_labels_mrf(
    mesh: trimesh.Trimesh,
    unary_cost: np.ndarray,          # (num_faces, 18)
    p3sam_regions: np.ndarray,       # (num_faces,)
    smoothness_weight: float = 1.0,
    max_iterations: int = 5,
) -> np.ndarray:
    """
    Multi-label MRF optimization over face adjacency graph.
    Smoothes labels across coplanar faces while preserving P3-SAM geometric region boundaries.
    """
    num_faces = len(unary_cost)
    labels = np.argmin(unary_cost, axis=1).astype(np.int32)
    adj, dihedral_angles = build_face_adjacency_graph(mesh)

    # Pre-compute edge penalties vectorized (0.01 seconds instead of hours)
    same_p3sam = (p3sam_regions[adj[:, 0]] == p3sam_regions[adj[:, 1]]) & (p3sam_regions[adj[:, 0]] >= 0)
    edge_penalties = smoothness_weight * (1.0 - 0.5 * dihedral_angles / np.pi)
    edge_penalties[~same_p3sam] *= 0.2

    for iteration in range(max_iterations):
        changed = 0
        for edge_idx, (i, j) in enumerate(adj):
            l_i, l_j = labels[i], labels[j]
            if l_i == l_j:
                continue

            penalty = edge_penalties[edge_idx]
            cost_i_as_j = unary_cost[i, l_j] + penalty
            cost_i_curr = unary_cost[i, l_i]

            if cost_i_as_j < cost_i_curr:
                labels[i] = l_j
                changed += 1

        if changed == 0:
            break

    # Clean up isolated small islands (< 15 faces), preserving hair
    cleaned_labels = remove_isolated_islands(mesh, labels, min_faces=15)
    return cleaned_labels


def remove_isolated_islands(mesh: trimesh.Trimesh, labels: np.ndarray, min_faces: int = 15) -> np.ndarray:
    """Removes small isolated face islands by replacing them with neighboring majority label."""
    refined = labels.copy()
    adj, _ = build_face_adjacency_graph(mesh)

    # Build adjacency list
    adj_list: Dict[int, List[int]] = {f: [] for f in range(len(labels))}
    for u, v in adj:
        adj_list[u].append(v)
        adj_list[v].append(u)

    visited = np.zeros(len(labels), dtype=bool)

    for face in range(len(labels)):
        if visited[face]:
            continue

        label = refined[face]
        # Hair and small hands/feet are preserved even if small
        if label in (SemanticLabel.HAIR, SemanticLabel.HAND_L, SemanticLabel.HAND_R):
            visited[face] = True
            continue

        # BFS connected component
        component = []
        queue = [face]
        visited[face] = True

        while queue:
            curr = queue.pop(0)
            component.append(curr)
            for nbr in adj_list[curr]:
                if not visited[nbr] and refined[nbr] == label:
                    visited[nbr] = True
                    queue.append(nbr)

        if len(component) < min_faces:
            # Find majority neighbor label
            nbr_labels = []
            for f in component:
                for nbr in adj_list[f]:
                    if refined[nbr] != label:
                        nbr_labels.append(refined[nbr])
            if nbr_labels:
                maj_label = max(set(nbr_labels), key=nbr_labels.count)
                for f in component:
                    refined[f] = maj_label

    return refined


def transfer_labels_proxy_to_full(
    proxy_mesh: trimesh.Trimesh,
    proxy_labels: np.ndarray,
    full_mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Transfers face labels from a decimated proxy mesh to the full-resolution mesh
    using KD-tree nearest centroid search.
    """
    proxy_centroids = proxy_mesh.triangles.mean(axis=1)
    full_centroids = full_mesh.triangles.mean(axis=1)

    kdtree = KDTree(proxy_centroids)
    _, nearest_indices = kdtree.query(full_centroids)

    full_labels = proxy_labels[nearest_indices]
    return full_labels.astype(np.int32)
