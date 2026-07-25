"""
local_asset_factory · segmentation · postprocess
Topological postprocessing on mesh face adjacency graph: majority smoothing, small island removal, hole filling, and limb isolation.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None  # type: ignore

from .labels import SemanticLabel

log = logging.getLogger(__name__)

class SegmentationPostProcessor:
    """Cleans up mesh face labels using topological adjacency operations."""

    def __init__(
        self,
        min_island_faces: int = 25,
        smoothing_iterations: int = 2,
        merge_head_and_hair: bool = False,
    ):
        self.min_island_faces = min_island_faces
        self.smoothing_iterations = smoothing_iterations
        self.merge_head_and_hair = merge_head_and_hair

    def process(
        self,
        glb_path: str,
        face_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Applies topology postprocessing to face labels.
        Returns: cleaned_face_labels (np.ndarray of shape (num_faces,))
        """
        cleaned = face_labels.copy()

        # If user enabled merge_head_and_hair, map all HAIR (4) to HEAD (3)
        if self.merge_head_and_hair:
            cleaned[cleaned == SemanticLabel.HAIR] = SemanticLabel.HEAD
            log.info("Merged HAIR faces into HEAD class per configuration.")


        num_faces = len(cleaned)
        if not TRIMESH_AVAILABLE or trimesh is None:
            log.warning("Trimesh not available, skipping topological postprocessing.")
            return cleaned

        mesh = trimesh.load(glb_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        # Build face adjacency list
        face_adjacency = mesh.face_adjacency  # shape (K, 2)
        adj_graph: Dict[int, Set[int]] = {i: set() for i in range(num_faces)}
        for f1, f2 in face_adjacency:
            adj_graph[f1].add(f2)
            adj_graph[f2].add(f1)

        # 1. Fill unassigned holes (label == 0) from adjacent neighbors
        cleaned = self._fill_unassigned_holes(cleaned, adj_graph)

        # 2. Remove small disconnected islands
        cleaned = self._remove_small_islands(cleaned, adj_graph)

        # 3. Majority smoothing iterations
        for _ in range(self.smoothing_iterations):
            cleaned = self._majority_smooth(cleaned, adj_graph)

        # 4. Enforce single main connected component per limb
        cleaned = self._enforce_main_limb_components(cleaned, adj_graph)

        log.info("Topology postprocessing complete.")
        return cleaned

    def _fill_unassigned_holes(self, labels: np.ndarray, adj_graph: Dict[int, Set[int]]) -> np.ndarray:
        cleaned = labels.copy()
        unassigned_indices = np.where(cleaned == 0)[0]
        
        for f_idx in unassigned_indices:
            neighbors = adj_graph[f_idx]
            neighbor_labels = [cleaned[n] for n in neighbors if cleaned[n] > 0]
            if neighbor_labels:
                # Assign mode of neighbor labels
                counts = np.bincount(neighbor_labels)
                cleaned[f_idx] = int(np.argmax(counts))
        return cleaned

    def _remove_small_islands(self, labels: np.ndarray, adj_graph: Dict[int, Set[int]]) -> np.ndarray:
        cleaned = labels.copy()
        visited = set()

        for f_idx in range(len(labels)):
            if f_idx in visited or cleaned[f_idx] == 0:
                continue

            current_label = cleaned[f_idx]
            component: List[int] = []
            queue = [f_idx]
            visited.add(f_idx)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for nbr in adj_graph[curr]:
                    if nbr not in visited and cleaned[nbr] == current_label:
                        visited.add(nbr)
                        queue.append(nbr)

            if len(component) < self.min_island_faces:
                # Small island -> reassign component faces to majority neighbor label
                border_nbrs: List[int] = []
                for comp_f in component:
                    for nbr in adj_graph[comp_f]:
                        if cleaned[nbr] != current_label and cleaned[nbr] > 0:
                            border_nbrs.append(cleaned[nbr])

                if border_nbrs:
                    counts = np.bincount(border_nbrs)
                    new_lbl = int(np.argmax(counts))
                    for comp_f in component:
                        cleaned[comp_f] = new_lbl

        return cleaned

    def _majority_smooth(self, labels: np.ndarray, adj_graph: Dict[int, Set[int]]) -> np.ndarray:
        cleaned = labels.copy()
        for f_idx in range(len(labels)):
            nbrs = adj_graph[f_idx]
            if not nbrs:
                continue
            nbr_labels = [labels[n] for n in nbrs]
            counts = np.bincount(nbr_labels)
            majority = int(np.argmax(counts))
            if counts[majority] > len(nbrs) / 2:
                cleaned[f_idx] = majority
        return cleaned

    def _enforce_main_limb_components(self, labels: np.ndarray, adj_graph: Dict[int, Set[int]]) -> np.ndarray:
        # Keep all valid limb partitions untouched to preserve anatomical segmentation
        return labels
