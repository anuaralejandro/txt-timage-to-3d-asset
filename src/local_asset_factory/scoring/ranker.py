"""
local_asset_factory · scoring · ranker
Ranks geometry candidates and selects the single winner mesh.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Any

log = logging.getLogger(__name__)

class GeometryCandidate:
    def __init__(self, metadata: dict, passed_gates: bool, score: float, path: str, id: str):
        self.metadata = metadata
        self.passed_gates = passed_gates
        self.score = score
        self.path = path
        self.id = id

def rank_candidates(
    candidates: List[Any],
    min_silhouette_iou: float = 0.40,
    max_non_manifold: int = 500,
    max_degenerate: int = 100,
) -> Tuple[Optional[Any], List[Any]]:
    """
    Rank a list of geometry candidates by composite score.
    Returns: (winning_candidate, sorted_candidates_all)
    """
    if not candidates:
        log.warning("No candidates to rank")
        return None, []

    ranked = sorted(candidates, key=lambda c: getattr(c, "score", 0.0) or 0.0, reverse=True)

    winner = None
    for cand in ranked:
        if not getattr(cand, "passed_gates", False):
            log.warning("Candidate %s skipped: failed initial gates", getattr(cand, "id", "unknown"))
            continue

        metrics = getattr(cand, "metadata", {}).get("metrics", {})
        
        iou = metrics.get("silhouette_iou", 1.0)
        if iou < min_silhouette_iou:
            log.warning("Candidate %s rejected: Silhouette IoU too low", cand.id)
            continue

        deg_faces = metrics.get("degenerate_faces", 0)
        if deg_faces > max_degenerate:
            log.warning("Candidate %s rejected: Too many degenerate faces", cand.id)
            continue

        # Winner found
        winner = cand
        log.info(
            "Selected winning candidate %s (score=%.4f, path=%s)",
            cand.id, getattr(cand, "score", 0.0), getattr(cand, "path", "")
        )
        break

    if winner is None:
        log.error("ALL candidates failed quality gates! No geometry selected.")

    return winner, ranked
