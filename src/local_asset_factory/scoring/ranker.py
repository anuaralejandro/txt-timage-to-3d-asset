"""
local_asset_factory · scoring · ranker
Ranks geometry candidates and selects the single winner mesh.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..domain.models import GeometryCandidate
from ..observability.artifact_store import ArtifactStore

log = logging.getLogger(__name__)


def rank_candidates(
    candidates: List[GeometryCandidate],
    store: Optional[ArtifactStore] = None,
    *,
    min_silhouette_iou: float = 0.40,
    max_non_manifold: int = 500,
    max_degenerate: int = 100,
) -> Tuple[Optional[GeometryCandidate], List[GeometryCandidate]]:
    """
    Rank a list of geometry candidates by composite_score after passing quality gates.

    Quality Gates:
      1. passed_gates is True
      2. silhouette_iou >= min_silhouette_iou
      3. non_manifold_edges <= max_non_manifold
      4. degenerate_faces <= max_degenerate

    Returns: (winning_candidate, sorted_candidates_all)
    """
    if not candidates:
        log.warning("No candidates to rank")
        return None, []

    ranked = sorted(candidates, key=lambda c: c.metrics.composite_score, reverse=True)

    winner: Optional[GeometryCandidate] = None
    for cand in ranked:
        if not cand.passed_gates:
            log.warning("Candidate %s skipped: failed initial gates (%s)", cand.id[:8], cand.warnings)
            if store:
                store.preserve_failed_candidate(cand.id, f"Initial gate failure: {cand.warnings}")
            continue

        m = cand.metrics
        if m.silhouette_iou < min_silhouette_iou:
            msg = f"Silhouette IoU too low ({m.silhouette_iou:.2f} < {min_silhouette_iou:.2f})"
            log.warning("Candidate %s rejected: %s", cand.id[:8], msg)
            if store:
                store.preserve_failed_candidate(cand.id, msg)
            continue

        if m.non_manifold_edges > max_non_manifold:
            msg = f"Too many non-manifold edges ({m.non_manifold_edges} > {max_non_manifold})"
            log.warning("Candidate %s rejected: %s", cand.id[:8], msg)
            if store:
                store.preserve_failed_candidate(cand.id, msg)
            continue

        if m.degenerate_faces > max_degenerate:
            msg = f"Too many degenerate faces ({m.degenerate_faces} > {max_degenerate})"
            log.warning("Candidate %s rejected: %s", cand.id[:8], msg)
            if store:
                store.preserve_failed_candidate(cand.id, msg)
            continue

        # Winner found
        winner = cand
        log.info(
            "Selected winning candidate %s (backend=%s, score=%.4f, path=%s)",
            cand.id[:8], cand.backend, m.composite_score, cand.relative_path
        )
        break

    if winner is None:
        log.error("ALL candidates failed quality gates! No geometry selected.")

    return winner, ranked
