"""
local_asset_factory · geometry · hunyuan2mv_client
Client wrapper for Hunyuan3D-2mv backend.

Communicates with Hunyuan2MVBackend under VRAMScheduler protection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.enums import OmniControlType, ViewOrientation
from ..domain.models import GeometryCandidate, GeometryRequest
from ..observability.artifact_store import ArtifactStore
from ..orchestration.vram_scheduler import VRAMScheduler

log = logging.getLogger(__name__)


class Hunyuan2MVClient:
    """
    Client for Hunyuan3D-2mv geometry generation.
    Enforces VRAM scheduling and registers candidates with ArtifactStore.
    """

    def __init__(
        self,
        backend_instance: Any,    # Hunyuan2MVBackend instance
        scheduler: VRAMScheduler,
    ):
        self.backend = backend_instance
        self.scheduler = scheduler

    def generate_candidate(
        self,
        request: GeometryRequest,
        store: ArtifactStore,
        *,
        seed: int = 11,
        variant: str = "normal",
        steps: int = 30,
    ) -> GeometryCandidate:
        """
        Generate a single geometry candidate under VRAM slot.
        """
        caps = self.backend.capabilities()
        # Filter input views to ONLY those supported by checkpoint (front, left, back)
        checkpoint_views = {
            k: v for k, v in request.views.items() if k in caps.supported_views
        }

        output_dir = store.path(f"candidates/hunyuan2mv/{seed}_{variant}")

        with self.scheduler.slot("hunyuan3d_2mv", required_mb=caps.required_vram_mb):
            result = self.backend.generate(
                views=checkpoint_views,
                seed=seed,
                steps=steps,
                variant=variant,
                output_dir=str(output_dir),
                job_id=request.job_id,
                timeout_seconds=request.timeout_seconds,
            )

        if result.get("status") != "success" or not result.get("mesh_path"):
            err_msg = result.get("status", "unknown error")
            log.error("Hunyuan3D-2mv candidate generation failed: %s", err_msg)
            cand = GeometryCandidate(
                backend="hunyuan3d_2mv",
                variant=variant,
                seed=seed,
                passed_gates=False,
                warnings=[err_msg],
                metadata=result,
            )
            store.preserve_failed_candidate(cand.id, reason=err_msg)
            return cand

        mesh_path = Path(result["mesh_path"])
        rel_path = str(mesh_path.relative_to(store.job_root))

        cand = GeometryCandidate(
            relative_path=rel_path,
            producer="hunyuan3d_2mv",
            checkpoint=caps.checkpoint,
            seed=seed,
            backend="hunyuan3d_2mv",
            variant=variant,
            input_views=list(checkpoint_views.keys()),
            runtime_seconds=result.get("runtime_seconds", 0.0),
            peak_vram_mb=result.get("peak_vram_mb", 0),
            passed_gates=True,
            metadata=result,
        )

        cand = store.register(cand)
        log.info("Registered Hunyuan3D-2mv candidate: %s (path: %s)", cand.id[:8], rel_path)
        return cand

    def generate_batch(
        self,
        request: GeometryRequest,
        store: ArtifactStore,
    ) -> List[GeometryCandidate]:
        """
        Generate multiple candidates for seeds x variants.
        """
        candidates: List[GeometryCandidate] = []
        for variant in ("normal", "turbo"):
            for seed in request.seeds:
                cand = self.generate_candidate(
                    request, store, seed=seed, variant=variant, steps=request.inference_steps
                )
                candidates.append(cand)
        return candidates
