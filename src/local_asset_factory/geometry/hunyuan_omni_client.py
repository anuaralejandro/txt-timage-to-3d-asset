"""
local_asset_factory · geometry · hunyuan_omni_client
Client wrapper for Hunyuan3D-Omni pose-controlled generation backend.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.enums import OmniControlType
from ..domain.models import GeometryCandidate, GeometryRequest
from ..observability.artifact_store import ArtifactStore
from ..orchestration.vram_scheduler import VRAMScheduler

log = logging.getLogger(__name__)


class HunyuanOmniClient:
    """
    Client for Hunyuan3D-Omni pose-controlled candidate generation.
    """

    def __init__(
        self,
        backend_instance: Any,    # HunyuanOmniBackend
        scheduler: VRAMScheduler,
    ):
        self.backend = backend_instance
        self.scheduler = scheduler

    def generate_pose_candidate(
        self,
        request: GeometryRequest,
        store: ArtifactStore,
        reference_image_path: str,
        *,
        seed: int = 11,
        proportions: Optional[Dict[str, float]] = None,
    ) -> GeometryCandidate:
        """
        Generate a single T-pose candidate using Hunyuan3D-Omni.
        """
        caps = self.backend.capabilities()
        output_dir = store.path(f"candidates/omni/pose_{seed}")

        with self.scheduler.slot("hunyuan3d_omni", required_mb=caps.required_vram_mb):
            result = self.backend.generate_pose_candidate(
                reference_image_path=reference_image_path,
                seed=seed,
                steps=request.inference_steps,
                output_dir=str(output_dir),
                job_id=request.job_id,
                proportions=proportions,
            )

        if result.get("status") != "success" or not result.get("mesh_path"):
            err_msg = result.get("status", "unknown error")
            log.error("Hunyuan3D-Omni candidate generation failed: %s", err_msg)
            cand = GeometryCandidate(
                backend="hunyuan3d_omni",
                omni_control=OmniControlType.POSE,
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
            producer="hunyuan3d_omni",
            checkpoint=caps.checkpoint,
            seed=seed,
            backend="hunyuan3d_omni",
            omni_control=OmniControlType.POSE,
            input_views=[reference_image_path],
            runtime_seconds=result.get("runtime_seconds", 0.0),
            peak_vram_mb=result.get("peak_vram_mb", 0),
            passed_gates=True,
            metadata=result,
        )

        cand = store.register(cand)
        log.info("Registered Hunyuan3D-Omni pose candidate: %s (path: %s)", cand.id[:8], rel_path)
        return cand
