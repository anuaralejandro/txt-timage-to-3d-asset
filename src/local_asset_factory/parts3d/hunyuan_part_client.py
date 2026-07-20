"""
local_asset_factory · parts3d · hunyuan_part_client
Client wrapper for Hunyuan3D-Part service backend.
Registers PartSet in ArtifactStore.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..domain.enums import PartClass
from ..domain.models import Part3D, PartSet
from ..observability.artifact_store import ArtifactStore
from ..orchestration.vram_scheduler import VRAMScheduler

log = logging.getLogger(__name__)


class HunyuanPartClient:
    """
    Client interface for Hunyuan3D-Part service.
    Handles VRAM scheduling and returns a registered PartSet domain model.
    """

    def __init__(self, backend_instance: Any, scheduler: VRAMScheduler):
        self.backend = backend_instance
        self.scheduler = scheduler

    def segment_geometry(
        self,
        mesh_path: str | Path,
        selected_geometry_id: str,
        store: ArtifactStore,
        *,
        sam_masks_dir: Optional[str] = None,
    ) -> PartSet:
        """
        Segment 3D character mesh into parts and return registered PartSet.
        """
        caps = self.backend.capabilities()
        out_dir = store.path("parts")

        with self.scheduler.slot("hunyuan3d_part", required_mb=caps.required_vram_mb):
            result = self.backend.segment_mesh(
                mesh_path=mesh_path,
                output_dir=str(out_dir),
                job_id=store.job_id,
                sam_masks_dir=sam_masks_dir,
            )

        if result.get("status") != "success":
            raise RuntimeError(f"Hunyuan3D-Part segmentation failed: {result.get('message')}")

        parts_dict = result.get("parts", {})
        parts_list = []

        for name, info in parts_dict.items():
            f_path = Path(info["file"])
            rel_path = str(f_path.relative_to(store.job_root))
            p_class = PartClass(info.get("part_class", "organic_deforming"))

            p3d = Part3D(
                semantic_name=name,
                part_class=p_class,
                relative_path=rel_path,
                confidence=info.get("confidence", 0.9),
                producer="hunyuan3d_part",
                checkpoint=caps.checkpoint,
            )
            parts_list.append(p3d)

        part_set = PartSet(
            job_id=store.job_id,
            selected_geometry_id=selected_geometry_id,
            parts=parts_list,
            runtime_seconds=result.get("runtime_seconds", 0.0),
        )

        log.info("PartSet registered with %d parts", len(parts_list))
        return part_set
