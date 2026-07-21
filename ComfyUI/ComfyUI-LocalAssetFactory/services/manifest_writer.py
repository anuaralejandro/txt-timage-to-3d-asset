"""
ComfyUI-LocalAssetFactory · Manifest Writer
Creates the comprehensive manifest.json for each generated asset.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..schemas import (
    AssetManifest,
    LocalBackends,
    OutputPaths,
    SeedRecord,
    TechnicalTarget,
)
from ..utilities.file_safety import safe_asset_dir
from ..utilities.logging_utils import get_logger

log = get_logger(__name__)


class ManifestWriter:
    """Build and save asset manifest files."""

    @staticmethod
    def build_manifest(
        asset_id: str,
        *,
        request: Optional[Dict[str, Any]] = None,
        specification: Optional[Dict[str, Any]] = None,
        concept_image_path: str = "",
        texture_paths: Optional[Dict[str, str]] = None,
        raw_model_path: str = "",
        processed_model_path: str = "",
        preview_render_paths: Optional[List[str]] = None,
        blender_report: Optional[Dict[str, Any]] = None,
        seeds: Optional[Dict[str, int]] = None,
        pipeline_stages: Optional[Dict[str, str]] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> AssetManifest:
        """Build a complete AssetManifest from pipeline outputs."""

        # Determine status
        has_errors = bool(errors)
        has_model = bool(processed_model_path or raw_model_path)
        has_concept = bool(concept_image_path)
        has_textures = bool(texture_paths)

        if has_errors and not (has_concept or has_textures or has_model):
            status = "failed"
        elif has_errors:
            status = "partial"
        else:
            status = "generated"

        # Technical target from spec
        spec = specification or {}
        mobile_req = spec.get("mobile_requirements", {})
        tech_target = TechnicalTarget(
            triangle_budget=mobile_req.get("triangle_budget", 2500),
            texture_resolution=mobile_req.get("texture_resolution", 512),
            maximum_materials=mobile_req.get("maximum_materials", 1),
        )

        # Build backends record
        blender_used = bool(blender_report and blender_report.get("success"))
        backends = LocalBackends(blender_processing=blender_used)

        # Build output paths
        outputs = OutputPaths(
            concept_image=concept_image_path,
            texture_files=list((texture_paths or {}).values()),
            raw_model=raw_model_path,
            processed_model=processed_model_path,
            preview_renders=preview_render_paths or [],
        )

        # Seeds
        seed_record = SeedRecord(**(seeds or {}))

        manifest = AssetManifest(
            asset_id=asset_id,
            created_at=datetime.utcnow().isoformat(),
            status=status,
            request=request or {},
            specification=specification or {},
            local_backends=backends,
            seeds=seed_record,
            outputs=outputs,
            technical_target=tech_target,
            pipeline_stages=pipeline_stages or {},
            errors=errors or [],
            warnings=warnings or [],
        )
        return manifest

    @staticmethod
    def save_manifest(
        manifest: AssetManifest,
        output_root: str,
    ) -> str:
        """Write the manifest to disk and return the file path."""
        asset_dir = safe_asset_dir(output_root, manifest.asset_id)
        path = os.path.join(asset_dir, "manifest.json")

        data = manifest.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Manifest saved: %s", path)
        return path
