"""
ComfyUI-LocalAssetFactory · Output Manager
Centralised management of the output directory tree.

Directory structure per asset:
    <output_root>/<asset_id>/
        ├── concept/          — concept art images
        ├── textures/
        │   └── source/       — generated texture images
        ├── raw_3d/           — TRELLIS output before processing
        ├── processed/        — Blender-processed models
        ├── previews/         — preview renders
        └── manifest.json
"""

from __future__ import annotations

import os
from typing import Optional

from .. import config
from ..utilities.file_safety import safe_asset_dir, sanitize_filename
from ..utilities.logging_utils import get_logger

log = get_logger(__name__)


class OutputManager:
    """Manages the per-asset output directory structure."""

    def __init__(self, output_root: Optional[str] = None):
        self.output_root = output_root or config.ASSET_FACTORY_OUTPUT_DIR

    # ------------------------------------------------------------------
    # Directory getters (create on access)
    # ------------------------------------------------------------------

    def asset_root(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id)

    def concept_dir(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id, "concept")

    def textures_source_dir(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id, "textures", "source")

    def raw_3d_dir(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id, "raw_3d")

    def processed_dir(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id, "processed")

    def previews_dir(self, asset_id: str) -> str:
        return safe_asset_dir(self.output_root, asset_id, "previews")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def concept_image_path(self, asset_id: str, ext: str = "png") -> str:
        """Return the path where the concept image should be saved."""
        d = self.concept_dir(asset_id)
        return os.path.join(d, f"{sanitize_filename(asset_id)}_concept.{ext}")

    def texture_path(self, asset_id: str, tex_type: str, ext: str = "png") -> str:
        """Return the path where a texture of *tex_type* should be saved."""
        d = self.textures_source_dir(asset_id)
        return os.path.join(d, f"{sanitize_filename(asset_id)}_{sanitize_filename(tex_type)}.{ext}")

    def manifest_path(self, asset_id: str) -> str:
        d = self.asset_root(asset_id)
        return os.path.join(d, "manifest.json")

    def ensure_all_dirs(self, asset_id: str) -> None:
        """Pre-create the full directory tree for an asset."""
        self.concept_dir(asset_id)
        self.textures_source_dir(asset_id)
        self.raw_3d_dir(asset_id)
        self.processed_dir(asset_id)
        self.previews_dir(asset_id)
        log.info("Output directories ready for asset: %s", asset_id)
