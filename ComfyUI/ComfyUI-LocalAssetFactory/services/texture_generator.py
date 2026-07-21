"""
ComfyUI-LocalAssetFactory · Texture Generator Service
Orchestrates local texture generation via ComfyUI workflows.

Supports two modes:
  A. Surface textures — tileable, for walls/floors/materials
  B. Object textures — unique, for hero assets like swords/shields

All generation is local-first using ComfyUI's text-to-image pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .. import config
from ..schemas import AssetSpecification, TexturePromptSet
from ..services.comfy_bridge import ComfyBridge
from ..utilities.file_safety import safe_asset_dir
from ..utilities.image_conversion import save_pil_image
from ..utilities.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

_SURFACE_PREFIX = (
    "flat albedo game texture, seamless tileable, stylized hand-painted, "
    "mobile-friendly clean base color, anime low poly style, no text, "
    "no watermark, no shadows, no lighting effects, "
)

_OBJECT_PREFIX = (
    "flat albedo game texture, stylized hand-painted, anime low poly style, "
    "clean base color, mobile-friendly, no text, no watermark, "
    "no dramatic lighting, no background, "
)

_TEXTURE_NEGATIVE = (
    "text, watermark, logo, signature, blurry, noisy, photorealistic, "
    "dramatic lighting, dark shadows, multiple objects, perspective, "
    "3d render, photograph, border, frame"
)


def build_texture_prompt(
    raw_prompt: str,
    is_tileable: bool = False,
    style: str = "low_poly_anime",
) -> str:
    """Build a final texture generation prompt from a raw description."""
    prefix = _SURFACE_PREFIX if is_tileable else _OBJECT_PREFIX
    style_tag = f"{style.replace('_', ' ')} style, "
    return f"{prefix}{style_tag}{raw_prompt.strip()}"


def get_texture_negative() -> str:
    """Return the standard negative prompt for texture generation."""
    return _TEXTURE_NEGATIVE


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class TextureGenerator:
    """Generate textures locally using ComfyUI text-to-image workflows."""

    def __init__(
        self,
        bridge: Optional[ComfyBridge] = None,
        workflow_path: Optional[str] = None,
    ):
        self.bridge = bridge or ComfyBridge()
        self.workflow_path = workflow_path or config.TEXTURE_WORKFLOW_PATH

    def generate_textures(
        self,
        texture_prompts: TexturePromptSet,
        asset_spec: AssetSpecification,
        output_root: str,
        *,
        generate_base_color: bool = True,
        generate_ornament_variant: bool = False,
        tileable_mode: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Generate requested textures and save them.

        Returns a dict with:
            texture_paths: dict[str, str] — map of texture type to file path
            preview_path: str — path to the first generated texture (for preview)
            status: str
        """
        result: Dict[str, Any] = {
            "texture_paths": {},
            "preview_path": "",
            "status": "pending",
        }

        if not config.ENABLE_TEXTURE_GENERATION:
            result["status"] = "skipped: texture generation disabled"
            log.info("Texture generation skipped (disabled).")
            return result

        tex_width = width or config.DEFAULT_TEXTURE_WIDTH
        tex_height = height or config.DEFAULT_TEXTURE_HEIGHT
        output_dir = safe_asset_dir(output_root, asset_spec.asset_id, "textures", "source")

        all_prompts = texture_prompts.all_prompts()
        if not all_prompts:
            result["status"] = "skipped: no texture prompts provided"
            return result

        # Decide which textures to generate
        to_generate: Dict[str, str] = {}
        if generate_base_color and "base_color" in all_prompts:
            to_generate["base_color"] = all_prompts["base_color"]
        elif generate_base_color and all_prompts:
            # Use the first available prompt as base_color
            first_key = next(iter(all_prompts))
            to_generate["base_color"] = all_prompts[first_key]

        if generate_ornament_variant and "ornament_variant" in all_prompts:
            to_generate["ornament_variant"] = all_prompts["ornament_variant"]

        # Also add material_style if available and requested
        if "material_style" in all_prompts and len(to_generate) < 3:
            to_generate["material_style"] = all_prompts["material_style"]

        # Generate each texture
        generated_paths: Dict[str, str] = {}
        errors: List[str] = []

        for tex_type, raw_prompt in to_generate.items():
            try:
                final_prompt = build_texture_prompt(
                    raw_prompt,
                    is_tileable=tileable_mode,
                    style=asset_spec.visual_style,
                )
                log.info("Generating texture '%s': %s", tex_type, final_prompt[:120])

                # Load and configure workflow
                wf = self.bridge.load_workflow(self.workflow_path)
                wf = self.bridge.inject_prompt_params(
                    wf,
                    positive_prompt=final_prompt,
                    negative_prompt=get_texture_negative(),
                    seed=seed,
                    width=tex_width,
                    height=tex_height,
                    steps=steps,
                    cfg=cfg,
                )

                # Execute
                images = self.bridge.execute_and_get_images(wf)
                if images:
                    _, pil_img = images[0]
                    filename = f"{asset_spec.asset_id}_{tex_type}.png"
                    save_path = os.path.join(output_dir, filename)
                    save_pil_image(pil_img, save_path)
                    generated_paths[tex_type] = save_path
                    log.info("Texture saved: %s", save_path)
                else:
                    errors.append(f"No image output for texture '{tex_type}'")

                # Increment seed for variety
                seed += 1

            except Exception as exc:
                msg = f"Texture '{tex_type}' failed: {exc}"
                log.error(msg)
                errors.append(msg)

        result["texture_paths"] = generated_paths
        if generated_paths:
            result["preview_path"] = next(iter(generated_paths.values()))
        if errors:
            result["status"] = f"partial: {len(generated_paths)}/{len(to_generate)} textures, errors: {'; '.join(errors)}"
        elif generated_paths:
            result["status"] = "success"
        else:
            result["status"] = "error: no textures generated"

        return result
