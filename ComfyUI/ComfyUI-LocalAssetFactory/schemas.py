"""
ComfyUI-LocalAssetFactory · Pydantic Schemas
All data contracts used across the pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_TEXTURE_RESOLUTIONS = (256, 512, 1024, 2048)

ASSET_TYPES = (
    "prop",
    "weapon",
    "architecture",
    "vegetation",
    "character",
    "enemy",
    "modular_piece",
)

VISUAL_STYLES = (
    "low_poly_anime",
    "stylized_fantasy",
    "hand_painted_anime",
    "royal_princess_fantasy",
)

GAMEPLAY_ROLES = ("hero", "gameplay", "background", "modular_environment")

TARGET_PLATFORMS = (
    "android_low",
    "android_mid",
    "android_high",
    "ios_mid",
    "ios_high",
)


def sanitize_asset_id(raw: str) -> str:
    """Create a safe filesystem-friendly asset_id from a raw string."""
    # Lowercase, replace spaces/special chars with underscore
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw.lower().strip())
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unnamed_asset"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class MobileRequirements(BaseModel):
    """Technical constraints for the target mobile platform."""

    target_platform: str = "android_mid"
    triangle_budget: int = Field(default=2500, gt=0)
    texture_resolution: int = 512
    maximum_materials: int = Field(default=1, ge=1, le=4)

    @field_validator("texture_resolution")
    @classmethod
    def _check_tex_res(cls, v: int) -> int:
        if v not in ALLOWED_TEXTURE_RESOLUTIONS:
            raise ValueError(
                f"texture_resolution must be one of {ALLOWED_TEXTURE_RESOLUTIONS}, got {v}"
            )
        return v


class ModelingRequirements(BaseModel):
    """Guidelines for concept image generation to aid 3D conversion."""

    isolated_object: bool = True
    full_object_visible: bool = True
    background: str = "plain neutral background"
    view: str = "three-quarter orthographic"
    symmetry: str = "preferred"
    thin_elements: str = "avoid"
    detachable_components: List[str] = Field(default_factory=list)


class TexturePromptSet(BaseModel):
    """Set of prompts for texture generation."""

    base_color: str = ""
    ornament_variant: str = ""
    material_style: str = ""
    # Allow additional free-form texture prompts
    extra: Dict[str, str] = Field(default_factory=dict)

    def all_prompts(self) -> Dict[str, str]:
        """Return all non-empty texture prompts as a dict."""
        result: Dict[str, str] = {}
        if self.base_color:
            result["base_color"] = self.base_color
        if self.ornament_variant:
            result["ornament_variant"] = self.ornament_variant
        if self.material_style:
            result["material_style"] = self.material_style
        result.update({k: v for k, v in self.extra.items() if v})
        return result


# ---------------------------------------------------------------------------
# Top-level models
# ---------------------------------------------------------------------------


class AssetRequest(BaseModel):
    """User-provided asset brief — input to the pipeline."""

    asset_name: str
    asset_type: str = "prop"
    description_es: str = ""
    visual_style: str = "low_poly_anime"
    gameplay_role: str = "hero"
    target_platform: str = "android_mid"
    triangle_budget: int = Field(default=2500, gt=0)
    texture_resolution: int = 512
    seed: int = 0

    @field_validator("asset_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in ASSET_TYPES:
            raise ValueError(f"asset_type must be one of {ASSET_TYPES}")
        return v

    @field_validator("visual_style")
    @classmethod
    def _check_style(cls, v: str) -> str:
        if v not in VISUAL_STYLES:
            raise ValueError(f"visual_style must be one of {VISUAL_STYLES}")
        return v

    @field_validator("gameplay_role")
    @classmethod
    def _check_role(cls, v: str) -> str:
        if v not in GAMEPLAY_ROLES:
            raise ValueError(f"gameplay_role must be one of {GAMEPLAY_ROLES}")
        return v

    @field_validator("target_platform")
    @classmethod
    def _check_platform(cls, v: str) -> str:
        if v not in TARGET_PLATFORMS:
            raise ValueError(f"target_platform must be one of {TARGET_PLATFORMS}")
        return v

    @field_validator("texture_resolution")
    @classmethod
    def _check_tex_res(cls, v: int) -> int:
        if v not in ALLOWED_TEXTURE_RESOLUTIONS:
            raise ValueError(
                f"texture_resolution must be one of {ALLOWED_TEXTURE_RESOLUTIONS}"
            )
        return v


class AssetSpecification(BaseModel):
    """Structured asset specification produced by Ollama Prompt Architect."""

    asset_id: str
    asset_name: str
    category: str = "prop"
    visual_style: str = "low_poly_anime"

    concept_prompt_en: str = ""
    negative_prompt_en: str = ""

    texture_prompts: TexturePromptSet = Field(default_factory=TexturePromptSet)
    modeling_requirements: ModelingRequirements = Field(
        default_factory=ModelingRequirements
    )
    mobile_requirements: MobileRequirements = Field(
        default_factory=MobileRequirements
    )

    palette: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    @field_validator("asset_id")
    @classmethod
    def _sanitize_id(cls, v: str) -> str:
        sanitized = sanitize_asset_id(v)
        if sanitized != v:
            # Auto-sanitize rather than reject
            return sanitized
        return v


class BlenderReport(BaseModel):
    """Report from Blender processing step."""

    success: bool = False
    input_model_path: str = ""
    output_model_path: str = ""
    output_format: str = "glb"
    triangle_count_before: int = 0
    triangle_count_after: int = 0
    texture_assigned: bool = False
    preview_renders: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LocalBackends(BaseModel):
    """Records which local backends were used."""

    llm: str = "ollama"
    concept_generator: str = "comfyui_local"
    texture_generator: str = "comfyui_local"
    three_d_generator: str = "hunyuan3d_local"
    blender_processing: bool = True


class SeedRecord(BaseModel):
    """Seeds used at each stage for reproducibility."""

    prompt: int = 0
    concept_image: int = 0
    texture: int = 0
    three_d: int = 0


class OutputPaths(BaseModel):
    """Paths to all generated outputs."""

    concept_image: str = ""
    texture_files: List[str] = Field(default_factory=list)
    raw_model: str = ""
    processed_model: str = ""
    preview_renders: List[str] = Field(default_factory=list)


class TechnicalTarget(BaseModel):
    """Target technical specifications."""

    triangle_budget: int = 2500
    texture_resolution: int = 512
    maximum_materials: int = 1


class AssetManifest(BaseModel):
    """Complete manifest for a generated asset."""

    schema_version: str = "1.0"
    asset_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "generated"

    request: Dict[str, Any] = Field(default_factory=dict)
    specification: Dict[str, Any] = Field(default_factory=dict)

    local_backends: LocalBackends = Field(default_factory=LocalBackends)
    seeds: SeedRecord = Field(default_factory=SeedRecord)
    outputs: OutputPaths = Field(default_factory=OutputPaths)
    technical_target: TechnicalTarget = Field(default_factory=TechnicalTarget)

    pipeline_stages: Dict[str, str] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("asset_id")
    @classmethod
    def _sanitize_id(cls, v: str) -> str:
        return sanitize_asset_id(v)
