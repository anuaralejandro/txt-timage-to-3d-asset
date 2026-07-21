"""
ComfyUI-LocalAssetFactory · Custom Nodes
Hunyuan multiview character pipeline — thin ComfyUI layer.

Active Nodes (legacy pipeline, kept for compatibility):
1. AssetBriefNode            — Asset Factory · Asset Brief
2. OllamaPromptArchitectNode — Asset Factory · Ollama Prompt Architect
3. LocalConceptImageNode     — Asset Factory · Local Concept Image
4. LocalTextureGeneratorNode — Asset Factory · Local Texture Generator
5. LocalHunyuanNode          — Asset Factory · Hunyuan3D-2 Image-to-3D
6. BlenderProcessorNode      — Asset Factory · Blender Processor
7. SaveManifestNode          — Asset Factory · Save Manifest
8. LocalAssetFactoryPreview3DNode — Asset Factory · 3D GLB Previewer

New Character Pipeline Nodes (src/local_asset_factory/*):
See comfyui/nodes_character_pipeline.py (created in M6).

NOTE: TRELLIS has been removed. Hunyuan3D-2mv is the sole geometry backend.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import config
from .schemas import (
    AssetRequest,
    AssetSpecification,
    TexturePromptSet,
    sanitize_asset_id,
)
from .services.ollama_client import OllamaClient, OllamaError
from .services.comfy_bridge import ComfyBridge, ComfyBridgeError
from .services.texture_generator import TextureGenerator
from .services.blender_runner import BlenderRunner
from .services.manifest_writer import ManifestWriter
from .services.output_manager import OutputManager
from .utilities.image_conversion import (
    pil_to_comfy_tensor,
    comfy_tensor_to_pil,
    save_pil_image,
    load_image_as_comfy,
)
from .utilities.logging_utils import get_logger

log = get_logger(__name__)

# ==========================================================================
# Helpers
# ==========================================================================

_CATEGORY = "Asset Factory"


def _json_dumps(obj: Any) -> str:
    """Safe JSON serialisation."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _json_loads(text: str) -> Dict[str, Any]:
    """Safe JSON parse."""
    if isinstance(text, dict):
        return text
    return json.loads(text)


# ==========================================================================
# NODE 1: Asset Brief
# ==========================================================================

class AssetBriefNode:
    """Captures the user's asset description and validates it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_name": ("STRING", {"default": "Royal Training Sword", "multiline": False}),
                "asset_type": (
                    ["prop", "weapon", "architecture", "vegetation", "character", "enemy", "modular_piece"],
                ),
                "description_es": ("STRING", {
                    "default": "Espada corta ceremonial para una princesa guerrera.",
                    "multiline": True,
                }),
                "visual_style": (
                    ["low_poly_anime", "stylized_fantasy", "hand_painted_anime", "royal_princess_fantasy"],
                ),
                "gameplay_role": (
                    ["hero", "gameplay", "background", "modular_environment"],
                ),
                "target_platform": (
                    ["android_low", "android_mid", "android_high", "ios_mid", "ios_high"],
                ),
                "triangle_budget": ("INT", {"default": 2500, "min": 100, "max": 50000, "step": 100}),
                "texture_resolution": (["256", "512", "1024", "2048"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ASSET_REQUEST_JSON", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(
        self,
        asset_name: str,
        asset_type: str,
        description_es: str,
        visual_style: str,
        gameplay_role: str,
        target_platform: str,
        triangle_budget: int,
        texture_resolution: str,
        seed: int,
    ):
        try:
            request = AssetRequest(
                asset_name=asset_name,
                asset_type=asset_type,
                description_es=description_es,
                visual_style=visual_style,
                gameplay_role=gameplay_role,
                target_platform=target_platform,
                triangle_budget=triangle_budget,
                texture_resolution=int(texture_resolution),
                seed=seed,
            )
            request_json = _json_dumps(request)
            log.info("Asset brief created: %s (%s)", asset_name, asset_type)
            return (request_json, "ok")
        except Exception as exc:
            log.error("Asset brief validation failed: %s", exc)
            return ("{}", f"error: {exc}")


# ==========================================================================
# NODE 2: Ollama Prompt Architect
# ==========================================================================

class OllamaPromptArchitectNode:
    """Calls local Ollama to produce structured spec and optimised prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ASSET_REQUEST_JSON": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "model": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192}),
                "generate_texture_prompts": ("BOOLEAN", {"default": True}),
                "generate_variants": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "ASSET_SPEC_JSON",
        "CONCEPT_IMAGE_PROMPT",
        "CONCEPT_NEGATIVE_PROMPT",
        "TEXTURE_PROMPT_JSON",
        "STATUS",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    # System prompt for the LLM
    _SYSTEM_PROMPT = """You are a game asset specification architect. You receive a game asset description and produce a structured JSON specification.

You MUST output ONLY valid JSON with these exact fields:
{
  "asset_id": "snake_case_unique_name_###",
  "asset_name": "Human Readable Name",
  "category": "weapon|prop|architecture|vegetation|character|enemy|modular_piece",
  "visual_style": "the style",
  "concept_prompt_en": "A detailed prompt for generating concept art of this asset. Must describe: a single isolated object, centered, full object visible, plain neutral background, three-quarter orthographic view, clean lighting, readable silhouette, no text, no watermark, anime/stylized style, low poly.",
  "negative_prompt_en": "text, watermark, logo, multiple objects, complex background, photorealistic, blurry, noisy, dark, cropped, partial view",
  "texture_prompts": {
    "base_color": "A prompt for generating the main base color texture. Should be: flat albedo, stylized, hand-painted, anime style, clean, mobile-friendly, no shadows, no lighting effects.",
    "ornament_variant": "A prompt for an ornamental or decorative texture variant if applicable.",
    "material_style": "A prompt describing the material style for texture generation."
  },
  "modeling_requirements": {
    "isolated_object": true,
    "full_object_visible": true,
    "background": "plain neutral background",
    "view": "three-quarter orthographic",
    "symmetry": "preferred",
    "thin_elements": "avoid",
    "detachable_components": []
  },
  "mobile_requirements": {
    "target_platform": "android_mid",
    "triangle_budget": 2500,
    "texture_resolution": 512,
    "maximum_materials": 1
  },
  "palette": ["#color1", "#color2"],
  "tags": ["tag1", "tag2"]
}

Rules:
- The concept_prompt_en must be ideal for image-to-3D conversion.
- Texture prompts must produce clean, mobile-friendly textures.
- All prompts must be in English.
- The asset_id must be snake_case with no spaces.
- Respond with ONLY the JSON object, no explanation."""

    def execute(
        self,
        ASSET_REQUEST_JSON: str,
        model: str = "",
        temperature: float = 0.5,
        max_tokens: int = 2048,
        generate_texture_prompts: bool = True,
        generate_variants: bool = False,
    ):
        try:
            request_data = _json_loads(ASSET_REQUEST_JSON)
            request = AssetRequest(**request_data)
        except Exception as exc:
            return ("", "", "", "", f"error: invalid request JSON: {exc}")

        # Build the user prompt
        user_prompt = (
            f"Create a game asset specification for:\n"
            f"Name: {request.asset_name}\n"
            f"Type: {request.asset_type}\n"
            f"Description (Spanish): {request.description_es}\n"
            f"Visual Style: {request.visual_style}\n"
            f"Gameplay Role: {request.gameplay_role}\n"
            f"Target Platform: {request.target_platform}\n"
            f"Triangle Budget: {request.triangle_budget}\n"
            f"Texture Resolution: {request.texture_resolution}\n"
        )
        if not generate_texture_prompts:
            user_prompt += "\nNote: You may leave texture_prompts empty."
        if generate_variants:
            user_prompt += "\nNote: Include variant suggestions in tags."

        # Call Ollama
        try:
            client = OllamaClient(model=model if model else None)

            if not client.is_available():
                return ("", "", "", "", "error: Ollama is not available at " + client.base_url)

            spec_data = client.generate_json(
                user_prompt,
                system=self._SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Validate with Pydantic
            spec = AssetSpecification(**spec_data)

            spec_json = _json_dumps(spec)
            concept_prompt = spec.concept_prompt_en
            negative_prompt = spec.negative_prompt_en
            texture_json = _json_dumps(spec.texture_prompts)

            log.info(
                "Spec generated for: %s (prompt len=%d)",
                spec.asset_id,
                len(concept_prompt),
            )
            return (spec_json, concept_prompt, negative_prompt, texture_json, "ok")

        except OllamaError as exc:
            log.error("Ollama error: %s", exc)
            return ("", "", "", "", f"error: {exc}")
        except Exception as exc:
            log.error("Prompt architect failed: %s\n%s", exc, traceback.format_exc())
            return ("", "", "", "", f"error: {exc}")


# ==========================================================================
# NODE 3: Local Concept Image Generator
# ==========================================================================

class LocalConceptImageNode:
    """Generates concept art locally via ComfyUI workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CONCEPT_IMAGE_PROMPT": ("STRING", {"forceInput": True}),
                "CONCEPT_NEGATIVE_PROMPT": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "ASSET_SPEC_JSON": ("STRING", {"forceInput": True, "default": ""}),
                "workflow_path": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "IMAGE_PATH", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(
        self,
        CONCEPT_IMAGE_PROMPT: str,
        CONCEPT_NEGATIVE_PROMPT: str,
        ASSET_SPEC_JSON: str = "",
        workflow_path: str = "",
        width: int = 768,
        height: int = 768,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0,
        enabled: bool = True,
    ):
        # Empty image placeholder
        empty_image = np.zeros((1, 64, 64, 3), dtype=np.float32)

        if not enabled:
            return (empty_image, "", "skipped: disabled")

        if not CONCEPT_IMAGE_PROMPT.strip():
            return (empty_image, "", "skipped: empty prompt")

        # Determine asset_id for saving
        asset_id = "concept"
        if ASSET_SPEC_JSON:
            try:
                spec_data = _json_loads(ASSET_SPEC_JSON)
                asset_id = spec_data.get("asset_id", "concept")
            except Exception:
                pass

        wf_path = workflow_path or config.CONCEPT_WORKFLOW_PATH

        try:
            bridge = ComfyBridge()

            if not bridge.is_available():
                return (empty_image, "", "error: ComfyUI not available at " + bridge.base_url)

            wf = bridge.load_workflow(wf_path)
            wf = bridge.inject_prompt_params(
                wf,
                positive_prompt=CONCEPT_IMAGE_PROMPT,
                negative_prompt=CONCEPT_NEGATIVE_PROMPT,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
            )

            images = bridge.execute_and_get_images(wf)
            if not images:
                return (empty_image, "", "error: no image generated")

            _, pil_img = images[0]

            # Save to output directory
            output_mgr = OutputManager()
            save_path = output_mgr.concept_image_path(asset_id)
            save_pil_image(pil_img, save_path)

            tensor = pil_to_comfy_tensor(pil_img)
            log.info("Concept image generated: %s", save_path)
            return (tensor, save_path, "ok")

        except Exception as exc:
            log.error("Concept image failed: %s\n%s", exc, traceback.format_exc())
            return (empty_image, "", f"error: {exc}")


# ==========================================================================
# NODE 4: Local Texture Generator
# ==========================================================================

class LocalTextureGeneratorNode:
    """Generates textures locally using ComfyUI text-to-image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TEXTURE_PROMPT_JSON": ("STRING", {"forceInput": True}),
                "ASSET_SPEC_JSON": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "workflow_path": ("STRING", {"default": ""}),
                "generate_base_color": ("BOOLEAN", {"default": True}),
                "generate_ornament_variant": ("BOOLEAN", {"default": False}),
                "tileable_mode": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("TEXTURE_PATHS_JSON", "PREVIEW_TEXTURE_IMAGE", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(
        self,
        TEXTURE_PROMPT_JSON: str,
        ASSET_SPEC_JSON: str,
        workflow_path: str = "",
        generate_base_color: bool = True,
        generate_ornament_variant: bool = False,
        tileable_mode: bool = False,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = 0,
        enabled: bool = True,
    ):
        empty_image = np.zeros((1, 64, 64, 3), dtype=np.float32)

        if not enabled or not config.ENABLE_TEXTURE_GENERATION:
            return ("{}", empty_image, "skipped: disabled")

        try:
            tex_data = _json_loads(TEXTURE_PROMPT_JSON)
            spec_data = _json_loads(ASSET_SPEC_JSON)
            texture_prompts = TexturePromptSet(**tex_data)
            spec = AssetSpecification(**spec_data)
        except Exception as exc:
            return ("{}", empty_image, f"error: invalid input JSON: {exc}")

        try:
            wf = workflow_path or None
            bridge = ComfyBridge()
            gen = TextureGenerator(bridge=bridge, workflow_path=wf)

            result = gen.generate_textures(
                texture_prompts=texture_prompts,
                asset_spec=spec,
                output_root=config.ASSET_FACTORY_OUTPUT_DIR,
                generate_base_color=generate_base_color,
                generate_ornament_variant=generate_ornament_variant,
                tileable_mode=tileable_mode,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )

            paths_json = json.dumps(result.get("texture_paths", {}), indent=2)

            # Load preview image
            preview_path = result.get("preview_path", "")
            if preview_path and os.path.isfile(preview_path):
                preview_tensor = load_image_as_comfy(preview_path)
            else:
                preview_tensor = empty_image

            return (paths_json, preview_tensor, result.get("status", "unknown"))

        except Exception as exc:
            log.error("Texture generation failed: %s\n%s", exc, traceback.format_exc())
            return ("{}", empty_image, f"error: {exc}")


# ==========================================================================
# NODE 5: Local Hunyuan3D-2 Image-to-3D (replaces TRELLIS)
# ==========================================================================

class LocalHunyuanNode:
    """Runs Hunyuan3D-2 image-to-3D via native ComfyUI nodes.

    Supports single-image (props/weapons) and multi-view (characters).
    Multi-view accepts front, left, back, right images for T-Pose characters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "FRONT_IMAGE": ("IMAGE",),
                "LEFT_IMAGE": ("IMAGE",),
                "BACK_IMAGE": ("IMAGE",),
                "RIGHT_IMAGE": ("IMAGE",),
                "FRONT_IMAGE_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "LEFT_IMAGE_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "BACK_IMAGE_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "RIGHT_IMAGE_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "ASSET_SPEC_JSON": ("STRING", {"forceInput": True, "default": ""}),
                "multiview_folder": ("STRING", {"default": ""}),
                "workflow_path": ("STRING", {"default": ""}),
                "generate_3d": ("BOOLEAN", {"default": True}),
                "resolution": ("INT", {"default": 3072, "min": 1024, "max": 8192, "step": 256}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "mesh_algorithm": (["surface net", "basic"],),
                "mesh_threshold": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "timeout_seconds": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL_PATH", "ALL_OUTPUT_PATHS_JSON", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(
        self,
        FRONT_IMAGE=None,
        LEFT_IMAGE=None,
        BACK_IMAGE=None,
        RIGHT_IMAGE=None,
        FRONT_IMAGE_PATH: str = "",
        LEFT_IMAGE_PATH: str = "",
        BACK_IMAGE_PATH: str = "",
        RIGHT_IMAGE_PATH: str = "",
        ASSET_SPEC_JSON: str = "",
        multiview_folder: str = "",
        workflow_path: str = "",
        generate_3d: bool = True,
        resolution: int = 3072,
        steps: int = 30,
        cfg: float = 5.0,
        mesh_algorithm: str = "surface net",
        mesh_threshold: float = 0.6,
        seed: int = 0,
        timeout_seconds: int = 600,
    ):
        if not generate_3d or not config.ENABLE_HUNYUAN:
            return ("", "[]", "skipped: Hunyuan3D disabled")

        # Save tensor images to temp paths if provided
        temp_dir = os.path.join(config.ASSET_FACTORY_OUTPUT_DIR, "_temp")
        os.makedirs(temp_dir, exist_ok=True)

        image_map = {
            "front": (FRONT_IMAGE, FRONT_IMAGE_PATH),
            "left": (LEFT_IMAGE, LEFT_IMAGE_PATH),
            "back": (BACK_IMAGE, BACK_IMAGE_PATH),
            "right": (RIGHT_IMAGE, RIGHT_IMAGE_PATH),
        }

        paths = {}
        for view_name, (tensor, path) in image_map.items():
            if path and os.path.isfile(path):
                paths[view_name] = path
            elif tensor is not None:
                try:
                    pil_img = comfy_tensor_to_pil(tensor)
                    save_path = os.path.join(temp_dir, f"hunyuan_{view_name}.png")
                    pil_img.save(save_path)
                    paths[view_name] = save_path
                except Exception as exc:
                    log.warning("Could not save %s image: %s", view_name, exc)

        # Scan folder for multiview images if provided
        if multiview_folder and os.path.isdir(multiview_folder):
            for filename in os.listdir(multiview_folder):
                lower_name = filename.lower()
                if not lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                filepath = os.path.join(multiview_folder, filename)
                if "front" in lower_name and "front" not in paths:
                    paths["front"] = filepath
                elif "left" in lower_name and "left" not in paths:
                    paths["left"] = filepath
                elif "back" in lower_name and "back" not in paths:
                    paths["back"] = filepath
                elif "right" in lower_name and "right" not in paths:
                    paths["right"] = filepath

        if "front" not in paths:
            return ("", "[]", "skipped: no front image provided")

        # Get asset_id
        asset_id = "hunyuan_output"
        if ASSET_SPEC_JSON:
            try:
                spec_data = _json_loads(ASSET_SPEC_JSON)
                asset_id = spec_data.get("asset_id", "hunyuan_output")
            except Exception:
                pass

        try:
            from .services.hunyuan_adapter import HunyuanAdapter

            wf = workflow_path or None
            adapter = HunyuanAdapter(workflow_path=wf)

            result = adapter.generate(
                front_image_path=paths["front"],
                asset_id=asset_id,
                output_root=config.ASSET_FACTORY_OUTPUT_DIR,
                left_image_path=paths.get("left", ""),
                back_image_path=paths.get("back", ""),
                right_image_path=paths.get("right", ""),
                seed=seed,
                resolution=resolution,
                steps=steps,
                cfg=cfg,
                mesh_algorithm=mesh_algorithm,
                mesh_threshold=mesh_threshold,
                timeout=timeout_seconds,
            )

            model_path = result.get("model_path", "")
            all_paths = json.dumps(result.get("all_output_paths", []))
            status = result.get("status", "unknown")

            if status.startswith("error") and not config.ALLOW_PARTIAL_SUCCESS:
                raise RuntimeError(f"Hunyuan3D failed: {status}")

            return (model_path, all_paths, status)

        except Exception as exc:
            log.error("Hunyuan3D node failed: %s\n%s", exc, traceback.format_exc())
            if config.ALLOW_PARTIAL_SUCCESS:
                return ("", "[]", f"error (continued): {exc}")
            return ("", "[]", f"error: {exc}")


# ==========================================================================
# NODE 6: Blender Asset Processor
# ==========================================================================

class BlenderProcessorNode:
    """Runs Blender in headless mode to process the 3D asset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "MODEL_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "TEXTURE_PATHS_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
                "ASSET_SPEC_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
                "blender_executable": ("STRING", {"default": ""}),
                "optimize_mesh": ("BOOLEAN", {"default": True}),
                "assign_texture": ("BOOLEAN", {"default": True}),
                "generate_previews": ("BOOLEAN", {"default": True}),
                "create_glb": ("BOOLEAN", {"default": True}),
                "create_fbx": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "PROCESSED_MODEL_PATH",
        "PREVIEW_RENDER_PATHS_JSON",
        "BLENDER_REPORT_JSON",
        "STATUS",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = False

    def execute(
        self,
        MODEL_PATH: str = "",
        TEXTURE_PATHS_JSON: str = "{}",
        ASSET_SPEC_JSON: str = "{}",
        blender_executable: str = "",
        optimize_mesh: bool = True,
        assign_texture: bool = True,
        generate_previews: bool = True,
        create_glb: bool = True,
        create_fbx: bool = False,
        enabled: bool = True,
    ):
        empty_report = json.dumps({"success": False, "errors": ["not executed"]})

        if not enabled or not config.ENABLE_BLENDER_PROCESSING:
            return ("", "[]", empty_report, "skipped: Blender disabled")

        if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
            return ("", "[]", empty_report, "skipped: no model file to process")

        try:
            tex_paths = _json_loads(TEXTURE_PATHS_JSON) if TEXTURE_PATHS_JSON else {}
            spec_data = _json_loads(ASSET_SPEC_JSON) if ASSET_SPEC_JSON else {}
        except Exception as exc:
            return ("", "[]", empty_report, f"error: invalid JSON input: {exc}")

        try:
            runner = BlenderRunner(
                blender_executable=blender_executable or None
            )

            report = runner.process_asset(
                model_path=MODEL_PATH,
                texture_paths=tex_paths,
                asset_spec_dict=spec_data,
                output_root=config.ASSET_FACTORY_OUTPUT_DIR,
                optimize_mesh=optimize_mesh,
                assign_texture=assign_texture,
                generate_previews=generate_previews,
                create_glb=create_glb,
                create_fbx=create_fbx,
            )

            report_json = _json_dumps(report)
            previews_json = json.dumps(report.preview_renders)
            processed_path = report.output_model_path

            status = "ok" if report.success else f"error: {'; '.join(report.errors)}"
            return (processed_path, previews_json, report_json, status)

        except Exception as exc:
            log.warning("Blender processing failed (is Blender installed?): %s. Falling back to raw GLB.", exc)
            return (MODEL_PATH, "[]", empty_report, f"error: {exc}")


# ==========================================================================
# NODE 7: Save Manifest
# ==========================================================================

class SaveManifestNode:
    """Creates the comprehensive manifest.json for the asset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ASSET_REQUEST_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
                "ASSET_SPEC_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
                "IMAGE_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "TEXTURE_PATHS_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
                "MODEL_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "PROCESSED_MODEL_PATH": ("STRING", {"forceInput": True, "default": ""}),
                "PREVIEW_RENDER_PATHS_JSON": ("STRING", {"forceInput": True, "default": "[]"}),
                "BLENDER_REPORT_JSON": ("STRING", {"forceInput": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("MANIFEST_PATH", "MANIFEST_JSON", "STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def execute(
        self,
        ASSET_REQUEST_JSON: str = "{}",
        ASSET_SPEC_JSON: str = "{}",
        IMAGE_PATH: str = "",
        TEXTURE_PATHS_JSON: str = "{}",
        MODEL_PATH: str = "",
        PROCESSED_MODEL_PATH: str = "",
        PREVIEW_RENDER_PATHS_JSON: str = "[]",
        BLENDER_REPORT_JSON: str = "{}",
    ):
        try:
            request_data = _json_loads(ASSET_REQUEST_JSON) if ASSET_REQUEST_JSON else {}
            spec_data = _json_loads(ASSET_SPEC_JSON) if ASSET_SPEC_JSON else {}
            tex_paths = _json_loads(TEXTURE_PATHS_JSON) if TEXTURE_PATHS_JSON else {}
            preview_paths = json.loads(PREVIEW_RENDER_PATHS_JSON) if PREVIEW_RENDER_PATHS_JSON else []
            blender_report = _json_loads(BLENDER_REPORT_JSON) if BLENDER_REPORT_JSON else {}
        except Exception as exc:
            return ("", "{}", f"error: JSON parse: {exc}")

        asset_id = spec_data.get("asset_id") or sanitize_asset_id(
            request_data.get("asset_name", "unknown_asset")
        )

        # Collect pipeline stage statuses
        stages: Dict[str, str] = {}
        stages["brief"] = "ok" if request_data else "missing"
        stages["specification"] = "ok" if spec_data.get("asset_id") else "missing"
        stages["concept_image"] = "ok" if IMAGE_PATH else "skipped"
        stages["textures"] = "ok" if tex_paths else "skipped"
        stages["hunyuan_3d"] = "ok" if MODEL_PATH else "skipped"
        stages["blender"] = "ok" if blender_report.get("success") else "skipped"

        # Collect errors/warnings
        errors: List[str] = []
        warnings: List[str] = []
        if blender_report.get("errors"):
            errors.extend(blender_report["errors"])
        if blender_report.get("warnings"):
            warnings.extend(blender_report["warnings"])

        # Build seeds
        seeds = {
            "prompt": request_data.get("seed", 0),
            "concept_image": request_data.get("seed", 0),
            "texture": request_data.get("seed", 0),
            "three_d": request_data.get("seed", 0),
        }

        try:
            writer = ManifestWriter()
            manifest = writer.build_manifest(
                asset_id=asset_id,
                request=request_data,
                specification=spec_data,
                concept_image_path=IMAGE_PATH,
                texture_paths=tex_paths,
                raw_model_path=MODEL_PATH,
                processed_model_path=PROCESSED_MODEL_PATH,
                preview_render_paths=preview_paths if isinstance(preview_paths, list) else [],
                blender_report=blender_report,
                seeds=seeds,
                pipeline_stages=stages,
                errors=errors,
                warnings=warnings,
            )

            manifest_path = writer.save_manifest(
                manifest, config.ASSET_FACTORY_OUTPUT_DIR
            )
            manifest_json = _json_dumps(manifest)

            log.info("Manifest saved: %s", manifest_path)
            return (manifest_path, manifest_json, "ok")

        except Exception as exc:
            log.error("Manifest creation failed: %s\n%s", exc, traceback.format_exc())
            return ("", "{}", f"error: {exc}")


class LocalAssetFactoryPreview3DNode:
    """
    3D GLB Preview Node for LocalAssetFactory.
    Takes a GLB file path string (from LocalHunyuanNode or BlenderProcessorNode)
    and renders an interactive 3D web viewer canvas inside ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"forceInput": True, "tooltip": "GLB model file path string to preview in 3D viewer"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "LocalAssetFactory"

    def execute(self, model_path: str = ""):
        results = []
        if model_path and os.path.isfile(model_path):
            import folder_paths
            import shutil
            output_dir = folder_paths.get_output_directory()
            filename_prefix = "3d/preview"
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
            ext = os.path.splitext(model_path)[1].lstrip(".").lower() or "glb"
            f = f"{filename}_{counter:05}_.{ext}"
            dest = os.path.join(full_output_folder, f)
            shutil.copy(model_path, dest)
            results.append({
                "filename": f,
                "subfolder": subfolder,
                "type": "output"
            })
        return {"ui": {"3d": results}}
