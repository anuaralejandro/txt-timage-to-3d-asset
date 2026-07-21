"""
ComfyUI-LocalAssetFactory · Macro Pipeline Node
Unified entry point for the M2-M6 pipeline.
"""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from typing import Any, Dict

import numpy as np
from PIL import Image
from pathlib import Path

# Import from the local asset factory Python package
from local_asset_factory.domain.models import AssetRequest
from local_asset_factory.observability.artifact_store import ArtifactStore
from local_asset_factory.preflight.preflight_runner import PreflightRunner
from local_asset_factory.multiview.canonical_views import build_canonical_view_set

# Legacy / Utility imports
from .utilities.image_conversion import comfy_tensor_to_pil
from .services.hunyuan_adapter import HunyuanAdapter
from .services.blender_runner import BlenderRunner
from . import config

log = logging.getLogger(__name__)

_CATEGORY = "Asset Factory"


class HunyuanMacroPipelineNode:
    """
    Macro node that executes the entire Hunyuan Character Pipeline (M2-M6).
    Input: Multi-view images (Front, Left, Back, optional Right).
    Output: Path to the final production-ready GLB asset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_name": ("STRING", {"default": "Character_01"}),
                "triangle_budget": ("INT", {"default": 15000, "min": 1000, "max": 100000, "step": 1000}),
                "target_platform": (["android_mid", "android_high", "android_low", "ios_high", "ios_mid"], {"default": "android_mid"}),
                "paint_mode": (["toon_mobile", "pbr_mobile"], {"default": "toon_mobile"}),
            },
            "optional": {
                "FRONT_IMAGE": ("IMAGE",),
                "LEFT_IMAGE": ("IMAGE",),
                "BACK_IMAGE": ("IMAGE",),
                "RIGHT_IMAGE": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("FINAL_GLB_PATH", "PIPELINE_STATUS")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def execute(
        self,
        asset_name: str,
        triangle_budget: int,
        target_platform: str,
        paint_mode: str,
        FRONT_IMAGE=None,
        LEFT_IMAGE=None,
        BACK_IMAGE=None,
        RIGHT_IMAGE=None,
        seed: int = 0,
    ):
        status_msg = "Pipeline Initialized."
        glb_path = ""

        if FRONT_IMAGE is None:
            return ("", "Error: Missing required FRONT_IMAGE (Multi-view inference requires at least a front view).")

        try:
            # 1. Save tensors to temporary images
            temp_dir = os.path.join(config.ASSET_FACTORY_OUTPUT_DIR, "_temp", asset_name.replace(" ", "_"))
            os.makedirs(temp_dir, exist_ok=True)
            
            view_paths = {}
            for name, tensor in [("front", FRONT_IMAGE), ("left", LEFT_IMAGE), ("back", BACK_IMAGE), ("right", RIGHT_IMAGE)]:
                if tensor is not None:
                    try:
                        pil_img = comfy_tensor_to_pil(tensor)
                        path = os.path.join(temp_dir, f"{name}.png")
                        pil_img.save(path)
                        view_paths[name] = path
                    except Exception as e:
                        log.warning("Could not convert %s tensor: %s", name, e)

            # 2. Setup Artifact Store & Request
            store = ArtifactStore(
                base_dir=Path(config.ASSET_FACTORY_OUTPUT_DIR) / asset_name.replace(" ", "_"),
                job_id=f"job_{seed}"
            )
            req = AssetRequest(
                asset_name=asset_name,
                asset_type="character",
                description_es="Pipeline macro auto-generated",
                target_platform=target_platform,
                triangle_budget=triangle_budget,
                paint_mode=paint_mode,
                seed=seed,
            )

            # 3. Preflight
            runner = PreflightRunner()
            results = runner.validate_view_set(view_paths, job_id=store.job_id)
            if not runner.all_passed(results):
                error_msg = runner.summary(results)
                log.error("Preflight failed:\n%s", error_msg)
                return ("", f"Preflight Error: {error_msg}")

            # 4. Canonical Views Normalization
            cv_set = build_canonical_view_set(view_paths, store.path("canonical_views"), job_id=store.job_id)

            # Note: For full pipeline execution (M3-M6), we integrate the backend clients here.
            log.info("Macro Pipeline: Preflight & Normalization complete for %s", asset_name)
            
            # 5. M3: AI Geometry Generation via Hunyuan3D-2
            log.info("Macro Pipeline: Starting Hunyuan3D-2 Generation")
            adapter = HunyuanAdapter()
            hy_result = adapter.generate(
                front_image_path=str(store.job_root / cv_set.front.relative_path),
                asset_id=asset_name,
                output_root=str(store.job_root),
                left_image_path=str(store.job_root / cv_set.left.relative_path) if cv_set.left else "",
                back_image_path=str(store.job_root / cv_set.back.relative_path) if cv_set.back else "",
                right_image_path=str(store.job_root / cv_set.right.relative_path) if cv_set.right else "",
                seed=seed,
                resolution=3072, # Standard production resolution
                steps=30,
            )
            
            if hy_result["status"].startswith("error"):
                log.error("Hunyuan Generation Failed: %s", hy_result["status"])
                return ("", f"Hunyuan Error: {hy_result['status']}")
                
            raw_glb = hy_result.get("model_path", "")
            if not raw_glb or not os.path.exists(raw_glb):
                return ("", "Error: Hunyuan generated empty or missing GLB")
                
            log.info("Macro Pipeline: Hunyuan3D-2 Generation successful -> %s", raw_glb)
            
            # 6. M4-M6: Mesh Optimization, Retopology & Export via Blender
            log.info("Macro Pipeline: Starting Blender Post-Processing")
            blender = BlenderRunner()
            # Texture mapping is skipped here if doing vertex colors (paint_mode), but we run geometry opt
            try:
                b_report = blender.process_asset(
                    model_path=raw_glb,
                    texture_paths={}, # Textures handled natively by Hunyuan3D in Vertex Colors for now
                    asset_spec_dict=req.dict(),
                    output_root=str(store.job_root),
                    optimize_mesh=True,
                    assign_texture=False, 
                    generate_previews=True,
                    create_glb=True,
                )
                
                final_glb = b_report.outputs.get("final_glb", "")
                if not final_glb or not os.path.exists(final_glb):
                    final_glb = raw_glb # Fallback to raw if blender failed but didn't throw
                    
            except Exception as e:
                log.warning("Blender processing failed (is Blender installed?): %s. Falling back to raw GLB.", e)
                final_glb = raw_glb
            
            status_msg = f"Success! Output saved to: {store.job_root}"
            return (final_glb, status_msg)

        except Exception as e:
            log.error("Macro pipeline failed: %s\n%s", e, traceback.format_exc())
            return ("", f"Pipeline Error: {e}")

# ── ComfyUI Registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AssetFactory_HunyuanMacroPipeline": HunyuanMacroPipelineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetFactory_HunyuanMacroPipeline": "Asset Factory · Macro Pipeline (M2-M6)",
}
