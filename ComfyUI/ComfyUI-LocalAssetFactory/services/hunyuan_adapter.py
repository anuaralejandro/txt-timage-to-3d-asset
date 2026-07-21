"""
ComfyUI-LocalAssetFactory · Hunyuan3D-2 Adapter
Wraps the native ComfyUI Hunyuan3D-2 nodes for local image-to-3D conversion.

Supports:
- Single image input (for props/weapons)
- Multi-view input: front, left, back, right (for characters/enemies)
- VRAM-aware: runs geometry generation only, textures handled separately.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import traceback
from typing import Any, Dict, List, Optional

from .. import config
from ..services.comfy_bridge import ComfyBridge, ComfyBridgeError
from ..utilities.file_safety import safe_asset_dir
from ..utilities.logging_utils import get_logger

log = get_logger(__name__)


class HunyuanError(RuntimeError):
    """Raised on unrecoverable Hunyuan3D failures."""


class HunyuanAdapter:
    """Interface to the native Hunyuan3D-2 pipeline via ComfyUI nodes."""

    # Template workflow node IDs (matching hunyuan_multiview_v1.json)
    _NODE_LOAD_FRONT = "10"
    _NODE_LOAD_LEFT = "11"
    _NODE_LOAD_BACK = "12"
    _NODE_LOAD_RIGHT = "13"
    _NODE_CLIP_FRONT = "20"
    _NODE_CLIP_LEFT = "21"
    _NODE_CLIP_BACK = "22"
    _NODE_CLIP_RIGHT = "23"
    _NODE_MV_COND = "30"
    _NODE_CHECKPOINT = "1"
    _NODE_KSAMPLER = "40"
    _NODE_EMPTY_LATENT = "35"
    _NODE_VAE_DECODE = "50"
    _NODE_VOXEL_TO_MESH = "60"
    _NODE_SAVE_3D = "70"

    def __init__(
        self,
        bridge: Optional[ComfyBridge] = None,
        workflow_path: Optional[str] = None,
    ):
        self.bridge = bridge or ComfyBridge()
        self.workflow_path = workflow_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workflows",
            "hunyuan_multiview_v1.json",
        )

    def generate(
        self,
        front_image_path: str,
        asset_id: str,
        output_root: str,
        *,
        left_image_path: str = "",
        back_image_path: str = "",
        right_image_path: str = "",
        seed: int = 0,
        resolution: int = 3072,
        steps: int = 30,
        cfg: float = 5.0,
        mesh_algorithm: str = "surface net",
        mesh_threshold: float = 0.6,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Hunyuan3D-2 image-to-3D and return output info.

        Returns a dict with:
            model_path: str — path to the generated 3D model (.glb)
            all_output_paths: list[str] — all generated files
            status: str — "success" | "skipped" | "error: ..."
        """
        result: Dict[str, Any] = {
            "model_path": "",
            "all_output_paths": [],
            "status": "pending",
        }

        if not config.ENABLE_HUNYUAN:
            result["status"] = "skipped: Hunyuan3D disabled in config"
            log.info("Hunyuan3D generation skipped (disabled).")
            return result

        if not os.path.isfile(front_image_path):
            result["status"] = f"error: front image not found: {front_image_path}"
            log.error(result["status"])
            return result

        # Load workflow template
        try:
            wf = self.bridge.load_workflow(self.workflow_path)
        except FileNotFoundError:
            result["status"] = f"error: Hunyuan workflow not found: {self.workflow_path}"
            log.error(result["status"])
            return result

        # Determine if multi-view
        is_multiview = any([left_image_path, back_image_path, right_image_path])

        # Inject parameters
        wf = self._inject_params(
            wf,
            front_path=front_image_path,
            left_path=left_image_path,
            back_path=back_image_path,
            right_path=right_image_path,
            seed=seed,
            resolution=resolution,
            steps=steps,
            cfg=cfg,
            mesh_algorithm=mesh_algorithm,
            mesh_threshold=mesh_threshold,
            is_multiview=is_multiview,
        )

        # Execute
        output_dir = safe_asset_dir(output_root, asset_id, "raw_3d")
        try:
            history = self.bridge.execute_workflow(wf, timeout=timeout)

            # Retrieve generated 3D files
            files = self.bridge.get_output_files(history)
            saved_paths: List[str] = []

            for f_info in files:
                filename = f_info.get("filename", "")
                if filename:
                    src = self._resolve_comfy_output(
                        filename, f_info.get("subfolder", "")
                    )
                    if src and os.path.isfile(src):
                        dst = os.path.join(output_dir, os.path.basename(filename))
                        shutil.copy2(src, dst)
                        saved_paths.append(dst)
                        log.info("Hunyuan3D output saved: %s", dst)

            # Also check the mesh output directory
            mesh_output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                ))),
                "output", "mesh",
            )
            if os.path.isdir(mesh_output_dir):
                for fname in sorted(os.listdir(mesh_output_dir), reverse=True):
                    if fname.endswith((".glb", ".obj", ".ply")):
                        src = os.path.join(mesh_output_dir, fname)
                        dst = os.path.join(output_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            saved_paths.append(dst)
                            log.info("Hunyuan3D mesh found: %s", dst)
                        break  # take only the latest

            if saved_paths:
                model_path = self._pick_primary_model(saved_paths)
                result["model_path"] = model_path
                result["all_output_paths"] = saved_paths
                result["status"] = "success"
            else:
                result["status"] = "warning: Hunyuan3D completed but no output files found"
                log.warning(result["status"])

        except (ComfyBridgeError, Exception) as exc:
            error_msg = str(exc)
            is_vram = any(
                kw in error_msg.lower()
                for kw in ("out of memory", "cuda", "vram", "allocat")
            )
            if is_vram:
                result["status"] = f"error_vram: {error_msg[:200]}"
                log.error("Hunyuan3D failed due to VRAM: %s", error_msg[:200])
            else:
                result["status"] = f"error: {error_msg[:200]}"
                log.error("Hunyuan3D failed: %s", error_msg[:200])
                log.debug(traceback.format_exc())

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _inject_params(
        self,
        workflow: Dict[str, Any],
        *,
        front_path: str,
        left_path: str,
        back_path: str,
        right_path: str,
        seed: int,
        resolution: int,
        steps: int,
        cfg: float,
        mesh_algorithm: str,
        mesh_threshold: float,
        is_multiview: bool,
    ) -> Dict[str, Any]:
        """Inject parameters into the Hunyuan3D workflow template."""
        wf = copy.deepcopy(workflow)

        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type", "")
            inputs = node.get("inputs", {})

            # Load image nodes
            if ct == "LoadImage":
                import shutil
                import folder_paths
                import os
                input_dir = folder_paths.get_input_directory()
                laf_input = os.path.join(input_dir, "local_asset_factory")
                os.makedirs(laf_input, exist_ok=True)
                
                title = node.get("_meta", {}).get("title", "").lower()
                img_val = str(inputs.get("image", "")).lower()
                target_path = None
                
                if ("front" in title or "front" in img_val or nid == "10") and front_path:
                    target_path = front_path
                elif ("left" in title or "left" in img_val or nid == "11") and left_path:
                    target_path = left_path
                elif ("back" in title or "back" in img_val or nid == "12") and back_path:
                    target_path = back_path
                elif ("right" in title or "right" in img_val or nid == "13") and right_path:
                    target_path = right_path
                    
                if target_path and os.path.isfile(target_path):
                    filename = os.path.basename(target_path)
                    dest_path = os.path.join(laf_input, filename)
                    shutil.copy2(target_path, dest_path)
                    inputs["image"] = f"local_asset_factory/{filename}"
                
                wf[nid]["inputs"] = inputs

            # Empty latent
            if ct == "EmptyLatentHunyuan3Dv2":
                inputs["resolution"] = resolution
                wf[nid]["inputs"] = inputs

            # CLIPVisionEncode needs crop
            if ct == "CLIPVisionEncode":
                inputs["crop"] = "center"
                wf[nid]["inputs"] = inputs

            # KSampler
            if ct in ("KSampler", "KSamplerAdvanced"):
                inputs["seed"] = seed
                inputs["steps"] = steps
                inputs["cfg"] = cfg
                wf[nid]["inputs"] = inputs

            # Voxel to Mesh
            if ct == "VoxelToMesh":
                inputs["algorithm"] = mesh_algorithm
                inputs["threshold"] = mesh_threshold
                wf[nid]["inputs"] = inputs

        return wf

    def _resolve_comfy_output(
        self, filename: str, subfolder: str = ""
    ) -> Optional[str]:
        """Try to resolve a ComfyUI output filename to an absolute path."""
        comfy_base = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
        candidates = [
            os.path.join(comfy_base, "output", subfolder, filename),
            os.path.join(comfy_base, "output", filename),
            os.path.join(comfy_base, "output", "mesh", subfolder, filename),
            os.path.join(comfy_base, "output", "mesh", filename),
            os.path.join(comfy_base, "temp", subfolder, filename),
            os.path.join(comfy_base, "temp", filename),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    @staticmethod
    def _pick_primary_model(paths: List[str]) -> str:
        """Pick the primary 3D model file from a list of paths."""
        preferred = (".glb", ".gltf", ".obj", ".fbx", ".ply")
        for ext in preferred:
            for p in paths:
                if p.lower().endswith(ext):
                    return p
        return paths[0] if paths else ""
