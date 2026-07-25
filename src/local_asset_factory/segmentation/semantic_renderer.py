"""
local_asset_factory · segmentation · semantic_renderer
Launches headless Blender script to render 6 or 8 orthographic views of the GLB model,
generating RGB, Depth, World Normals, and 24-bit encoded Face ID maps.
"""

from __future__ import annotations
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure ComfyUI-LocalAssetFactory is in sys.path
_CUSTOM_NODES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ComfyUI_windows_portable" / "ComfyUI" / "custom_nodes" / "ComfyUI-LocalAssetFactory"
if _CUSTOM_NODES_DIR.exists() and str(_CUSTOM_NODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CUSTOM_NODES_DIR))

try:
    from services.blender_runner import BlenderRunner
except ImportError:
    try:
        from ...services.blender_runner import BlenderRunner
    except ImportError:
        BlenderRunner = None  # type: ignore

from .contracts import ViewRenderInfo


log = logging.getLogger(__name__)

class SemanticRenderer:
    """Invokes Blender to render orthographic semantic views."""

    def __init__(self, blender_runner: Optional[BlenderRunner] = None):
        self.runner = blender_runner or BlenderRunner()
        self.script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "..",
            "ComfyUI_windows_portable",
            "ComfyUI",
            "custom_nodes",
            "ComfyUI-LocalAssetFactory",
            "blender",
            "render_semantic_views.py",
        )
        if not os.path.isfile(self.script_path):
            # Fallback path if directory structure differs
            self.script_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "blender",
                    "render_semantic_views.py",
                )
            )

    def render_semantic_views(
        self,
        glb_path: str,
        output_dir: str,
        view_count: int = 8,
        resolution: int = 768,
    ) -> Tuple[List[ViewRenderInfo], str]:
        """
        Executes Blender to render 6 or 8 orthographic views.
        Returns: (list of ViewRenderInfo, render_manifest_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        args_file = os.path.join(output_dir, "_render_views_args.json")
        manifest_file = os.path.join(output_dir, "render_manifest.json")

        args = {
            "glb_path": os.path.abspath(glb_path),
            "output_dir": os.path.abspath(output_dir),
            "view_count": view_count,
            "resolution": resolution,
            "manifest_file": os.path.abspath(manifest_file),
        }

        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(args, f, indent=2)

        # Check script existence
        blender_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "ComfyUI_windows_portable",
                "ComfyUI",
                "custom_nodes",
                "ComfyUI-LocalAssetFactory",
                "blender",
                "render_semantic_views.py",
            )
        )
        if not os.path.isfile(blender_script):
            blender_script = self.script_path

        log.info(f"Running Blender semantic render script: {blender_script}")
        res = self.runner._run_blender_script(blender_script, args_file)

        if res.returncode != 0:
            stderr = res.stderr[:500] if res.stderr else "Unknown Blender error"
            raise RuntimeError(f"Blender semantic rendering failed (code {res.returncode}): {stderr}")

        if not os.path.isfile(manifest_file):
            raise FileNotFoundError(f"Blender rendered successfully but manifest not found at {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        views_list: List[ViewRenderInfo] = []
        for vdata in manifest_data.get("views", []):
            views_list.append(ViewRenderInfo(**vdata))

        return views_list, manifest_file
