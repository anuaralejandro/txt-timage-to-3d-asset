"""
ComfyUI-LocalAssetFactory · Blender Runner
Executes Blender headless scripts via subprocess for asset processing.

Operations:
- Import 3D model
- Clean geometry, recalculate normals, place pivot
- Optionally simplify mesh
- Assign textures
- Export to GLB / FBX
- Generate preview renders
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from .. import config
from ..schemas import BlenderReport
from ..utilities.file_safety import safe_asset_dir
from ..utilities.logging_utils import get_logger

log = get_logger(__name__)


class BlenderError(RuntimeError):
    """Raised when a Blender subprocess fails fatally."""


class BlenderRunner:
    """Execute Blender scripts in headless mode."""

    def __init__(self, blender_executable: Optional[str] = None):
        self.blender_exe = blender_executable or config.BLENDER_EXECUTABLE
        self._scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "blender",
        )

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_asset(
        self,
        model_path: str,
        texture_paths: Dict[str, str],
        asset_spec_dict: Dict[str, Any],
        output_root: str,
        *,
        optimize_mesh: bool = True,
        assign_texture: bool = True,
        generate_previews: bool = True,
        create_glb: bool = True,
        create_fbx: bool = False,
    ) -> BlenderReport:
        """Full Blender processing pipeline.

        Returns a BlenderReport with results and any errors.
        """
        report = BlenderReport()

        if not config.ENABLE_BLENDER_PROCESSING:
            report.errors.append("Blender processing disabled in config")
            return report

        if not model_path or not os.path.isfile(model_path):
            report.errors.append(f"Model file not found: {model_path}")
            return report

        asset_id = asset_spec_dict.get("asset_id", "unknown")
        output_dir = safe_asset_dir(output_root, asset_id, "processed")
        report.input_model_path = model_path

        # Build the arguments JSON that the Blender script will read
        args = {
            "model_path": model_path,
            "texture_paths": texture_paths,
            "asset_spec": asset_spec_dict,
            "output_dir": output_dir,
            "optimize_mesh": optimize_mesh,
            "assign_texture": assign_texture,
            "generate_previews": generate_previews,
            "create_glb": create_glb,
            "create_fbx": create_fbx,
        }

        # Write args to a temp file
        args_file = os.path.join(output_dir, "_blender_args.json")
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(args, f, indent=2)

        # Run Blender
        script_path = os.path.join(self._scripts_dir, "process_asset.py")
        if not os.path.isfile(script_path):
            report.errors.append(f"Blender script not found: {script_path}")
            return report

        try:
            result = self._run_blender_script(script_path, args_file)
            report_path = os.path.join(output_dir, "_blender_report.json")

            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                report = BlenderReport(**report_data)
            else:
                report.success = result.returncode == 0
                if result.returncode != 0:
                    stderr = result.stderr[:500] if result.stderr else ""
                    report.errors.append(f"Blender exited with code {result.returncode}: {stderr}")

        except Exception as exc:
            report.errors.append(f"Blender execution failed: {exc}")
            log.error("Blender failed: %s", exc)

        # Clean up temp args file
        try:
            os.remove(args_file)
        except OSError:
            pass

        return report

    # ------------------------------------------------------------------
    # Low-level execution
    # ------------------------------------------------------------------

    def _run_blender_script(
        self,
        script_path: str,
        args_file: str,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a Blender Python script in background mode."""
        cmd = [
            self.blender_exe,
            "--background",
            "--python",
            script_path,
            "--",
            args_file,
        ]

        log.info("Running Blender: %s", " ".join(cmd[:4]))

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or config.DEFAULT_TIMEOUT_SECONDS,
        )

    def verify(self) -> Dict[str, Any]:
        """Quick verification that Blender is accessible."""
        try:
            result = subprocess.run(
                [self.blender_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
            return {"available": True, "version": version}
        except Exception as exc:
            return {"available": False, "error": str(exc)}
