"""
services · hunyuan3d_omni · backend
Hunyuan3D-Omni pose-controlled generation backend.

Supports separate control types — NEVER combine them simultaneously:
  - pose  (MVP — T-pose skeleton control)
  - bbox  (bounding box control)
  - voxel (voxel grid control)
  - point (point cloud control)

Generates SEPARATE candidates per control type.
Does NOT replace Hunyuan3D-2mv — it produces complementary candidates.

VRAM requirement: ~6 000 MB on RTX 4070 8GB.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class OmniCapabilities:
    supported_controls: List[str] = field(
        default_factory=lambda: ["pose", "bbox", "voxel", "point"]
    )
    mvp_control: str = "pose"
    required_vram_mb: int = 6000
    checkpoint: str = "Tencent-Hunyuan/Hunyuan3D-Omni"
    max_parallel_candidates: int = 1


# Default T-pose proportions for a 4.5-head chibi-ish anime character
DEFAULT_TPOSE_PROPORTIONS = {
    "total_heads": 4.5,
    "head_height_ratio": 0.32,       # head height / total height
    "shoulder_width_ratio": 0.28,    # shoulder width / total height
    "arm_abduction_deg": 90.0,       # arms fully out (T-pose)
    "elbow_flexion_deg": 0.0,        # arms straight
    "leg_separation_ratio": 0.12,    # gap between legs / total height
}


class HunyuanOmniBackend:
    """
    Interface to Hunyuan3D-Omni for pose-controlled character generation.

    Key constraints:
    - Generate SEPARATE candidates per control type (pose, bbox, voxel, point)
    - Do NOT combine multiple controls simultaneously
    - Each candidate is labeled with its control type
    - VRAM slot must be held by caller (VRAMSlot context manager)
    """

    CHECKPOINT = "Tencent-Hunyuan/Hunyuan3D-Omni"

    def __init__(self, model_cache_dir: Optional[str] = None):
        self._model = None
        self._model_cache_dir = model_cache_dir or os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )

    def capabilities(self) -> OmniCapabilities:
        return OmniCapabilities()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        log.info("Loading Hunyuan3D-Omni from %s", self.CHECKPOINT)
        try:
            from hy3dgen.shapegen import HunyuanDiTOmniPipeline  # type: ignore
            self._model = HunyuanDiTOmniPipeline.from_pretrained(
                self.CHECKPOINT,
                cache_dir=self._model_cache_dir,
            )
            log.info("Hunyuan3D-Omni loaded")
        except ImportError as e:
            raise RuntimeError(
                f"Hunyuan3D-Omni dependencies not installed: {e}. "
                "Install in env-hunyuan-omni (separate from ComfyUI python)."
            ) from e

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("Hunyuan3D-Omni unloaded")

    # ------------------------------------------------------------------
    # Pose control (MVP — T-pose)
    # ------------------------------------------------------------------

    def generate_pose_candidate(
        self,
        reference_image_path: str,
        *,
        seed: int = 11,
        steps: int = 30,
        output_dir: str,
        job_id: str,
        proportions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate one candidate with pose control (T-pose skeleton).

        This is the MVP control type. Uses a synthetic T-pose skeleton
        derived from configurable character proportions.

        Args:
            reference_image_path: front-view reference image
            seed: RNG seed
            steps: inference steps
            output_dir: where to save raw mesh
            job_id: pipeline job identifier
            proportions: T-pose proportions dict (default: DEFAULT_TPOSE_PROPORTIONS)

        Returns:
            dict with status, mesh_path, control_type="pose", runtime_seconds
        """
        props = proportions or DEFAULT_TPOSE_PROPORTIONS
        result = {
            "status": "pending",
            "mesh_path": "",
            "control_type": "pose",
            "seed": seed,
            "checkpoint": self.CHECKPOINT,
            "proportions": props,
            "runtime_seconds": 0.0,
            "peak_vram_mb": 0,
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"omni_pose_{seed}.glb"

        t_start = time.perf_counter()
        try:
            self._load_model()

            import torch
            with torch.no_grad():
                torch.manual_seed(seed)

                # Build T-pose skeleton control signal
                pose_control = self._build_tpose_skeleton(props)

                from PIL import Image
                ref_img = Image.open(reference_image_path).convert("RGBA")

                log.info(
                    "Omni pose generation: seed=%d, steps=%d, proportions=%s",
                    seed, steps, props
                )

                # Run Omni with pose control ONLY
                mesh = self._model(
                    image=ref_img,
                    pose=pose_control,           # only pose — no bbox/voxel/point
                    num_inference_steps=steps,
                    generator=torch.Generator().manual_seed(seed),
                )
                mesh.export(str(out_path))
                log.info("Omni pose mesh saved: %s", out_path)

            if torch.cuda.is_available():
                result["peak_vram_mb"] = int(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
                torch.cuda.reset_peak_memory_stats()

            result["status"] = "success"
            result["mesh_path"] = str(out_path)

        except RuntimeError as e:
            err = str(e)
            if any(k in err.lower() for k in ("out of memory", "cuda", "vram")):
                result["status"] = f"error_vram: {err[:200]}"
            else:
                result["status"] = f"error: {err[:200]}"
            log.error("Omni pose generation failed: %s", err[:200])
        except Exception as e:
            result["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
            log.error("Omni unexpected error: %s", e)
        finally:
            result["runtime_seconds"] = round(time.perf_counter() - t_start, 2)
            self._unload_model()

        return result

    def _build_tpose_skeleton(self, props: Dict[str, float]) -> Any:
        """
        Build a T-pose skeleton control signal from character proportions.
        Returns a pose representation compatible with Hunyuan3D-Omni's pose param.

        Proportions:
            total_heads: character height in head units
            head_height_ratio: head / total height
            shoulder_width_ratio: shoulder width / total height
            arm_abduction_deg: shoulder abduction angle (90° = T-pose)
            elbow_flexion_deg: elbow angle (0° = straight)
            leg_separation_ratio: leg gap / total height
        """
        import numpy as np

        total_heads = props.get("total_heads", 4.5)
        head_h = props.get("head_height_ratio", 0.32)
        shoulder_w = props.get("shoulder_width_ratio", 0.28)
        arm_abd = np.radians(props.get("arm_abduction_deg", 90.0))
        elbow_flex = np.radians(props.get("elbow_flexion_deg", 0.0))
        leg_sep = props.get("leg_separation_ratio", 0.12)

        # Normalized skeleton keypoints (origin at pelvis, unit height = 1.0)
        # Y+ = up, X+ = right, Z+ = forward
        total_height = 1.0
        head_top = total_height - head_h * 0.1
        head_center_y = total_height - head_h / 2
        neck_y = total_height - head_h
        shoulder_y = neck_y - 0.05

        shoulder_x = shoulder_w / 2
        arm_len = (total_height - shoulder_y) * 0.42
        forearm_len = arm_len * 0.85

        # Elbow position (arm abducted at arm_abd degrees from vertical)
        elbow_x = shoulder_x + arm_len * np.sin(arm_abd)
        elbow_y = shoulder_y - arm_len * np.cos(arm_abd)

        # Wrist
        wrist_x = elbow_x + forearm_len * np.sin(arm_abd + elbow_flex)
        wrist_y = elbow_y - forearm_len * np.cos(arm_abd + elbow_flex)

        hip_y = 0.47 * total_height
        knee_y = 0.22 * total_height
        ankle_y = 0.01 * total_height
        leg_x = leg_sep / 2

        joints = {
            # Head
            "head":         [0.0,  head_center_y, 0.0],
            "neck":         [0.0,  neck_y,         0.0],
            # Spine
            "spine_upper":  [0.0,  (neck_y + hip_y) * 0.7, 0.0],
            "pelvis":       [0.0,  hip_y,           0.0],
            # Arms (mirrored)
            "shoulder_L":   [-shoulder_x, shoulder_y, 0.0],
            "elbow_L":      [-elbow_x,    elbow_y,    0.0],
            "wrist_L":      [-wrist_x,    wrist_y,    0.0],
            "shoulder_R":   [ shoulder_x, shoulder_y, 0.0],
            "elbow_R":      [ elbow_x,    elbow_y,    0.0],
            "wrist_R":      [ wrist_x,    wrist_y,    0.0],
            # Legs
            "hip_L":        [-leg_x, hip_y,   0.0],
            "knee_L":       [-leg_x, knee_y,  0.0],
            "ankle_L":      [-leg_x, ankle_y, 0.0],
            "hip_R":        [ leg_x, hip_y,   0.0],
            "knee_R":       [ leg_x, knee_y,  0.0],
            "ankle_R":      [ leg_x, ankle_y, 0.0],
        }

        log.debug("T-pose skeleton: %d joints, total_heads=%.1f", len(joints), total_heads)
        # Return as dict — Omni API will convert to its internal format
        return joints

    # ------------------------------------------------------------------
    # Stub methods for future control types
    # ------------------------------------------------------------------

    def generate_bbox_candidate(self, *args, **kwargs) -> Dict[str, Any]:
        """BBox control — not implemented in MVP. Returns not_implemented status."""
        log.info("Omni bbox candidate requested — not implemented in MVP")
        return {"status": "not_implemented", "control_type": "bbox"}

    def generate_voxel_candidate(self, *args, **kwargs) -> Dict[str, Any]:
        """Voxel control — not implemented in MVP. Returns not_implemented status."""
        log.info("Omni voxel candidate requested — not implemented in MVP")
        return {"status": "not_implemented", "control_type": "voxel"}

    def generate_point_candidate(self, *args, **kwargs) -> Dict[str, Any]:
        """Point cloud control — not implemented in MVP. Returns not_implemented status."""
        log.info("Omni point candidate requested — not implemented in MVP")
        return {"status": "not_implemented", "control_type": "point"}
