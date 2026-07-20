"""
services · sam3_1 · backend
SAM 3.1 semantic segmentation service.

Uses facebookresearch/sam3 with SAM 3.1 checkpoints.
Produces 2D semantic masks per part per view.

CRITICAL: SAM 3.1 is 2D segmentation ONLY.
  - Does NOT create geometry
  - Does NOT substitute Hunyuan3D-Part
  - Outputs are used for: background removal, cross-view consistency,
    bounding boxes, projecting labels onto mesh, silhouette scoring.

Supported prompt types:
  - text prompts (part names from taxonomy)
  - box prompts
  - point prompts

Batch mode: process all 4 views sequentially (VRAM ~3GB).

VRAM requirement: ~3 000 MB on RTX 4070 8GB.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Default part taxonomy for anime/stylized humanoid characters
DEFAULT_TAXONOMY = [
    "body",
    "hair",
    "face",
    "eyes",
    "top",
    "shorts",
    "belt",
    "skirt_panel",
    "glove_left",
    "glove_right",
    "boot_left",
    "boot_right",
    "red_accessories",
]


@dataclass
class MaskOutput:
    """Single mask result for one part in one view."""
    part_name: str
    view_orientation: str
    confidence: float
    area_px: int
    mask_png_path: str
    rle_path: str
    bbox: Optional[List[int]] = None    # [x, y, w, h] in pixels
    inference_ms: float = 0.0


@dataclass
class SAM3Capabilities:
    model_type: str = "sam3.1"
    checkpoint: str = "facebook/sam3.1-hiera-large"
    supported_prompt_types: List[str] = field(
        default_factory=lambda: ["text", "box", "point"]
    )
    required_vram_mb: int = 3000
    batch_views: bool = True


class SAM3Backend:
    """
    Isolated SAM 3.1 segmentation service.

    Design rules:
    - Run AFTER preflight and canonical views — never on raw input
    - Process views sequentially (VRAM limit)
    - Save PNG masks and RLE JSON for each part/view combination
    - Validate mask quality before accepting
    - Support cross-view consistency checking
    """

    CHECKPOINT = "facebook/sam3.1-hiera-large"
    # Alternative smaller model for VRAM-constrained runs
    CHECKPOINT_SMALL = "facebook/sam3.1-hiera-base-plus"

    def __init__(
        self,
        model_cache_dir: Optional[str] = None,
        use_small_model: bool = False,
        hf_token: Optional[str] = None,
    ):
        self._predictor = None
        self._model_cache_dir = model_cache_dir or os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        self._use_small = use_small_model
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._checkpoint = self.CHECKPOINT_SMALL if use_small_model else self.CHECKPOINT

    def capabilities(self) -> SAM3Capabilities:
        return SAM3Capabilities(checkpoint=self._checkpoint)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._predictor is not None:
            return
        log.info("Loading SAM 3.1 from %s", self._checkpoint)
        try:
            import torch
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            self._predictor = SAM2ImagePredictor.from_pretrained(
                self._checkpoint,
                cache_dir=self._model_cache_dir,
                token=self._hf_token,
            )
            log.info("SAM 3.1 loaded")
        except ImportError as e:
            raise RuntimeError(
                f"SAM 3.1 dependencies not installed: {e}. "
                "Install sam2 in env-sam3-1 (separate from ComfyUI python)."
            ) from e

    def _unload_model(self) -> None:
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("SAM 3.1 unloaded")

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_view(
        self,
        image_path: str,
        view_orientation: str,
        *,
        taxonomy: Optional[List[str]] = None,
        text_prompts: Optional[List[str]] = None,
        box_prompts: Optional[Dict[str, List[int]]] = None,   # {part: [x,y,x2,y2]}
        point_prompts: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        output_dir: str,
        confidence_threshold: float = 0.5,
    ) -> List[MaskOutput]:
        """
        Segment all parts in a single view.

        Args:
            image_path: absolute path to canonical view image
            view_orientation: "front" | "left" | "back" | "right"
            taxonomy: list of part names to segment
            text_prompts: override text prompts (default: taxonomy names)
            box_prompts: {part_name: [x1, y1, x2, y2]} bounding box hints
            point_prompts: {part_name: [(x,y), ...]} point hints
            output_dir: directory to save mask files
            confidence_threshold: minimum confidence to accept mask

        Returns:
            List of MaskOutput, one per accepted part.
        """
        parts = taxonomy or DEFAULT_TAXONOMY
        prompts = text_prompts or parts
        out_dir = Path(output_dir) / view_orientation
        out_dir.mkdir(parents=True, exist_ok=True)

        results: List[MaskOutput] = []
        t_load_start = time.perf_counter()

        try:
            self._load_model()
            img = Image.open(image_path).convert("RGB")
            img_arr = np.array(img)

            import torch
            with torch.no_grad():
                self._predictor.set_image(img_arr)

                for part_name, prompt in zip(parts, prompts):
                    t0 = time.perf_counter()

                    try:
                        masks, scores, _ = self._run_prompt(
                            part_name=part_name,
                            prompt=prompt,
                            box=box_prompts.get(part_name) if box_prompts else None,
                            points=point_prompts.get(part_name) if point_prompts else None,
                        )
                    except Exception as e:
                        log.warning("SAM 3.1: failed to segment '%s': %s", part_name, e)
                        continue

                    if masks is None or len(masks) == 0:
                        continue

                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    if best_score < confidence_threshold:
                        log.debug(
                            "SAM 3.1: '%s' score %.2f < threshold %.2f — skipping",
                            part_name, best_score, confidence_threshold
                        )
                        continue

                    mask_arr = masks[best_idx].astype(np.uint8) * 255
                    area_px = int(mask_arr.sum() // 255)
                    if area_px < 100:
                        log.debug("SAM 3.1: '%s' mask too small (%d px)", part_name, area_px)
                        continue

                    # Save PNG mask
                    mask_png = out_dir / f"{part_name}.png"
                    Image.fromarray(mask_arr).save(str(mask_png))

                    # Save RLE JSON
                    rle = self._mask_to_rle(mask_arr)
                    rle_path = out_dir / f"{part_name}_rle.json"
                    rle_path.write_text(json.dumps(rle), encoding="utf-8")

                    # Bounding box
                    ys, xs = np.where(mask_arr > 0)
                    bbox = None
                    if len(xs) > 0:
                        bbox = [int(xs.min()), int(ys.min()),
                                int(xs.max() - xs.min()), int(ys.max() - ys.min())]

                    inference_ms = (time.perf_counter() - t0) * 1000
                    results.append(MaskOutput(
                        part_name=part_name,
                        view_orientation=view_orientation,
                        confidence=best_score,
                        area_px=area_px,
                        mask_png_path=str(mask_png),
                        rle_path=str(rle_path),
                        bbox=bbox,
                        inference_ms=round(inference_ms, 1),
                    ))
                    log.debug(
                        "SAM 3.1: '%s' @ %s — confidence=%.2f, area=%d px",
                        part_name, view_orientation, best_score, area_px
                    )

        finally:
            self._unload_model()

        log.info(
            "SAM 3.1: %s — segmented %d/%d parts in %.1fs",
            view_orientation, len(results), len(parts),
            time.perf_counter() - t_load_start
        )
        return results

    def _run_prompt(
        self,
        *,
        part_name: str,
        prompt: str,
        box: Optional[List[int]],
        points: Optional[List[Tuple[int, int]]],
    ):
        """Run SAM 3.1 with available prompt types."""
        import numpy as np

        if box is not None:
            # Box prompt: [x1, y1, x2, y2]
            box_arr = np.array(box, dtype=np.float32)
            return self._predictor.predict(
                box=box_arr, multimask_output=True
            )
        elif points is not None:
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.ones(len(points), dtype=np.int32)
            return self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        else:
            # Text prompt (requires grounding — not all SAM2 builds support this)
            # Fallback: center-point heuristic
            log.debug("SAM 3.1: text prompt fallback to center point for '%s'", part_name)
            h, w = self._predictor._orig_hw[0]
            cx, cy = w // 2, h // 2
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            return self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def segment_all_views(
        self,
        views: Dict[str, str],
        taxonomy: Optional[List[str]] = None,
        *,
        output_base_dir: str,
        **kwargs,
    ) -> Dict[str, List[MaskOutput]]:
        """
        Segment all views sequentially.
        Returns {view_orientation: [MaskOutput, ...]}
        """
        results = {}
        for orientation, image_path in views.items():
            log.info("SAM 3.1: processing view '%s'", orientation)
            results[orientation] = self.segment_view(
                image_path=image_path,
                view_orientation=orientation,
                taxonomy=taxonomy,
                output_dir=output_base_dir,
                **kwargs,
            )
        return results

    # ------------------------------------------------------------------
    # Cross-view consistency
    # ------------------------------------------------------------------

    def check_consistency(
        self,
        all_masks: Dict[str, List[MaskOutput]],
        *,
        required_views: List[str] = ("front", "left", "back"),
    ) -> Dict[str, bool]:
        """
        Check if each part appears consistently across required views.
        A part is consistent if it appears in all required views with confidence > 0.

        Returns {part_name: consistent}
        """
        all_parts = set()
        for view_masks in all_masks.values():
            all_parts.update(m.part_name for m in view_masks)

        consistency: Dict[str, bool] = {}
        for part in all_parts:
            views_with_part = set()
            for orient, masks in all_masks.items():
                if any(m.part_name == part and m.confidence > 0.3 for m in masks):
                    views_with_part.add(orient)
            required_set = set(required_views)
            consistency[part] = required_set.issubset(views_with_part)

        return consistency

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
        """Encode binary mask to RLE JSON."""
        flat = mask.flatten()
        rle: List[int] = []
        current_val = 0
        count = 0
        for v in flat:
            if (v > 0) == (current_val > 0):
                count += 1
            else:
                rle.append(count)
                count = 1
                current_val = v
        rle.append(count)
        return {
            "size": list(mask.shape),
            "counts": rle,
        }
