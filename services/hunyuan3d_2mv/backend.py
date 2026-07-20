"""
services · hunyuan3d_2mv · backend
Hunyuan3D-2mv geometry backend.

Checkpoint: tencent/Hunyuan3D-2mv
Official example documents: front, left, back views.
Right view is reserved for QA only (not fed to checkpoint).

Interface:
    class Hunyuan2MVBackend:
        def healthcheck(self) -> HealthStatus
        def capabilities(self) -> Capabilities
        def generate(self, request: GeometryRequest) -> GeometryCandidate

VRAM requirement: ~6 000 MB on RTX 4070 8GB.
Load model → inference → unload before next service.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    healthy: bool
    message: str = ""
    cuda_available: bool = False
    vram_free_mb: Optional[int] = None
    checkpoint_loaded: bool = False


@dataclass
class Capabilities:
    """
    Reports what this backend actually supports.
    Callers MUST check supported_views before feeding data.
    Per official Hunyuan3D-2mv docs, only front/left/back are documented.
    """
    supported_views: List[str] = field(default_factory=lambda: ["front", "left", "back"])
    supported_variants: List[str] = field(default_factory=lambda: ["normal", "turbo"])
    required_vram_mb: int = 6000
    supported_output_formats: List[str] = field(default_factory=lambda: ["glb", "obj"])
    checkpoint: str = "tencent/Hunyuan3D-2mv"
    subfolder: str = "hunyuan3d-dit-v2-mv"
    max_parallel_candidates: int = 1    # RTX 4070: always 1


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Hunyuan2MVBackend:
    """
    Interface to Hunyuan3D-2mv for multiview character reconstruction.

    Design rules:
    - capabilities() is ALWAYS called before generate() by the pipeline.
    - Only views listed in capabilities().supported_views are fed to the model.
    - right view is NEVER fed to the model (QA only).
    - Each raw mesh is saved without modification.
    - OOM errors are caught and reported as status="error_vram".
    """

    CHECKPOINT = "tencent/Hunyuan3D-2mv"
    SUBFOLDER_NORMAL = "hunyuan3d-dit-v2-mv"
    SUBFOLDER_TURBO = "hunyuan3d-dit-v2-mv-turbo"

    def __init__(self, model_cache_dir: Optional[str] = None):
        self._model = None
        self._model_cache_dir = model_cache_dir or os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        self._loaded_variant: Optional[str] = None

    # ------------------------------------------------------------------
    # Health and capabilities
    # ------------------------------------------------------------------

    def healthcheck(self) -> HealthStatus:
        """Check if backend is operational."""
        status = HealthStatus(healthy=False)
        try:
            import torch
            status.cuda_available = torch.cuda.is_available()
            if status.cuda_available:
                free, total = torch.cuda.mem_get_info()
                status.vram_free_mb = free // (1024 * 1024)
            status.checkpoint_loaded = self._model is not None
            status.healthy = status.cuda_available
            status.message = (
                f"CUDA: {status.cuda_available}, "
                f"VRAM free: {status.vram_free_mb} MB, "
                f"Model loaded: {status.checkpoint_loaded}"
            )
        except ImportError:
            status.message = "torch not available — cannot run Hunyuan3D-2mv"
        return status

    def capabilities(self) -> Capabilities:
        """
        Return capabilities of this backend.
        Pipeline MUST call this before generate() to know supported_views.
        """
        return Capabilities(
            supported_views=["front", "left", "back"],
            supported_variants=["normal", "turbo"],
            required_vram_mb=6000,
            supported_output_formats=["glb", "obj"],
            checkpoint=self.CHECKPOINT,
            subfolder=self.SUBFOLDER_NORMAL,
            max_parallel_candidates=1,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, variant: str = "normal") -> None:
        """
        Load the Hunyuan3D-2mv model into GPU memory.
        Call only when VRAM slot is held.
        """
        if self._model is not None and self._loaded_variant == variant:
            return  # already loaded

        subfolder = (
            self.SUBFOLDER_TURBO if variant == "turbo" else self.SUBFOLDER_NORMAL
        )
        log.info("Loading Hunyuan3D-2mv %s from %s/%s", variant, self.CHECKPOINT, subfolder)

        try:
            # Import deferred: only when actually running
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
            self._model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.CHECKPOINT,
                subfolder=subfolder,
                cache_dir=self._model_cache_dir,
            )
            self._model.enable_flashattn()  # memory efficiency
            self._loaded_variant = variant
            log.info("Hunyuan3D-2mv loaded (%s variant)", variant)
        except ImportError as e:
            raise RuntimeError(
                f"Hunyuan3D-2mv dependencies not installed: {e}. "
                "Install hy3dgen in env-hunyuan2mv (separate from ComfyUI python)."
            ) from e

    def _unload_model(self) -> None:
        """Remove model from GPU memory. Always call after generation."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_variant = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("Hunyuan3D-2mv unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        views: Dict[str, str],
        *,
        seed: int = 11,
        steps: int = 30,
        variant: str = "normal",
        output_dir: str,
        job_id: str,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """
        Generate a single raw mesh from multiview images.

        Args:
            views: {orientation: absolute_path} — ONLY checkpoint-supported views.
                   Pipeline must have called capabilities() to filter views.
                   right view must NOT be in this dict.
            seed: RNG seed
            steps: inference steps (30 or 40)
            variant: "normal" | "turbo"
            output_dir: directory to write raw mesh
            job_id: pipeline job identifier
            timeout_seconds: hard timeout

        Returns:
            dict with:
                status: "success" | "error_vram" | "error: ..."
                mesh_path: path to saved GLB (raw, unmodified)
                runtime_seconds: float
                peak_vram_mb: int
                seed: int
                variant: str
        """
        result = {
            "status": "pending",
            "mesh_path": "",
            "runtime_seconds": 0.0,
            "peak_vram_mb": 0,
            "seed": seed,
            "variant": variant,
            "checkpoint": self.CHECKPOINT,
        }

        # Validate: right view must not be in views
        if "right" in views:
            log.warning(
                "generate() received 'right' view — this is not supported by the checkpoint. "
                "Removing it. Use right view for QA/scoring only."
            )
            views = {k: v for k, v in views.items() if k != "right"}

        # Validate required views present
        caps = self.capabilities()
        for required in caps.supported_views:
            if required not in views:
                result["status"] = f"error: missing required view '{required}'"
                return result

        t_start = time.perf_counter()
        out_path = Path(output_dir) / f"raw_{seed}_{variant}.glb"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._load_model(variant)

            import torch
            with torch.no_grad():
                # Set seed for reproducibility
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Load images
                from PIL import Image
                pil_views = {
                    orient: Image.open(path).convert("RGBA")
                    for orient, path in views.items()
                    if Path(path).exists()
                }

                log.info(
                    "Generating with seed=%d, steps=%d, variant=%s, views=%s",
                    seed, steps, variant, list(pil_views.keys())
                )

                # Run pipeline
                mesh = self._model(
                    image=pil_views,
                    num_inference_steps=steps,
                    generator=torch.Generator().manual_seed(seed),
                )

                # Export raw GLB — NEVER modify before saving
                mesh.export(str(out_path))
                log.info("Raw mesh saved: %s", out_path)

            # VRAM peak
            if torch.cuda.is_available():
                result["peak_vram_mb"] = int(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
                torch.cuda.reset_peak_memory_stats()

            result["status"] = "success"
            result["mesh_path"] = str(out_path)

        except RuntimeError as e:
            err_str = str(e)
            is_oom = any(k in err_str.lower() for k in ("out of memory", "cuda", "vram"))
            if is_oom:
                result["status"] = f"error_vram: {err_str[:300]}"
                log.error("OOM in Hunyuan3D-2mv (seed=%d, variant=%s): %s", seed, variant, err_str[:200])
            else:
                result["status"] = f"error: {err_str[:300]}"
                log.error("Hunyuan3D-2mv failed: %s", err_str[:200])
        except Exception as e:
            result["status"] = f"error: {type(e).__name__}: {str(e)[:300]}"
            log.error("Hunyuan3D-2mv unexpected error: %s", e)
        finally:
            result["runtime_seconds"] = round(time.perf_counter() - t_start, 2)
            self._unload_model()

        return result

    def generate_batch(
        self,
        views: Dict[str, str],
        *,
        seeds: List[int] = (11, 29, 47, 83),
        steps_list: List[int] = (30, 40),
        variants: List[str] = ("normal", "turbo"),
        output_dir: str,
        job_id: str,
        on_candidate: Optional[Any] = None,    # callback(result) after each
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple candidates (seeds × steps × variants), sequentially.
        One at a time to respect RTX 4070 8GB VRAM limit.

        on_candidate: optional callback called after each candidate is generated.
        """
        results = []
        for variant in variants:
            for steps in steps_list:
                for seed in seeds:
                    log.info("Batch: seed=%d, steps=%d, variant=%s", seed, steps, variant)
                    r = self.generate(
                        views=views,
                        seed=seed,
                        steps=steps,
                        variant=variant,
                        output_dir=str(Path(output_dir) / f"candidate_{seed}_{steps}_{variant}"),
                        job_id=job_id,
                    )
                    results.append(r)
                    if on_candidate:
                        try:
                            on_candidate(r)
                        except Exception:
                            pass
        return results
