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

import json
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
    """
    def __init__(self, model_cache_dir: Optional[str] = None):
        self._model = None
        self._model_cache_dir = model_cache_dir or os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        self._loaded_variant: Optional[str] = None
        self._loaded_checkpoint: Optional[str] = None

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
        return Capabilities()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: str, variant: str, attention_backend: str, enable_cpu_offload: bool) -> None:
        """
        Load the Hunyuan3D-2mv model into GPU memory.
        """
        if self._model is not None and self._loaded_variant == variant and self._loaded_checkpoint == checkpoint:
            return  # already loaded

        subfolder = "hunyuan3d-dit-v2-mv-turbo" if variant == "turbo" else "hunyuan3d-dit-v2-mv"
        log.info("Loading Hunyuan3D-2mv %s from %s/%s", variant, checkpoint, subfolder)

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
            import torch
            
            self._model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                checkpoint,
                subfolder=subfolder,
                cache_dir=self._model_cache_dir,
                torch_dtype=torch.float16,
            )
            
            if attention_backend == "flash_attn":
                try:
                    self._model.enable_flashattn()
                except Exception as e:
                    log.warning("Flash Attention not available: %s", e)
            
            if enable_cpu_offload:
                try:
                    self._model.enable_model_cpu_offload()
                except AttributeError:
                    log.warning("Model CPU offload not supported by pipeline version")
            else:
                self._model.to("cuda")

            self._loaded_variant = variant
            self._loaded_checkpoint = checkpoint
            log.info("Hunyuan3D-2mv loaded (%s variant)", variant)
        except ImportError as e:
            raise RuntimeError(
                f"Hunyuan3D-2mv dependencies not installed: {e}."
            ) from e

    def _unload_model(self) -> None:
        """Remove model from GPU memory. Always call after generation."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_variant = None
            self._loaded_checkpoint = None
        try:
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()
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
        model_id: str = "tencent/Hunyuan3D-2mv",
        model_variant: str = "normal",
        seed: int = 11,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        octree_resolution: int = 380,
        num_chunks: int = 20000,
        output_type: str = "trimesh",
        device: str = "cuda",
        dtype: str = "float16",
        low_vram_mode: bool = True,
        enable_cpu_offload: bool = True,
        attention_backend: str = "flash_attn",
        output_dir: str,
        job_id: str,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """
        Generate a single raw mesh from multiview images.
        """
        result = {
            "status": "pending",
            "mesh_path": "",
            "runtime_seconds": 0.0,
            "peak_vram_mb": 0,
            "seed": seed,
            "model_variant": model_variant,
            "model_id": model_id,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "octree_resolution": octree_resolution,
                "num_chunks": num_chunks,
                "output_type": output_type,
            },
            "warnings": [],
            "error": None,
            "model_hash": "unknown_hash",
            "input_manifest_path": "",
        }

        caps = self.capabilities()
        checkpoint_views = {k: v for k, v in views.items() if k in caps.supported_views}
        unsupported_views = {k: v for k, v in views.items() if k not in caps.supported_views}

        for required in caps.supported_views:
            if required not in views:
                result["status"] = f"error: missing required view '{required}'"
                result["error"] = result["status"]
                return result

        if unsupported_views:
            msg = f"Unsupported views excluded from model inference: {list(unsupported_views.keys())}"
            log.info(msg)
            result["warnings"].append(msg)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / f"manifest_{seed}.json"
        
        # Save manifest
        manifest = {
            "job_id": job_id,
            "seed": seed,
            "views_provided": views,
            "views_used_by_model": checkpoint_views,
            "parameters": result["parameters"]
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result["input_manifest_path"] = str(manifest_path)

        t_start = time.perf_counter()
        out_path = out_dir / f"raw_{seed}_{model_variant}.glb"

        try:
            self._load_model(model_id, model_variant, attention_backend, enable_cpu_offload)

            import torch
            with torch.no_grad():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                from PIL import Image
                pil_views = {
                    orient: Image.open(path).convert("RGBA")
                    for orient, path in checkpoint_views.items()
                    if Path(path).exists()
                }

                # We must enforce order if possible, though dicts keep insertion order.
                # The model typically expects: front, left, back... but the PIL dict should be fine if matched by keys internally by the pipeline.

                mesh = self._model(
                    image=pil_views,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution,
                    mc_algo="marching_cubes",
                    generator=torch.Generator().manual_seed(seed),
                )
                
                # Check output type and validate
                if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                    # Some pipelines return a wrapper or list
                    if isinstance(mesh, list) and len(mesh) > 0:
                        mesh = mesh[0]
                
                # Validate vertices and faces
                if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                    raise ValueError("Generated mesh has no vertices.")
                if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                    raise ValueError("Generated mesh has no faces.")
                
                mesh.export(str(out_path))
                log.info("Raw mesh saved: %s", out_path)

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
                result["error"] = result["status"]
                log.error("OOM in Hunyuan3D-2mv (seed=%d, variant=%s): %s", seed, model_variant, err_str[:200])
            else:
                result["status"] = f"error: {err_str[:300]}"
                result["error"] = result["status"]
                log.error("Hunyuan3D-2mv failed: %s", err_str[:200])
        except Exception as e:
            result["status"] = f"error: {type(e).__name__}: {str(e)[:300]}"
            result["error"] = result["status"]
            log.error("Hunyuan3D-2mv unexpected error: %s", e)
        finally:
            result["runtime_seconds"] = round(time.perf_counter() - t_start, 2)
            self._unload_model()

        return result
