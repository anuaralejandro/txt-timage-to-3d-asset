"""
local_asset_factory · segmentation · memory
Model Lifecycle Manager for 8GB VRAM allocation control, lazy loading, context cleanup, and CUDA VRAM logging.
"""

from __future__ import annotations
import gc
import logging
import time
from typing import Any, Optional, Generator
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .contracts import VRAMLogEntry

log = logging.getLogger(__name__)

class ModelLifecycleManager:
    """Manages lazy model loading, strict VRAM unloading, and memory logging."""

    def __init__(self, low_vram_mode: bool = True):
        self.low_vram_mode = low_vram_mode
        self._current_backend_name: Optional[str] = None
        self._vram_logs: list[VRAMLogEntry] = []

    def start_phase(self, phase_name: str) -> float:
        """Resets peak VRAM metrics and records start time."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        log.info(f"--- Starting Phase: {phase_name} ---")
        return time.time()

    def end_phase(self, phase_name: str, start_time: float) -> VRAMLogEntry:
        """Cleans memory, collects peak VRAM allocated, and logs performance."""
        elapsed = time.time() - start_time
        peak_alloc = 0.0
        peak_res = 0.0
        gpu_name = "CPU"

        if TORCH_AVAILABLE and torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)

        entry = VRAMLogEntry(
            phase=phase_name,
            gpu_name=gpu_name,
            peak_allocated_mb=peak_alloc,
            peak_reserved_mb=peak_res,
            elapsed_sec=elapsed,
        )
        self._vram_logs.append(entry)
        log.info(
            f"--- Completed Phase: {phase_name} in {elapsed:.2f}s | "
            f"Peak VRAM: {peak_alloc:.1f} MB (Allocated), {peak_res:.1f} MB (Reserved) ---"
        )
        return entry

    def unload_model(self, model: Any) -> None:
        """Moves model to CPU, deletes reference, forces GC and clears CUDA cache."""
        if model is None:
            return

        try:
            if hasattr(model, "to") and TORCH_AVAILABLE:
                model.to("cpu")
        except Exception as e:
            log.warning(f"Could not move model to CPU: {e}")

        del model
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def execution_context(self) -> Generator[None, None, None]:
        """Provides torch.inference_mode() when torch is available."""
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                yield
        else:
            yield

    def get_logs(self) -> list[VRAMLogEntry]:
        return list(self._vram_logs)
