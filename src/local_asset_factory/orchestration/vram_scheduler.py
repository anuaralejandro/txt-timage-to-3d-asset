"""
local_asset_factory · orchestration · vram_scheduler
Sequential VRAM scheduler for RTX 4070 8GB.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

log = logging.getLogger(__name__)

VRAM_TOTAL_MB = 8192
VRAM_SAFETY_HEADROOM_MB = 512

def _try_get_vram_free_mb() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except ImportError:
        pass
    return None

def _try_empty_cache() -> None:
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

class VRAMSlot:
    def __init__(
        self,
        service_name: str,
        required_mb: int = 0,
        *,
        on_before_load: Optional[Callable] = None,
        on_after_unload: Optional[Callable] = None,
    ):
        self.service_name = service_name
        self.required_mb = required_mb
        self.on_before_load = on_before_load
        self.on_after_unload = on_after_unload
        self._start_time: float = 0.0
        self._peak_vram_mb: int = 0

    def __enter__(self) -> "VRAMSlot":
        log.info("[VRAM] Loading %s (requires ~%d MB)", self.service_name, self.required_mb)
        _try_empty_cache()
        free = _try_get_vram_free_mb()
        
        if free is not None and self.required_mb > 0:
            available = free - VRAM_SAFETY_HEADROOM_MB
            if self.required_mb > available:
                log.warning("[VRAM] %s requires %d MB but only %d MB available. Risk of OOM.", self.service_name, self.required_mb, available)
            else:
                log.info("[VRAM] OK: %d MB available for %s", available, self.service_name)

        if self.on_before_load:
            self.on_before_load()

        self._start_time = time.perf_counter()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start_time
        
        import torch
        if torch.cuda.is_available():
            self._peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

        if exc_type is not None:
            log.error("[VRAM] %s failed after %.1fs: %s. Peak VRAM: %d MB", self.service_name, elapsed, exc_val, self._peak_vram_mb)
        else:
            log.info("[VRAM] %s completed in %.1fs. Peak VRAM: %d MB", self.service_name, elapsed, self._peak_vram_mb)

        _try_empty_cache()

        if self.on_after_unload:
            try:
                self.on_after_unload()
            except Exception as e:
                log.warning("[VRAM] on_after_unload failed: %s", e)

        free = _try_get_vram_free_mb()
        if free is not None:
            log.info("[VRAM] After unload: %d MB free", free)

        return False

class VRAMScheduler:
    VRAM_ESTIMATES_MB: dict[str, int] = {
        "sam3_1": 3000,
        "hunyuan3d_2mv_shape_standard": 6500,
        "hunyuan3d_2mv_shape_turbo": 6000,
        "render_validation": 1000,
        "paint": 6000,
        "blender_postprocess": 1500,
    }

    def __init__(self):
        self._active_service: Optional[str] = None

    def slot(
        self,
        service_name: str,
        *,
        required_mb: Optional[int] = None,
        on_before_load: Optional[Callable] = None,
        on_after_unload: Optional[Callable] = None,
    ) -> VRAMSlot:
        if required_mb is None:
            required_mb = self.VRAM_ESTIMATES_MB.get(service_name, 0)

        return VRAMSlot(
            service_name,
            required_mb=required_mb,
            on_before_load=on_before_load,
            on_after_unload=on_after_unload,
        )

    def report_status(self) -> dict[str, Any]:
        free = _try_get_vram_free_mb()
        return {
            "vram_total_mb": VRAM_TOTAL_MB,
            "vram_free_mb": free,
            "vram_used_mb": (VRAM_TOTAL_MB - free) if free is not None else None,
            "active_service": self._active_service,
        }
