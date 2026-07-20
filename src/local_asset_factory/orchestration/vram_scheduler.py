"""
local_asset_factory · orchestration · vram_scheduler
Sequential VRAM scheduler for RTX 4070 8GB.

Each ML service must load, execute, then unload before the next.
This scheduler enforces that contract and logs VRAM usage.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

log = logging.getLogger(__name__)

# RTX 4070 Laptop VRAM limits
VRAM_TOTAL_MB = 8192
VRAM_SAFETY_HEADROOM_MB = 512   # Keep free for system/driver overhead


def _try_get_vram_free_mb() -> Optional[int]:
    """Query free VRAM using torch if available."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except ImportError:
        pass
    return None


def _try_empty_cache() -> None:
    """Release unused VRAM via torch if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


class VRAMSlot:
    """
    Context manager for a single VRAM-consuming service.

    Usage:
        with VRAMSlot("sam3_1", required_mb=3000) as slot:
            result = sam_service.run(...)
    """

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

    def __enter__(self) -> "VRAMSlot":
        log.info("[VRAM] Loading %s (requires ~%d MB)", self.service_name, self.required_mb)

        # Empty cache before loading
        _try_empty_cache()

        free = _try_get_vram_free_mb()
        if free is not None and self.required_mb > 0:
            available = free - VRAM_SAFETY_HEADROOM_MB
            if self.required_mb > available:
                log.warning(
                    "[VRAM] %s requires %d MB but only %d MB available (after headroom). "
                    "Will attempt anyway — may OOM.",
                    self.service_name, self.required_mb, available
                )
            else:
                log.info("[VRAM] OK: %d MB available for %s", available, self.service_name)

        if self.on_before_load:
            self.on_before_load()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start_time

        if exc_type is not None:
            log.error("[VRAM] %s failed after %.1fs: %s", self.service_name, elapsed, exc_val)
        else:
            log.info("[VRAM] %s completed in %.1fs", self.service_name, elapsed)

        # Always unload — even on failure
        _try_empty_cache()

        if self.on_after_unload:
            try:
                self.on_after_unload()
            except Exception as e:
                log.warning("[VRAM] on_after_unload failed: %s", e)

        free = _try_get_vram_free_mb()
        if free is not None:
            log.info("[VRAM] After unload: %d MB free", free)

        # Do not suppress exceptions
        return False


class VRAMScheduler:
    """
    Schedules sequential ML service execution with automatic VRAM management.

    All services run one at a time. Each service is expected to:
    1. Load its model on enter
    2. Execute inference
    3. Unload its model on exit

    This class does NOT load/unload models directly — it enforces sequential
    execution and provides VRAMSlot context managers to callers.

    VRAM budget per service (approximate for RTX 4070 8GB):
        SAM 3.1:             ~3 000 MB
        Hunyuan3D-2mv:       ~6 000 MB
        Hunyuan3D-Omni:      ~6 000 MB
        Hunyuan3D-Part:      ~4 000 MB
        Hunyuan3D-Paint:     ~6 000 MB
        RigAnything:         ~2 000 MB
    """

    VRAM_ESTIMATES_MB: dict[str, int] = {
        "sam3_1": 3000,
        "hunyuan3d_2mv": 6000,
        "hunyuan3d_omni": 6000,
        "hunyuan3d_part": 4000,
        "hunyuan3d_paint": 6000,
        "riganything": 2000,
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
        """
        Get a VRAMSlot context manager for the given service.
        Required VRAM is looked up from VRAM_ESTIMATES_MB if not provided.
        """
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
