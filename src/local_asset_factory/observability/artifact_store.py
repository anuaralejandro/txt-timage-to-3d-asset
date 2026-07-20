"""
local_asset_factory · observability · artifact_store
Atomic, hash-verified, resume-capable artifact store.

All paths stored are RELATIVE to the job root.
Structure:
    artifacts/<job_id>/
    ├── request.json
    ├── input/
    ├── normalized/
    ├── views/
    ├── masks_2d/
    ├── candidates/
    │   ├── hunyuan2mv/
    │   └── omni/
    ├── selected/
    ├── parts_3d/
    ├── topology/
    ├── uv/
    ├── textures/
    ├── rig/
    ├── animation_qa/
    ├── lod/
    ├── exports/
    ├── previews/
    ├── metrics.json
    ├── manifest.json
    └── logs/
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..domain.models import ArtifactMeta, PipelineManifest, sha256_file

log = logging.getLogger(__name__)

# Subdirectory layout
JOB_SUBDIRS = [
    "input",
    "normalized",
    "views",
    "masks_2d",
    "candidates/hunyuan2mv",
    "candidates/omni",
    "selected",
    "parts_3d",
    "topology",
    "uv",
    "textures",
    "rig",
    "animation_qa",
    "lod",
    "exports",
    "previews",
    "logs",
]


class ArtifactStore:
    """
    Job-scoped artifact store with atomic writes, SHA-256 verification,
    resume support, and cancellation token.
    """

    def __init__(self, base_dir: str | Path, job_id: str):
        self.job_root = Path(base_dir) / "artifacts" / job_id
        self.job_id = job_id
        self._cancelled = False
        self._manifest: Optional[PipelineManifest] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create job directory structure."""
        self.job_root.mkdir(parents=True, exist_ok=True)
        for sub in JOB_SUBDIRS:
            (self.job_root / sub).mkdir(parents=True, exist_ok=True)
        log.info("ArtifactStore initialized: %s", self.job_root)

    def cancel(self) -> None:
        """Signal cancellation. Long-running operations should check is_cancelled()."""
        self._cancelled = True
        log.warning("ArtifactStore: cancellation signalled for job %s", self.job_id)

    def is_cancelled(self) -> bool:
        return self._cancelled

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def has_stage(self, stage_key: str) -> bool:
        """
        Check if a stage was completed in a previous run.
        Uses a marker file: logs/<stage_key>.done
        """
        marker = self.job_root / "logs" / f"{stage_key}.done"
        return marker.exists()

    def mark_stage_done(self, stage_key: str) -> None:
        """Write a stage completion marker for resume support."""
        marker = self.job_root / "logs" / f"{stage_key}.done"
        marker.write_text("done", encoding="utf-8")

    def clear_stage(self, stage_key: str) -> None:
        """Remove a stage completion marker (for re-run)."""
        marker = self.job_root / "logs" / f"{stage_key}.done"
        marker.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Atomic file operations
    # ------------------------------------------------------------------

    def atomic_write(self, relative_path: str, content: bytes | str) -> Path:
        """
        Write content to a file atomically (write to temp, then rename).
        Ensures no partial writes are visible.
        """
        dest = self.job_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest.parent)
        try:
            if isinstance(content, str):
                content = content.encode("utf-8")
            os.write(tmp_fd, content)
            os.close(tmp_fd)
            shutil.move(tmp_path, dest)
        except Exception:
            os.close(tmp_fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        log.debug("ArtifactStore: written %s", relative_path)
        return dest

    def atomic_copy(self, src: str | Path, relative_path: str) -> Path:
        """Atomically copy a file into the artifact store."""
        src = Path(src)
        dest = self.job_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest.parent)
        os.close(tmp_fd)
        try:
            shutil.copy2(src, tmp_path)
            shutil.move(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        log.debug("ArtifactStore: copied %s -> %s", src.name, relative_path)
        return dest

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------

    def register(
        self,
        artifact: ArtifactMeta,
        *,
        compute_hash: bool = True,
    ) -> ArtifactMeta:
        """
        Register an artifact: compute its SHA-256 and return updated copy.
        The artifact's relative_path must already be set.
        """
        if compute_hash and artifact.relative_path:
            artifact = artifact.model_copy(update={
                "sha256": sha256_file(self.job_root / artifact.relative_path)
            })
        return artifact

    # ------------------------------------------------------------------
    # JSON manifest
    # ------------------------------------------------------------------

    def save_manifest(self, manifest: PipelineManifest) -> Path:
        """Atomically write the pipeline manifest."""
        content = manifest.model_dump_json(indent=2)
        return self.atomic_write("manifest.json", content)

    def load_manifest(self) -> Optional[PipelineManifest]:
        """Load manifest if it exists (for resume)."""
        path = self.job_root / "manifest.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PipelineManifest.model_validate(data)
        except Exception as e:
            log.warning("Failed to load manifest: %s", e)
            return None

    def save_json(self, relative_path: str, data: Any) -> Path:
        """Atomically write any JSON data."""
        content = json.dumps(data, indent=2, default=str)
        return self.atomic_write(relative_path, content)

    def load_json(self, relative_path: str) -> Optional[Any]:
        """Load JSON data if present."""
        path = self.job_root / relative_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to load JSON %s: %s", relative_path, e)
            return None

    # ------------------------------------------------------------------
    # Cache by hash
    # ------------------------------------------------------------------

    def cache_key_exists(self, cache_key: str) -> bool:
        """Check if a cache key exists (cache_key is SHA-256 of inputs)."""
        cache_marker = self.job_root / "logs" / f"cache_{cache_key[:16]}.hit"
        return cache_marker.exists()

    def record_cache_hit(self, cache_key: str, artifact_path: str) -> None:
        """Record a cache hit for a given input hash."""
        marker = self.job_root / "logs" / f"cache_{cache_key[:16]}.hit"
        marker.write_text(artifact_path, encoding="utf-8")

    def get_cached_path(self, cache_key: str) -> Optional[str]:
        """Retrieve the cached artifact path for a given input hash."""
        marker = self.job_root / "logs" / f"cache_{cache_key[:16]}.hit"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return None

    # ------------------------------------------------------------------
    # Failed candidate preservation
    # ------------------------------------------------------------------

    def preserve_failed_candidate(
        self, candidate_id: str, reason: str, mesh_path: Optional[str] = None
    ) -> None:
        """
        Mark a candidate as rejected but preserve all its files.
        Never delete failed candidates — they may be useful for debugging.
        """
        rejection_info = {
            "candidate_id": candidate_id,
            "reason": reason,
            "mesh_path": mesh_path,
        }
        self.save_json(f"logs/rejected_{candidate_id[:8]}.json", rejection_info)
        log.info("Preserved failed candidate %s: %s", candidate_id[:8], reason)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def path(self, relative: str) -> Path:
        """Resolve a relative path to an absolute path within the job root."""
        return self.job_root / relative

    def relative(self, absolute: str | Path) -> str:
        """Convert an absolute path to relative (from job root)."""
        return str(Path(absolute).relative_to(self.job_root))

    def exists(self, relative: str) -> bool:
        return (self.job_root / relative).exists()

    def __repr__(self) -> str:
        return f"ArtifactStore(job_id={self.job_id}, root={self.job_root})"
