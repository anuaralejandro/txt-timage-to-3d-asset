"""
ComfyUI-LocalAssetFactory · File Safety Utilities
Path validation, sanitisation, and traversal prevention.
"""

import os
import re
from pathlib import Path
from typing import Optional

try:
    from .logging_utils import get_logger
except ImportError:
    # Fallback for standalone execution or test contexts
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

log = get_logger(__name__)


def sanitize_filename(raw: str, max_length: int = 200) -> str:
    """Produce a safe filename from *raw*.

    - Removes or replaces unsafe characters.
    - Collapses multiple underscores.
    - Truncates to *max_length*.
    """
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", raw.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "unnamed"
    return safe[:max_length]


def ensure_within(target: str, root: str) -> str:
    """Resolve *target* and verify it is inside *root*.

    Raises ``ValueError`` on path traversal attempt.
    Returns the resolved absolute path as a string.
    """
    root_resolved = Path(root).resolve()
    target_resolved = Path(target).resolve()
    # On Windows, Path.is_relative_to was added in 3.9
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal blocked: {target_resolved} is outside {root_resolved}"
        )
    return str(target_resolved)


def safe_join(root: str, *parts: str) -> str:
    """Join *parts* onto *root* and validate the result stays inside *root*.

    Each part is sanitised individually before joining.
    """
    sanitised_parts = [sanitize_filename(p) for p in parts]
    candidate = os.path.join(root, *sanitised_parts)
    return ensure_within(candidate, root)


def ensure_directory(path: str) -> str:
    """Create directory if it doesn't exist.  Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_asset_dir(output_root: str, asset_id: str, *subdirs: str) -> str:
    """Return a validated and created directory for a given asset.

    Example::

        safe_asset_dir("/output", "sword_001", "textures", "source")
        # → "/output/sword_001/textures/source"  (created on disk)
    """
    parts = [asset_id] + list(subdirs)
    result = safe_join(output_root, *parts)
    ensure_directory(result)
    return result
