"""
ComfyUI-LocalAssetFactory · Environment Utilities
Helpers for verifying runtime environment prerequisites.
"""

import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

try:
    from .logging_utils import get_logger
except ImportError:
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


def check_python_version(min_major: int = 3, min_minor: int = 10) -> Tuple[bool, str]:
    """Return (ok, message) indicating whether Python version is sufficient."""
    v = sys.version_info
    ok = v.major >= min_major and v.minor >= min_minor
    msg = f"Python {v.major}.{v.minor}.{v.micro}"
    if not ok:
        msg += f" — requires >= {min_major}.{min_minor}"
    return ok, msg


def check_env_var(name: str) -> Tuple[bool, str]:
    """Return (ok, message) for a given environment variable."""
    val = os.environ.get(name, "").strip()
    if val:
        # Never print the full value for safety
        return True, f"{name} is set"
    return False, f"{name} is NOT set"


def check_directory_writable(path: str) -> Tuple[bool, str]:
    """Check that *path* exists (or can be created) and is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True, f"Directory writable: {path}"
    except Exception as exc:
        return False, f"Directory NOT writable ({path}): {exc}"


def check_executable(name: str, path: str = "") -> Tuple[bool, str]:
    """Check that an executable exists at *path* or is on PATH."""
    target = path or name
    resolved = shutil.which(target)
    if resolved:
        return True, f"{name} found: {resolved}"
    return False, f"{name} NOT found (tried: {target})"


def check_blender(blender_path: str = "blender") -> Tuple[bool, str]:
    """Verify Blender exists and can run in background mode."""
    resolved = shutil.which(blender_path)
    if not resolved:
        return False, f"Blender NOT found at: {blender_path}"
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        version_line = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
        return True, f"Blender OK: {version_line}"
    except Exception as exc:
        return False, f"Blender test failed: {exc}"


def run_full_check() -> Dict[str, Tuple[bool, str]]:
    """Run all environment checks and return a summary dict."""
    from .. import config  # local import to avoid circular

    results: Dict[str, Tuple[bool, str]] = {}
    results["python"] = check_python_version()
    results["OLLAMA_BASE_URL"] = check_env_var("OLLAMA_BASE_URL")
    results["COMFYUI_BASE_URL"] = check_env_var("COMFYUI_BASE_URL")
    results["BLENDER_EXECUTABLE"] = check_env_var("BLENDER_EXECUTABLE")
    results["output_dir"] = check_directory_writable(config.ASSET_FACTORY_OUTPUT_DIR)
    results["blender_exec"] = check_blender(config.BLENDER_EXECUTABLE)
    return results
