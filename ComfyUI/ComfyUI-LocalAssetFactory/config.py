"""
ComfyUI-LocalAssetFactory · Configuration Module
Loads settings from environment variables first, then .env file as fallback.
"""

import os
import pathlib
from typing import Optional

# Try loading .env but do not override existing system env vars
try:
    from dotenv import load_dotenv

    _env_path = pathlib.Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
    else:
        # Also check parent directories in case of alternative layouts
        load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv is optional at import time


def _bool_env(key: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _int_env(key: str, default: int) -> int:
    """Read an integer environment variable."""
    val = os.environ.get(key, "").strip()
    if val.isdigit():
        return int(val)
    return default


def _str_env(key: str, default: str = "") -> str:
    """Read a string environment variable."""
    return os.environ.get(key, default).strip()


# ---------------------------------------------------------------------------
# Core backend URLs
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = _str_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = _str_env("OLLAMA_MODEL", "mistral")

COMFYUI_BASE_URL: str = _str_env("COMFYUI_BASE_URL", "http://127.0.0.1:8188")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_default_output = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.parent.resolve(),  # ComfyUI root
    "output",
    "local_asset_factory",
)
ASSET_FACTORY_OUTPUT_DIR: str = _str_env("ASSET_FACTORY_OUTPUT_DIR", _default_output)

# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------
def _detect_blender() -> str:
    env_val = _str_env("BLENDER_EXECUTABLE", "")
    if env_val:
        return env_val
    paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
    ]
    for p in paths:
        if os.path.isfile(p):
            return p
    return "blender"

BLENDER_EXECUTABLE: str = _detect_blender()

# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------
ASSET_FACTORY_MODE: str = _str_env("ASSET_FACTORY_MODE", "full")
# Modes: spec_only | concept | concept_textures | full | full_blender

# ---------------------------------------------------------------------------
# Workflow paths (relative to package root or absolute)
# ---------------------------------------------------------------------------
_pkg_root = str(pathlib.Path(__file__).parent.resolve())

CONCEPT_WORKFLOW_PATH: str = _str_env(
    "CONCEPT_WORKFLOW_PATH",
    os.path.join(_pkg_root, "workflows", "concept_image_v1.json"),
)
TEXTURE_WORKFLOW_PATH: str = _str_env(
    "TEXTURE_WORKFLOW_PATH",
    os.path.join(_pkg_root, "workflows", "texture_generation_v1.json"),
)
HUNYUAN_WORKFLOW_PATH: str = _str_env(
    "HUNYUAN_WORKFLOW_PATH",
    os.path.join(_pkg_root, "workflows", "hunyuan_multiview_v1.json"),
)
HUNYUAN_2MV_WORKFLOW_PATH: str = _str_env(
    "HUNYUAN_2MV_WORKFLOW_PATH",
    os.path.join(_pkg_root, "workflows", "01_multiview_to_mesh.json"),
)

# ---------------------------------------------------------------------------
# Image generation defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_MODEL_NAME: str = _str_env("DEFAULT_IMAGE_MODEL_NAME", "")
DEFAULT_TEXTURE_MODEL_NAME: str = _str_env("DEFAULT_TEXTURE_MODEL_NAME", "")

DEFAULT_CONCEPT_WIDTH: int = _int_env("DEFAULT_CONCEPT_WIDTH", 768)
DEFAULT_CONCEPT_HEIGHT: int = _int_env("DEFAULT_CONCEPT_HEIGHT", 768)
DEFAULT_TEXTURE_WIDTH: int = _int_env("DEFAULT_TEXTURE_WIDTH", 512)
DEFAULT_TEXTURE_HEIGHT: int = _int_env("DEFAULT_TEXTURE_HEIGHT", 512)

# ---------------------------------------------------------------------------
# Timeouts & retries
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_SECONDS: int = _int_env("DEFAULT_TIMEOUT_SECONDS", 900)
DEFAULT_MAX_RETRIES: int = _int_env("DEFAULT_MAX_RETRIES", 2)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
# NOTE: TRELLIS has been removed. Hunyuan3D-2mv is the sole geometry backend.
ENABLE_HUNYUAN: bool = _bool_env("ENABLE_HUNYUAN", True)
ENABLE_HUNYUAN_OMNI: bool = _bool_env("ENABLE_HUNYUAN_OMNI", True)
ENABLE_SAM3: bool = _bool_env("ENABLE_SAM3", True)
ENABLE_HUNYUAN_PART: bool = _bool_env("ENABLE_HUNYUAN_PART", True)
ENABLE_HUNYUAN_PAINT: bool = _bool_env("ENABLE_HUNYUAN_PAINT", True)
ENABLE_RIGANYTHING: bool = _bool_env("ENABLE_RIGANYTHING", False)  # non-commercial
ENABLE_TEXTURE_GENERATION: bool = _bool_env("ENABLE_TEXTURE_GENERATION", True)
ENABLE_BLENDER_PROCESSING: bool = _bool_env("ENABLE_BLENDER_PROCESSING", True)
ALLOW_PARTIAL_SUCCESS: bool = _bool_env("ALLOW_PARTIAL_SUCCESS", True)

# ---------------------------------------------------------------------------
# Allowed texture resolutions
# ---------------------------------------------------------------------------
ALLOWED_TEXTURE_RESOLUTIONS = (256, 512, 1024, 2048)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _str_env("ASSET_FACTORY_LOG_LEVEL", "INFO")
