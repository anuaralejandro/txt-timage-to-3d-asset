"""local_asset_factory · preflight"""
from .alpha_detector import detect_alpha, AlphaResult
from .checkerboard_detector import detect_checkerboard, CheckerboardResult
from .preflight_runner import PreflightRunner, PreflightConfig, DEFAULT_CONFIG

__all__ = [
    "detect_alpha", "AlphaResult",
    "detect_checkerboard", "CheckerboardResult",
    "PreflightRunner", "PreflightConfig", "DEFAULT_CONFIG",
]
