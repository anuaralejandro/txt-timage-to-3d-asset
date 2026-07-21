"""
ComfyUI-LocalAssetFactory · Logging Utilities
Provides a centralized logger for the entire package.
"""

import logging
import os
import sys

_LOG_FORMAT = "[LocalAssetFactory] %(levelname)s  %(name)s  %(message)s"

_level_str = os.environ.get("ASSET_FACTORY_LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_str, logging.INFO)


def get_logger(name: str = "LocalAssetFactory") -> logging.Logger:
    """Return a configured logger.  Safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(_level)
    return logger
