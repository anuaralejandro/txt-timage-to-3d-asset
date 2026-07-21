"""
ComfyUI-LocalAssetFactory · Retry Utilities
Simple retry decorator with exponential back-off.
"""

import time
import functools
from typing import Callable, Tuple, Type

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


def retry(
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator that retries a function on specified exceptions.

    Parameters
    ----------
    max_retries : int
        Maximum number of retries (0 = no retry, 1 = one retry, etc.).
    delay : float
        Initial delay in seconds between retries.
    backoff : float
        Multiplier applied to delay after each retry.
    exceptions : tuple
        Exception types that should trigger a retry.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        log.warning(
                            "Attempt %d/%d for %s failed: %s — retrying in %.1fs",
                            attempt + 1,
                            1 + max_retries,
                            func.__name__,
                            exc,
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        log.error(
                            "All %d attempts for %s failed: %s",
                            1 + max_retries,
                            func.__name__,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
