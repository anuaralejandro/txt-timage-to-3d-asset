"""
pytest conftest — local_asset_factory test suite.
Adds src/ to sys.path so tests can import the package directly.
"""
import sys
from pathlib import Path

# Ensure src/ is on the path (works without installing the package)
src = Path(__file__).parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
