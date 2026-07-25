"""
pytest conftest — local_asset_factory test suite.
Adds root, src/, and custom_nodes/ to sys.path so tests can import all modules directly.
"""
import sys
from pathlib import Path

root = Path(__file__).parent.parent
src = root / "src"
custom_nodes = root / "ComfyUI_windows_portable" / "ComfyUI" / "custom_nodes" / "ComfyUI-LocalAssetFactory"

for path_dir in [root, src, custom_nodes]:
    if str(path_dir) not in sys.path:
        sys.path.insert(0, str(path_dir))
