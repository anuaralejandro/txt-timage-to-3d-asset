"""Batch syntax check for all Python files in the project."""
import ast
import os
import sys

root = r"c:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI\custom_nodes\ComfyUI-LocalAssetFactory"
errors = []
checked = 0

for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith(".py"):
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root)
            try:
                with open(fpath, encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)
                checked += 1
            except SyntaxError as e:
                errors.append(f"  FAIL: {rel} line {e.lineno}: {e.msg}")

print(f"Checked {checked} files.")
if errors:
    print("ERRORS:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("All files have valid syntax.")
