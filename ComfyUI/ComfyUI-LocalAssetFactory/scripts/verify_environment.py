#!/usr/bin/env python3
"""
verify_environment.py — Check all prerequisites for LocalAssetFactory.

Run from any directory:
    python scripts/verify_environment.py
"""

import os
import shutil
import subprocess
import sys


def main():
    print("=" * 60)
    print("  LocalAssetFactory — Environment Verification")
    print("=" * 60)
    print()

    all_ok = True

    # 1. Python version
    v = sys.version_info
    py_ok = v.major >= 3 and v.minor >= 10
    status = "✓" if py_ok else "✗"
    print(f"  {status}  Python {v.major}.{v.minor}.{v.micro}  (need ≥ 3.10)")
    if not py_ok:
        all_ok = False

    # 2. Required env vars
    required_vars = [
        ("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ("COMFYUI_BASE_URL", "http://127.0.0.1:8188"),
        ("BLENDER_EXECUTABLE", "blender"),
    ]
    for var, default in required_vars:
        val = os.environ.get(var, "").strip()
        if val:
            print(f"  ✓  {var} is set")
        else:
            print(f"  ⚠  {var} not set (will use default: {default})")

    # 3. Output directory
    out_dir = os.environ.get("ASSET_FACTORY_OUTPUT_DIR", "").strip()
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            test_file = os.path.join(out_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            print(f"  ✓  Output directory writable: {out_dir}")
        except Exception as exc:
            print(f"  ✗  Output directory NOT writable: {exc}")
            all_ok = False
    else:
        print("  ⚠  ASSET_FACTORY_OUTPUT_DIR not set (will use default)")

    # 4. Blender
    blender_path = os.environ.get("BLENDER_EXECUTABLE", "blender").strip()
    resolved = shutil.which(blender_path)
    if resolved:
        try:
            result = subprocess.run(
                [resolved, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
            print(f"  ✓  Blender found: {version}")
        except Exception as exc:
            print(f"  ✗  Blender found but failed to run: {exc}")
            all_ok = False
    else:
        print(f"  ✗  Blender not found at: {blender_path}")
        all_ok = False

    # 5. Required Python packages
    packages = ["requests", "httpx", "pydantic", "dotenv", "PIL", "numpy"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓  Python package: {pkg}")
        except ImportError:
            print(f"  ✗  Python package missing: {pkg}")
            all_ok = False

    print()
    if all_ok:
        print("  ✅  All checks passed!")
    else:
        print("  ❌  Some checks failed. See above for details.")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
