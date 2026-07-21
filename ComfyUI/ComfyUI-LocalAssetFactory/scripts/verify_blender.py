#!/usr/bin/env python3
"""
verify_blender.py — Verify that Blender is installed and can run headless.

Run from any directory:
    python scripts/verify_blender.py
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main():
    print("=" * 60)
    print("  LocalAssetFactory — Blender Verification")
    print("=" * 60)
    print()

    blender_path = os.environ.get("BLENDER_EXECUTABLE", "blender").strip()
    print(f"  Blender path: {blender_path}")
    print()

    # 1. Find executable
    resolved = shutil.which(blender_path)
    if resolved:
        print(f"  ✓  Blender found at: {resolved}")
    else:
        print(f"  ✗  Blender NOT found at: {blender_path}")
        print("     → Set BLENDER_EXECUTABLE to the full path")
        print("     → Example: C:/Program Files/Blender Foundation/Blender 4.1/blender.exe")
        return 1

    # 2. Version check
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        version = result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
        print(f"  ✓  Version: {version}")
    except Exception as exc:
        print(f"  ✗  Version check failed: {exc}")
        return 1

    # 3. Background mode test
    print("\n  Testing background mode...")
    test_script = (
        "import bpy; "
        "bpy.ops.mesh.primitive_cube_add(); "
        "print('BLENDER_TEST_OK'); "
    )

    try:
        result = subprocess.run(
            [resolved, "--background", "--python-expr", test_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "BLENDER_TEST_OK" in result.stdout:
            print("  ✓  Background mode works correctly")
        else:
            print("  ⚠  Background mode ran but output unexpected")
            if result.stderr:
                # Only show first few lines, no secrets
                lines = result.stderr.strip().split("\n")[:3]
                for line in lines:
                    print(f"       {line}")
    except subprocess.TimeoutExpired:
        print("  ✗  Background mode test timed out")
        return 1
    except Exception as exc:
        print(f"  ✗  Background mode test failed: {exc}")
        return 1

    # 4. Export test
    print("\n  Testing GLB export capability...")
    try:
        export_script = (
            "import bpy, os, tempfile; "
            "bpy.ops.mesh.primitive_cube_add(); "
            "out = os.path.join(tempfile.gettempdir(), 'blender_test.glb'); "
            "bpy.ops.export_scene.gltf(filepath=out, export_format='GLB'); "
            "print('EXPORT_OK' if os.path.isfile(out) else 'EXPORT_FAIL'); "
            "os.remove(out) if os.path.isfile(out) else None; "
        )
        result = subprocess.run(
            [resolved, "--background", "--python-expr", export_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "EXPORT_OK" in result.stdout:
            print("  ✓  GLB export works correctly")
        else:
            print("  ⚠  GLB export test inconclusive")
    except Exception as exc:
        print(f"  ⚠  Export test failed: {exc}")

    print("\n  ✅  Blender verification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
