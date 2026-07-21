#!/usr/bin/env python3
"""
verify_comfy_models.py — Verify ComfyUI connectivity and workflow availability.

Run from any directory:
    python scripts/verify_comfy_models.py
"""

import json
import os
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not installed. Run: pip install requests")
    sys.exit(1)


def main():
    print("=" * 60)
    print("  LocalAssetFactory — ComfyUI Verification")
    print("=" * 60)
    print()

    base_url = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").strip().rstrip("/")
    print(f"  ComfyUI URL: {base_url}")
    print()

    # 1. Connectivity
    try:
        r = requests.get(f"{base_url}/system_stats", timeout=10)
        r.raise_for_status()
        stats = r.json()
        print("  ✓  ComfyUI server is reachable")

        # Show some system info
        devices = stats.get("devices", [])
        for dev in devices:
            name = dev.get("name", "unknown")
            vram_total = dev.get("vram_total", 0)
            vram_free = dev.get("vram_free", 0)
            vram_gb = vram_total / (1024**3) if vram_total else 0
            free_gb = vram_free / (1024**3) if vram_free else 0
            print(f"       GPU: {name} | VRAM: {vram_gb:.1f} GB total, {free_gb:.1f} GB free")

    except requests.ConnectionError:
        print(f"  ✗  Cannot connect to ComfyUI at {base_url}")
        print("     → Make sure ComfyUI is running")
        return 1
    except Exception as exc:
        print(f"  ✗  ComfyUI connection error: {exc}")
        return 1

    # 2. Check for object_info (node types)
    try:
        r = requests.get(f"{base_url}/object_info", timeout=15)
        r.raise_for_status()
        info = r.json()
        total_nodes = len(info)
        print(f"  ✓  Available node types: {total_nodes}")

        # Check for our custom nodes
        our_nodes = [k for k in info.keys() if k.startswith("AssetFactory_")]
        if our_nodes:
            print(f"  ✓  LocalAssetFactory nodes registered: {len(our_nodes)}")
            for n in our_nodes:
                print(f"       - {n}")
        else:
            print("  ⚠  LocalAssetFactory nodes NOT found in ComfyUI")
            print("     → Restart ComfyUI after installing the custom nodes")

        # Check for TRELLIS nodes
        trellis_nodes = [k for k in info.keys() if "trellis" in k.lower()]
        if trellis_nodes:
            print(f"  ✓  TRELLIS nodes found: {len(trellis_nodes)}")
        else:
            print("  ⚠  TRELLIS nodes not found (optional)")

    except Exception as exc:
        print(f"  ⚠  Could not check node types: {exc}")

    # 3. Check workflow files
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflows_dir = os.path.join(pkg_root, "workflows")

    expected_workflows = [
        "concept_image_v1.json",
        "texture_generation_v1.json",
        "trellis_local_v1.json",
        "local_asset_factory_v1.json",
        "local_asset_factory_v1_api.json",
    ]

    print(f"\n  Workflow files in: {workflows_dir}")
    for wf_name in expected_workflows:
        wf_path = os.path.join(workflows_dir, wf_name)
        if os.path.isfile(wf_path):
            try:
                with open(wf_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                node_count = len([k for k in data.keys() if not k.startswith("_")])
                print(f"  ✓  {wf_name} ({node_count} nodes)")
            except json.JSONDecodeError:
                print(f"  ✗  {wf_name} — invalid JSON!")
        else:
            print(f"  ✗  {wf_name} — not found")

    print("\n  ✅  ComfyUI verification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
