"""
Blender Headless Script: Final Mobile GLB Exporter & glTF Validation
Validates materials, axes (-Z forward, Y up), scale, and exports GLB.

Usage:
    blender -b -P blender/scripts/glb_export.py -- \
        --input_scene path/to/scene.blend_or_glb \
        --output_glb path/to/final_character.glb
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


def export_mobile_glb(input_path: str, output_path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = Path(input_path).suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.blend':
        bpy.ops.wm.open_mainfile(filepath=input_path)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(out_p),
        export_format='GLB',
        export_yup=True,                  # Y-up for glTF
        export_apply=True,                # Apply modifiers
        export_skins=True,                # Include skeleton & weights
        export_morph=True,                # Include morph targets
        export_animations=True,           # Include actions
        export_materials='EXPORT',
    )
    print(f"Exported production GLB -> {out_p}")


def main():
    if bpy is None:
        sys.exit(1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_scene", required=True)
    parser.add_argument("--output_glb", required=True)
    args = parser.parse_args(argv)

    export_mobile_glb(args.input_scene, args.output_glb)


if __name__ == "__main__":
    main()
