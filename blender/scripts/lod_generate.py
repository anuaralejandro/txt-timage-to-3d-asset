"""
Blender Headless Script: LOD Pyramid Generator (LOD0 -> LOD1 -> LOD2 -> LOD3)
Preserves bone skinning weights while decimating triangle budgets for mobile profiles.

Usage:
    blender -b -P blender/scripts/lod_generate.py -- \
        --input_mesh path/to/character_rigged.glb \
        --output_dir path/to/lods/ \
        --profile android_mid
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None

PROFILES = {
    "android_low": [16000, 8000, 3500, 1200],
    "android_mid": [28000, 14000, 6500, 2200],
    "android_high": [45000, 22000, 10000, 4000],
}


def generate_lods(input_path: str, output_dir: str, profile_name: str = "android_mid"):
    budgets = PROFILES.get(profile_name, PROFILES["android_mid"])
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    for idx, target_tris in enumerate(budgets):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        ext = Path(input_path).suffix.lower()
        if ext in ('.glb', '.gltf'):
            bpy.ops.import_scene.gltf(filepath=input_path)

        mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        if mesh_objs:
            obj = mesh_objs[0]
            bpy.context.view_layer.objects.active = obj

            # Apply decimation to reach target budget while preserving skinning
            mod = obj.modifiers.new(name="LODDecimate", type='DECIMATE')
            curr_tris = len(obj.data.polygons)
            if curr_tris > target_tris:
                mod.ratio = target_tris / curr_tris
                bpy.ops.object.modifier_apply(modifier=mod.name)

        lod_file = out_d / f"LOD{idx}.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(lod_file),
            export_format='GLB',
            export_skins=True,
            export_morph=True,
            export_animations=True,
        )
        print(f"Generated LOD{idx} (budget={target_tris} tris) -> {lod_file}")


def main():
    if bpy is None:
        sys.exit(1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile", default="android_mid")
    args = parser.parse_args(argv)

    generate_lods(args.input_mesh, args.output_dir, args.profile)


if __name__ == "__main__":
    main()
