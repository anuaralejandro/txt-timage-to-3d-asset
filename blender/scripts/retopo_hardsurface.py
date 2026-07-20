"""
Blender Headless Script: Hard Surface Retopology
Uses Planar Decimation / Dissolve Degenerates for hard-surface parts (boots, belt, armor, accessories).

Usage:
    blender -b -P blender/scripts/retopo_hardsurface.py -- \
        --input_mesh path/to/part.glb \
        --output_mesh path/to/part_low.glb \
        --angle_limit 5.0
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


def retopo_hardsurface(input_path: str, output_path: str, angle_limit: float = 5.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = Path(input_path).suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {input_path}")

    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Planar decimation preserves sharp edges
    mod = obj.modifiers.new(name="PlanarDecimate", type='DECIMATE')
    mod.decimate_type = 'DISSOLVE'
    mod.angle_limit = angle_limit
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    print(f"Exported low-poly hard-surface mesh -> {output_path}")


def main():
    if bpy is None:
        sys.exit(1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--angle_limit", type=float, default=5.0)
    args = parser.parse_args(argv)

    retopo_hardsurface(args.input_mesh, args.output_mesh, args.angle_limit)


if __name__ == "__main__":
    main()
