"""
Blender Headless Script: Hair / Secondary Motion Retopology & Ponytail Separation
Usage:
    blender -b -P blender/scripts/retopo_hair.py -- \
        --input_mesh path/to/hair.glb \
        --output_mesh path/to/hair_low.glb
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


def retopo_hair(input_path: str, output_path: str):
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

    # Remesh hair curves / locks
    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.ratio = 0.4
    bpy.ops.object.modifier_apply(modifier=mod.name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    print(f"Exported low-poly hair mesh -> {output_path}")


def main():
    if bpy is None:
        sys.exit(1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    args = parser.parse_args(argv)

    retopo_hair(args.input_mesh, args.output_mesh)


if __name__ == "__main__":
    main()
