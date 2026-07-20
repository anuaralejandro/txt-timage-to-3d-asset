"""
Blender Headless Script: Organic Part Retopology
Uses QuadriFlow or Voxel Remesh + Quadrangulate for organic/cloth deforming parts (body, top, shorts).

Usage:
    blender -b -P blender/scripts/retopo_organic.py -- \
        --input_mesh path/to/part.glb \
        --output_mesh path/to/part_low.glb \
        --target_faces 3000
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


def retopo_organic(input_path: str, output_path: str, target_faces: int = 3000):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = Path(input_path).suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)

    # Select imported mesh
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects found in {input_path}")

    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Clean geometry
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.fill_holes()
    bpy.ops.object.mode_set(mode='OBJECT')

    # QuadriFlow remesh if available, else Decimate
    try:
        bpy.ops.object.quadriflow_remesh(target_faces=target_faces)
        print(f"QuadriFlow remesh applied (target={target_faces} faces).")
    except Exception as e:
        print(f"QuadriFlow unavailable ({e}), using Decimate modifier fallback...")
        mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
        curr_faces = len(obj.data.polygons)
        if curr_faces > 0:
            mod.ratio = max(0.05, min(1.0, target_faces / curr_faces))
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Export GLB
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    print(f"Exported low-poly organic mesh -> {output_path}")


def main():
    if bpy is None:
        print("Must be run inside Blender")
        sys.exit(1)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--target_faces", type=int, default=3000)
    args = parser.parse_args(argv)

    retopo_organic(args.input_mesh, args.output_mesh, args.target_faces)


if __name__ == "__main__":
    main()
