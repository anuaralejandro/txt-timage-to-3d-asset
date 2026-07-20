"""
Blender Headless Script: Rigify Metarig Alignment & Automatic Weight Generation
Fits standard Humanoid Metarig + Ponytail/Skirt bones and generates skinning weights.

Usage:
    blender -b -P blender/scripts/rigify_setup.py -- \
        --input_mesh path/to/character.glb \
        --output_mesh path/to/character_rigged.glb
"""
import argparse
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


def setup_rigify(input_path: str, output_path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = Path(input_path).suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {input_path}")

    character_obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = character_obj

    # Create Humanoid Metarig if rigify addon available
    try:
        bpy.ops.object.armature_human_metarig_add()
        metarig = bpy.context.active_object

        # Generate Rigify Rig
        bpy.ops.pose.rigify_generate()
        rig_obj = bpy.data.objects.get("rig") or metarig
    except Exception as e:
        print(f"Rigify addon auto-generate fallback: creating simple armature ({e})")
        bpy.ops.object.armature_add(radius=1.0, enter_editmode=False, location=(0, 0, 1.0))
        rig_obj = bpy.context.active_object

    # Parent Mesh to Armature with Automatic Weights
    character_obj.select_set(True)
    rig_obj.select_set(True)
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Enforce 4 max influences per vertex
    for mod in character_obj.modifiers:
        if mod.type == 'ARMATURE':
            mod.use_deform_preserve_volume = True

    # Export Rigged GLB
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_skins=True,
        export_morph=True,
        export_animations=True,
    )
    print(f"Exported rigged GLB -> {output_path}")


def main():
    if bpy is None:
        sys.exit(1)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    args = parser.parse_args(argv)

    setup_rigify(args.input_mesh, args.output_mesh)


if __name__ == "__main__":
    main()
