"""
Blender Headless Script: Canonical Multi-View Render
Usage:
    blender --background --python blender/scripts/canonical_render.py -- \
        --input_mesh path/to/model.glb \
        --output_dir path/to/renders/ \
        --resolution 1024
"""
import argparse
import math
import sys
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None


def setup_scene(resolution: int = 1024):
    """Clean default scene and set up camera & lights."""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, 'RenderEngineEEVEENext') else 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Add key light
    light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    # Add fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='SUN')
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(-30), math.radians(-45), math.radians(200))

    # Add camera
    cam_data = bpy.data.cameras.new(name="CanonicalCam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 2.2
    cam_obj = bpy.data.objects.new(name="CanonicalCam", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    return cam_obj


def load_mesh(mesh_path: str):
    """Import GLB/OBJ/FBX mesh into scene."""
    ext = Path(mesh_path).suffix.lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")


def position_camera(cam_obj, yaw_deg: float, pitch_deg: float = 0.0, distance: float = 5.0):
    """Position ortho camera looking at target center."""
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)

    x = distance * math.sin(yaw_rad) * math.cos(pitch_rad)
    y = -distance * math.cos(yaw_rad) * math.cos(pitch_rad)
    z = distance * math.sin(pitch_rad) + 1.0   # offset to character center

    cam_obj.location = (x, y, z)

    # Point at (0, 0, 1.0)
    target = mathutils.Vector((0.0, 0.0, 1.0))
    direction = target - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


def render_views(cam_obj, output_dir: Path):
    """Render canonical set of 7 views."""
    output_dir.mkdir(parents=True, exist_ok=True)

    views = {
        "front": (0.0, 0.0),
        "left": (90.0, 0.0),
        "back": (180.0, 0.0),
        "right": (270.0, 0.0),
        "front_3q": (45.0, 15.0),
        "back_3q": (225.0, 15.0),
        "top": (0.0, 89.0),
    }

    scene = bpy.context.scene
    for name, (yaw, pitch) in views.items():
        position_camera(cam_obj, yaw_deg=yaw, pitch_deg=pitch)
        out_file = output_dir / f"{name}.png"
        scene.render.filepath = str(out_file)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {out_file}")


def main():
    if bpy is None:
        print("This script must be run inside Blender: blender -b -P canonical_render.py")
        sys.exit(1)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render canonical multi-views of a 3D mesh")
    parser.add_argument("--input_mesh", required=True, help="Path to GLB/OBJ mesh")
    parser.add_argument("--output_dir", required=True, help="Directory to save renders")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution")
    args = parser.parse_args(argv)

    cam = setup_scene(resolution=args.resolution)
    load_mesh(args.input_mesh)
    render_views(cam, Path(args.output_dir))
    print("Canonical rendering finished successfully.")


if __name__ == "__main__":
    main()
