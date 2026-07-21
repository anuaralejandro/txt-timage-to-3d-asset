"""
Blender Script: render_glb_previews.py
Renders high-quality 2D preview thumbnails of a .glb 3D model from 4 camera angles.

Usage:
    blender --background --python render_glb_previews.py -- <glb_path> <output_dir> <asset_id>
"""

import json
import math
import os
import sys

try:
    import bpy
    import mathutils
except ImportError:
    print("ERROR: This script must be executed inside Blender.")
    sys.exit(1)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_glb(glb_path: str):
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(mesh_objs) > 1:
        for obj in list(mesh_objs):
            if obj.name.lower() in ("cube", "default cube", "cube.001"):
                print(f"[PREVIEW] Filtered out default cube object: {obj.name}")
                bpy.data.objects.remove(obj, do_unlink=True)
                mesh_objs.remove(obj)
    if not mesh_objs:
        raise ValueError(f"No mesh objects found in imported GLB: {glb_path}")
    return mesh_objs


def calculate_bounding_box(mesh_objs):
    min_coords = [float("inf")] * 3
    max_coords = [float("-inf")] * 3
    for obj in mesh_objs:
        matrix = obj.matrix_world
        for corner in obj.bound_box:
            world_corner = matrix @ mathutils.Vector(corner)
            for i in range(3):
                min_coords[i] = min(min_coords[i], world_corner[i])
                max_coords[i] = max(max_coords[i], world_corner[i])
    center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
    size = [max_coords[i] - min_coords[i] for i in range(3)]
    max_dim = max(size) if max(size) > 0 else 1.0
    return center, max_dim


def setup_lighting(center, max_dim):
    # Key light (Sun)
    bpy.ops.object.light_add(type="SUN", location=(center[0] + max_dim * 2, center[1] - max_dim * 2, center[2] + max_dim * 3))
    key = bpy.context.active_object
    key.data.energy = 4.0
    
    # Fill light
    bpy.ops.object.light_add(type="SUN", location=(center[0] - max_dim * 2, center[1] - max_dim * 2, center[2] + max_dim * 1.5))
    fill = bpy.context.active_object
    fill.data.energy = 2.0

    # Rim light
    bpy.ops.object.light_add(type="SUN", location=(center[0], center[1] + max_dim * 3, center[2] + max_dim * 2))
    rim = bpy.context.active_object
    rim.data.energy = 2.5


def frame_camera_and_render(center, max_dim, output_dir, asset_id):
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        scene.display.shading.color_type = "TEXTURE"
        scene.display.shading.light = "STUDIO"
    except Exception:
        pass

    # Camera distance based on bounding box
    dist = max_dim * 2.2 if max_dim > 0 else 5.0
    cx, cy, cz = center

    angles = [
        ("front", (cx, cy - dist, cz), (1.5708, 0, 0)),
        ("three_quarter", (cx + dist * 0.7, cy - dist * 0.7, cz + dist * 0.5), (1.1, 0, 0.785)),
        ("side", (cx + dist, cy, cz), (1.5708, 0, 1.5708)),
        ("top", (cx, cy, cz + dist), (0, 0, 0)),
    ]

    rendered_files = []
    for name, cam_pos, cam_rot in angles:
        bpy.ops.object.camera_add(location=cam_pos, rotation=cam_rot)
        cam = bpy.context.active_object
        scene.camera = cam

        output_path = os.path.join(output_dir, f"{asset_id}_preview_{name}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        rendered_files.append(output_path)
        print(f"[PREVIEW] Rendered {name}: {output_path}")

        bpy.data.objects.remove(cam)

    return rendered_files


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python render_glb_previews.py -- <glb_path> <output_dir> <asset_id>")
        sys.exit(1)

    args = argv[argv.index("--") + 1 :]
    if len(args) < 3:
        print("ERROR: Expected <glb_path> <output_dir> <asset_id>")
        sys.exit(1)

    glb_path, output_dir, asset_id = args[0], args[1], args[2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[PREVIEW] Loading GLB: {glb_path}")
    clear_scene()
    mesh_objs = import_glb(glb_path)
    center, max_dim = calculate_bounding_box(mesh_objs)
    print(f"[PREVIEW] Bounding box center: {center}, max_dim: {max_dim:.4f}")

    setup_lighting(center, max_dim)
    rendered = frame_camera_and_render(center, max_dim, output_dir, asset_id)

    # Save JSON manifest of preview renders
    manifest_path = os.path.join(output_dir, f"{asset_id}_previews.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(rendered, f, indent=2)
    print(f"[PREVIEW] Done. Saved {len(rendered)} renders.")


if __name__ == "__main__":
    main()
