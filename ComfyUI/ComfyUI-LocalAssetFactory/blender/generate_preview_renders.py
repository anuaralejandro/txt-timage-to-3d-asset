"""
Blender Script — generate_preview_renders.py
Standalone script to generate preview renders of a 3D model.

Usage:
    blender --background model.glb --python generate_preview_renders.py -- output_dir asset_id
"""

import os
import sys

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background model.glb --python generate_preview_renders.py -- output_dir asset_id")
        sys.exit(1)

    extra_args = argv[argv.index("--") + 1 :]
    if len(extra_args) < 2:
        print("ERROR: Need output_dir and asset_id after '--'")
        sys.exit(1)

    output_dir = extra_args[0]
    asset_id = extra_args[1]
    os.makedirs(output_dir, exist_ok=True)

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"

    # Add light
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    angles = [
        ("front", (0, -8, 3), (1.2, 0, 0)),
        ("three_quarter", (5, -5, 4), (1.1, 0, 0.8)),
        ("side", (8, 0, 3), (1.2, 0, 1.57)),
        ("top", (0, 0, 10), (0, 0, 0)),
    ]

    renders = []
    for name, loc, rot in angles:
        bpy.ops.object.camera_add(location=loc, rotation=rot)
        cam = bpy.context.active_object
        scene.camera = cam

        path = os.path.join(output_dir, f"{asset_id}_preview_{name}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        renders.append(path)
        print(f"Rendered: {path}")

        bpy.data.objects.remove(cam)

    bpy.data.objects.remove(sun)
    print(f"Generated {len(renders)} preview renders.")


if __name__ == "__main__":
    main()
