"""
Blender Script — optimize_mesh.py
Standalone script to optimize a mesh: remove doubles, clean normals,
apply transforms, and optionally decimate.

Usage:
    blender --background model.glb --python optimize_mesh.py -- output.glb [max_triangles]
"""

import os
import sys

try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


def count_triangles() -> int:
    total = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        for poly in mesh.polygons:
            total += max(1, len(poly.vertices) - 2)
        eval_obj.to_mesh_clear()
    return total


def optimize():
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")

        obj.select_set(False)


def decimate_if_needed(max_triangles: int):
    current = count_triangles()
    if current <= max_triangles:
        print(f"Triangle count {current} is within budget {max_triangles}, skipping decimation.")
        return

    ratio = max_triangles / max(current, 1)
    print(f"Decimating: {current} → target {max_triangles} (ratio {ratio:.3f})")

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
        mod.ratio = max(0.05, ratio)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background model.glb --python optimize_mesh.py -- output.glb [max_triangles]")
        sys.exit(1)

    extra_args = argv[argv.index("--") + 1 :]
    output_path = extra_args[0] if extra_args else "optimized.glb"
    max_tris = int(extra_args[1]) if len(extra_args) > 1 else 0

    print(f"Triangles before optimization: {count_triangles()}")
    optimize()
    print(f"Triangles after optimization: {count_triangles()}")

    if max_tris > 0:
        decimate_if_needed(max_tris)
        print(f"Triangles after decimation: {count_triangles()}")

    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB", export_apply=True)
    elif ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=output_path)
    else:
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB", export_apply=True)

    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
