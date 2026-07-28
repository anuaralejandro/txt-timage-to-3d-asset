"""
Blender 4.4 High-Fidelity 30k Retopology, Smooth Shading & UV Unwrapping
Usage:
    blender --background --python scripts/blender_process_30k.py -- input.glb output_30k.glb 30000
"""

import os
import sys

try:
    import bpy
except ImportError:
    print("ERROR: Must run inside Blender.")
    sys.exit(1)


def log(msg):
    print(msg, flush=True)


def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


def count_triangles(objs):
    total = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objs:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        for poly in mesh.polygons:
            total += max(1, len(poly.vertices) - 2)
        eval_obj.to_mesh_clear()
    return total


def process_mesh_30k(input_path, output_path, target_tris=30000):
    log(f"=== Starting Blender 30k Optimization & Retopology ===")
    log(f"Input File: {input_path}")
    log(f"Target Triangle Count: {target_tris}")

    # 1. Clear Scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 2. Import GLB
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=input_path)
    else:
        log(f"Unsupported format: {ext}")
        sys.exit(1)

    mesh_objs = get_mesh_objects()
    if not mesh_objs:
        log("ERROR: No mesh objects imported.")
        sys.exit(1)

    # Join mesh objects if multiple
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()

    main_obj = get_mesh_objects()[0]
    bpy.context.view_layer.objects.active = main_obj
    bpy.ops.object.select_all(action='DESELECT')
    main_obj.select_set(True)

    initial_tris = count_triangles([main_obj])
    log(f"Initial Triangles: {initial_tris}")

    # 3. High-Fidelity Manifold Voxel Remesh to ~60k-80k tris
    max_dim = max(main_obj.dimensions) if max(main_obj.dimensions) > 0 else 1.0
    voxel_sz = max(0.005, max_dim / 180.0)
    log(f"Step 1/5: Manifold Voxel Remesh (size {voxel_sz:.5f})...")
    main_obj.data.remesh_voxel_size = voxel_sz
    bpy.ops.object.voxel_remesh()

    manifold_tris = count_triangles([main_obj])
    log(f"Triangles after Voxel Remesh: {manifold_tris}")

    # 4. Decimate Manifold Mesh to exactly target_tris (30,000)
    if manifold_tris > target_tris:
        ratio = target_tris / manifold_tris
        log(f"Step 2/5: Decimating {manifold_tris} -> {target_tris} triangles (ratio: {ratio:.5f})...")
        dec_mod = main_obj.modifiers.new(name="TargetDecimate", type='DECIMATE')
        dec_mod.decimate_type = 'COLLAPSE'
        dec_mod.ratio = max(0.01, ratio)
        bpy.ops.object.modifier_apply(modifier=dec_mod.name)

    dec_tris = count_triangles([main_obj])
    log(f"Triangles after Decimation: {dec_tris}")

    # 5. Clean Loose Geometry
    log("Step 3/5: Cleaning loose verts & recalculating normals...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')

    # 6. Surface Smoothing (Laplacian + Smooth Filters for Organic Smoothness)
    log("Step 4/5: Applying Surface Smoothness (Laplacian + Smooth Filters)...")
    lap_mod = main_obj.modifiers.new(name="LaplacianSmooth", type='LAPLACIANSMOOTH')
    lap_mod.lambda_factor = 0.5
    lap_mod.iterations = 3
    bpy.ops.object.modifier_apply(modifier=lap_mod.name)

    smooth_mod = main_obj.modifiers.new(name="Smooth", type='SMOOTH')
    smooth_mod.factor = 0.5
    smooth_mod.iterations = 3
    bpy.ops.object.modifier_apply(modifier=smooth_mod.name)

    log("Step 5/5: Optimizing Shading, Weighted Normals & UV Unwrapping...")
    bpy.ops.object.shade_smooth()

    wn_mod = main_obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
    wn_mod.weight = 50
    wn_mod.keep_sharp = True
    bpy.ops.object.modifier_apply(modifier=wn_mod.name)

    # Clean UV Unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=66.0,
        island_margin=0.004,
        correct_aspect=True,
        scale_to_bounds=False
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    final_tris = count_triangles([main_obj])
    log(f"Final Triangles: {final_tris}")

    log(f"Exporting optimized GLB asset to: {output_path}")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_apply=True,
        export_yup=True,
        export_materials='EXPORT',
        export_normals=True,
        export_texcoords=True
    )
    log(f"=== SUCCESS! Clean 30k Asset Saved: {output_path} ===")


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        inp = args[0] if len(args) > 0 else "input.glb"
        outp = args[1] if len(args) > 1 else "output_30k.glb"
        target = int(args[2]) if len(args) > 2 else 30000
    else:
        inp = "input.glb"
        outp = "output_30k.glb"
        target = 30000

    process_mesh_30k(inp, outp, target)
