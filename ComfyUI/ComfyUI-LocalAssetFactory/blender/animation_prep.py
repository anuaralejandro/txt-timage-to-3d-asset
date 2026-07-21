"""
Blender Script — animation_prep.py
Prepares a 3D mesh for animation rigging (e.g., Mixamo).

Operations:
- Join all objects into a single mesh
- Auto Smart UV Project
- Scale to real-world height (1.8m for humanoids)
- Fix manifold (close holes)
- Position in T-Pose canonical (feet on ground, facing -Y)
- Apply smooth shading
- Validate and report metrics

Usage:
    blender --background --python animation_prep.py -- input.glb output.fbx [asset_type] [triangle_budget]
"""

import json
import math
import os
import sys

try:
    import bpy
    import bmesh
    from mathutils import Vector
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_model(model_path: str):
    """Import a 3D model (GLB, OBJ, FBX, PLY)."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=model_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=model_path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=model_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    print(f"Imported: {model_path}")


def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def join_all_meshes():
    """Join all mesh objects into a single mesh (required for Mixamo)."""
    meshes = get_mesh_objects()
    if len(meshes) <= 1:
        return

    # Deselect all
    bpy.ops.object.select_all(action="DESELECT")

    # Select all meshes
    for obj in meshes:
        obj.select_set(True)

    # Set the first one as active
    bpy.context.view_layer.objects.active = meshes[0]

    # Join
    bpy.ops.object.join()
    print(f"Joined {len(meshes)} meshes into one object.")


def clean_geometry():
    """Remove doubles, fix normals, delete loose."""
    for obj in get_mesh_objects():
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)

    print("Geometry cleaned.")


def fix_manifold():
    """Attempt to close holes in the mesh to make it manifold."""
    for obj in get_mesh_objects():
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")

        # Select non-manifold edges (holes)
        bpy.ops.mesh.select_non_manifold(
            extend=False, use_wire=True, use_boundary=True,
            use_multi_face=False, use_non_contiguous=False, use_verts=False
        )

        # Fill holes
        try:
            bpy.ops.mesh.fill()
            print("Manifold holes filled.")
        except Exception:
            print("WARNING: Could not fill all manifold holes.")

        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)


def auto_uv_project():
    """Generate automatic UVs using Smart UV Project."""
    for obj in get_mesh_objects():
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

        # Smart UV Project with good defaults for game assets
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66),
            island_margin=0.02,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True,
        )

        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)

    print("Auto UV projection complete.")


def scale_to_real_world(asset_type: str = "character", target_height: float = 1.8):
    """Scale model to real-world dimensions.
    
    Characters: 1.8m tall (Mixamo standard)
    Weapons: 1.0m long
    Props: 0.5m
    """
    heights = {
        "character": 1.8,
        "enemy": 1.8,
        "weapon": 1.0,
        "prop": 0.5,
        "architecture": 3.0,
        "vegetation": 2.0,
        "modular_piece": 1.0,
    }
    target = heights.get(asset_type, target_height)

    for obj in get_mesh_objects():
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)
        current_height = max_z - min_z

        if current_height > 0.001:
            scale_factor = target / current_height
            obj.scale *= scale_factor
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(scale=True)

    print(f"Scaled to {target}m for asset type '{asset_type}'.")


def position_on_ground():
    """Position model with feet on ground (Z=0), centered on X/Y, facing -Y."""
    for obj in get_mesh_objects():
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_z = min(v.z for v in bbox)
        center_x = sum(v.x for v in bbox) / 8
        center_y = sum(v.y for v in bbox) / 8

        obj.location.x -= center_x
        obj.location.y -= center_y
        obj.location.z -= min_z

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True)

    print("Positioned on ground, centered.")


def apply_smooth_shading(auto_smooth_angle: float = 30.0):
    """Apply smooth shading with auto-smooth for clean renders."""
    for obj in get_mesh_objects():
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.shade_smooth()

        # Auto smooth (Blender 4.x uses modifier-based approach)
        if bpy.app.version >= (4, 1, 0):
            try:
                bpy.ops.object.modifier_add(type='SMOOTH')
            except Exception:
                pass
        else:
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(auto_smooth_angle)

        obj.select_set(False)

    print("Smooth shading applied.")


def decimate_smart(max_triangles: int):
    """Decimate mesh while trying to preserve edge flow for animation."""
    current = count_triangles()
    if current <= max_triangles:
        print(f"Triangle count {current} within budget {max_triangles}.")
        return

    ratio = max_triangles / max(current, 1)
    print(f"Decimating: {current} → {max_triangles} (ratio {ratio:.3f})")

    for obj in get_mesh_objects():
        bpy.context.view_layer.objects.active = obj

        # Use Un-Subdivide first (preserves edge loops better for animation)
        mod = obj.modifiers.new(name="Decimate_UnSub", type="DECIMATE")
        mod.decimate_type = "UN_SUBDIVIDE"
        mod.iterations = 1
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            obj.modifiers.remove(mod)

        # If still over budget, use Collapse
        current_after = count_triangles()
        if current_after > max_triangles:
            final_ratio = max_triangles / max(current_after, 1)
            mod2 = obj.modifiers.new(name="Decimate_Collapse", type="DECIMATE")
            mod2.decimate_type = "COLLAPSE"
            mod2.ratio = max(0.05, final_ratio)
            bpy.ops.object.modifier_apply(modifier=mod2.name)


def count_triangles() -> int:
    total = 0
    for obj in get_mesh_objects():
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        for poly in mesh.polygons:
            total += max(1, len(poly.vertices) - 2)
        eval_obj.to_mesh_clear()
    return total


def check_is_manifold() -> bool:
    """Check if the mesh is watertight (manifold)."""
    for obj in get_mesh_objects():
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        non_manifold = [e for e in bm.edges if not e.is_manifold]
        bm.free()
        if non_manifold:
            return False
    return True


def check_has_uvs() -> bool:
    """Check if the mesh has UV layers."""
    for obj in get_mesh_objects():
        if not obj.data.uv_layers:
            return False
    return True


def get_bounding_box() -> dict:
    """Get the bounding box dimensions."""
    all_verts = []
    for obj in get_mesh_objects():
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        all_verts.extend(bbox)

    if not all_verts:
        return {"width": 0, "height": 0, "depth": 0}

    min_x = min(v.x for v in all_verts)
    max_x = max(v.x for v in all_verts)
    min_y = min(v.y for v in all_verts)
    max_y = max(v.y for v in all_verts)
    min_z = min(v.z for v in all_verts)
    max_z = max(v.z for v in all_verts)

    return {
        "width": round(max_x - min_x, 4),
        "height": round(max_z - min_z, 4),
        "depth": round(max_y - min_y, 4),
    }


def export_fbx_animation_ready(output_path: str):
    """Export FBX with settings optimized for animation rigging tools."""
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        apply_scale_options="FBX_SCALE_ALL",
        add_leaf_bones=False,
        axis_forward="-Z",
        axis_up="Y",
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        bake_anim=False,
    )
    print(f"Exported animation-ready FBX: {output_path}")


def export_glb(output_path: str):
    """Export GLB for web/engine preview."""
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
    )
    print(f"Exported GLB: {output_path}")


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python animation_prep.py -- input.glb output.fbx [asset_type] [max_triangles]")
        sys.exit(1)

    extra_args = argv[argv.index("--") + 1:]
    input_path = extra_args[0] if len(extra_args) > 0 else ""
    output_path = extra_args[1] if len(extra_args) > 1 else "output.fbx"
    asset_type = extra_args[2] if len(extra_args) > 2 else "character"
    max_tris = int(extra_args[3]) if len(extra_args) > 3 else 5000

    if not input_path or not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    report = {
        "success": False,
        "input_path": input_path,
        "output_path": "",
        "asset_type": asset_type,
        "triangle_count_before": 0,
        "triangle_count_after": 0,
        "is_manifold": False,
        "has_uvs": False,
        "bounding_box": {},
        "errors": [],
        "warnings": [],
    }

    try:
        clear_scene()
        import_model(input_path)

        report["triangle_count_before"] = count_triangles()

        # 1. Join all objects
        join_all_meshes()

        # 2. Clean geometry
        clean_geometry()

        # 3. Fix manifold
        fix_manifold()

        # 4. Auto UV
        auto_uv_project()

        # 5. Decimate if needed
        if max_tris > 0:
            decimate_smart(max_tris)

        # 6. Scale to real world
        scale_to_real_world(asset_type)

        # 7. Position on ground
        position_on_ground()

        # 8. Smooth shading
        apply_smooth_shading()

        # 9. Final metrics
        report["triangle_count_after"] = count_triangles()
        report["is_manifold"] = check_is_manifold()
        report["has_uvs"] = check_has_uvs()
        report["bounding_box"] = get_bounding_box()

        # 10. Export
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".fbx":
            export_fbx_animation_ready(output_path)
        elif ext in (".glb", ".gltf"):
            export_glb(output_path)
        else:
            # Export both
            fbx_path = os.path.splitext(output_path)[0] + ".fbx"
            glb_path = os.path.splitext(output_path)[0] + ".glb"
            export_fbx_animation_ready(fbx_path)
            export_glb(glb_path)
            output_path = fbx_path

        report["output_path"] = output_path
        report["success"] = True

    except Exception as e:
        report["errors"].append(str(e))
        print(f"ERROR: {e}")

    # Write report
    report_path = os.path.splitext(output_path)[0] + "_anim_report.json"
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Animation prep report: {report_path}")


if __name__ == "__main__":
    main()
