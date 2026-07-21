"""
Blender Script — process_asset.py
Main entry point for headless Blender processing of 3D assets.

Called via:
    blender --background --python process_asset.py -- args_file.json

The args JSON must contain:
    model_path, texture_paths, asset_spec, output_dir,
    optimize_mesh, assign_texture, generate_previews, create_glb, create_fbx
"""

import json
import os
import sys

# Blender's bpy is only available inside Blender
try:
    import bpy
    import mathutils
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    sys.exit(1)


def load_args() -> dict:
    """Load arguments from the JSON file passed via '--' on the command line."""
    argv = sys.argv
    if "--" in argv:
        args_file = argv[argv.index("--") + 1]
    else:
        print("ERROR: No args file provided after '--'")
        sys.exit(1)

    with open(args_file, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Remove orphan data
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
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")
    mesh_objs = get_mesh_objects()
    if len(mesh_objs) > 1:
        for obj in list(mesh_objs):
            if obj.name.lower() in ("cube", "default cube", "cube.001"):
                print(f"Filtered out default cube object: {obj.name}")
                bpy.data.objects.remove(obj, do_unlink=True)
    print(f"Imported: {model_path}")


def get_mesh_objects():
    """Return all mesh objects in the scene."""
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def optimize_geometry():
    """Clean up and optimize mesh geometry safely."""
    for obj in get_mesh_objects():
        if len(obj.data.vertices) == 0:
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        try:
            bpy.ops.object.mode_set(mode="EDIT")

            # Remove exact duplicate vertices safely
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=0.00001)

            # Recalculate normals outwards
            bpy.ops.mesh.normals_make_consistent(inside=False)

            bpy.ops.object.mode_set(mode="OBJECT")

            # Apply transform
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception as exc:
            print(f"WARNING: Mesh optimization step failed for {obj.name}: {exc}")
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass

        obj.select_set(False)

    print("Geometry optimized.")


def set_origin_to_center():
    """Set origin to the center of mass for all mesh objects."""
    for obj in get_mesh_objects():
        if len(obj.data.vertices) == 0:
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")
            obj.location = (0, 0, 0)
        except Exception as exc:
            print(f"WARNING: Origin set failed for {obj.name}: {exc}")
        obj.select_set(False)
    print("Origins set to center.")


def count_triangles() -> int:
    """Count total triangles across all mesh objects."""
    total = 0
    for obj in get_mesh_objects():
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        for poly in mesh.polygons:
            total += max(1, len(poly.vertices) - 2)
        eval_obj.to_mesh_clear()
    return total


def assign_texture_to_objects(texture_path: str):
    """Assign a texture as the base color of all mesh objects."""
    if not os.path.isfile(texture_path):
        print(f"WARNING: Texture file not found: {texture_path}")
        return False

    mat = bpy.data.materials.new(name="AssetMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.location = (-400, 0)
    tex_node.image = bpy.data.images.load(texture_path)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    for obj in get_mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    print(f"Texture assigned: {texture_path}")
    return True


def export_glb(output_path: str):
    """Export the scene as GLB."""
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
    )
    print(f"Exported GLB: {output_path}")


def export_fbx(output_path: str):
    """Export the scene as FBX."""
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        apply_scale_options="FBX_SCALE_ALL",
    )
    print(f"Exported FBX: {output_path}")


def generate_preview_render(output_dir: str, asset_id: str) -> list:
    """Generate high-quality multi-angle 2D preview renders with auto camera framing."""
    renders = []
    mesh_objs = get_mesh_objects()

    if not mesh_objs:
        print("WARNING: No mesh objects in scene to render previews.")
        return renders

    # Calculate bounding box for camera framing
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
    dist = max_dim * 2.2

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

    # Lighting setup
    bpy.ops.object.light_add(type="SUN", location=(center[0] + dist, center[1] - dist, center[2] + dist * 1.5))
    sun1 = bpy.context.active_object
    sun1.data.energy = 4.0

    bpy.ops.object.light_add(type="SUN", location=(center[0] - dist, center[1] - dist, center[2] + dist))
    sun2 = bpy.context.active_object
    sun2.data.energy = 2.0

    cx, cy, cz = center
    angles = [
        ("front", (cx, cy - dist, cz), (1.5708, 0, 0)),
        ("three_quarter", (cx + dist * 0.7, cy - dist * 0.7, cz + dist * 0.5), (1.1, 0, 0.785)),
        ("side", (cx + dist, cy, cz), (1.5708, 0, 1.5708)),
        ("top", (cx, cy, cz + dist), (0, 0, 0)),
    ]

    for name, loc, rot in angles:
        bpy.ops.object.camera_add(location=loc, rotation=rot)
        cam = bpy.context.active_object
        scene.camera = cam

        render_path = os.path.join(output_dir, f"{asset_id}_preview_{name}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        renders.append(render_path)
        print(f"Preview rendered: {render_path}")

        bpy.data.objects.remove(cam)

    bpy.data.objects.remove(sun1)
    bpy.data.objects.remove(sun2)

    return renders


def main():
    args = load_args()
    report = {
        "success": False,
        "input_model_path": args["model_path"],
        "output_model_path": "",
        "output_format": "glb",
        "triangle_count_before": 0,
        "triangle_count_after": 0,
        "texture_assigned": False,
        "preview_renders": [],
        "errors": [],
        "warnings": [],
    }

    try:
        clear_scene()
        import_model(args["model_path"])

        report["triangle_count_before"] = count_triangles()

        if args.get("optimize_mesh", True):
            optimize_geometry()
            set_origin_to_center()

        report["triangle_count_after"] = count_triangles()

        # Assign texture
        if args.get("assign_texture", True):
            tex_paths = args.get("texture_paths", {})
            # Use base_color texture, or the first available
            tex_path = tex_paths.get("base_color", "")
            if not tex_path and tex_paths:
                tex_path = next(iter(tex_paths.values()))
            if tex_path:
                report["texture_assigned"] = assign_texture_to_objects(tex_path)

        output_dir = args["output_dir"]
        asset_id = args.get("asset_spec", {}).get("asset_id", "asset")

        # Export GLB
        if args.get("create_glb", True):
            glb_path = os.path.join(output_dir, f"{asset_id}.glb")
            export_glb(glb_path)
            report["output_model_path"] = glb_path
            report["output_format"] = "glb"

        # Export FBX
        if args.get("create_fbx", False):
            fbx_path = os.path.join(output_dir, f"{asset_id}.fbx")
            export_fbx(fbx_path)

        # Preview renders
        if args.get("generate_previews", True):
            preview_dir = os.path.join(os.path.dirname(output_dir), "previews")
            os.makedirs(preview_dir, exist_ok=True)
            report["preview_renders"] = generate_preview_render(preview_dir, asset_id)

        report["success"] = True

    except Exception as e:
        report["errors"].append(str(e))
        print(f"ERROR: {e}")

    # Write report
    report_path = os.path.join(args["output_dir"], "_blender_report.json")
    os.makedirs(args["output_dir"], exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
