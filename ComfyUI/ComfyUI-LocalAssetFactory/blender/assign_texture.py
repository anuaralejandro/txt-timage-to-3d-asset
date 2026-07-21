"""
Blender Script — assign_texture.py
Standalone script to assign a texture to all mesh objects in a scene.

Usage:
    blender --background model.glb --python assign_texture.py -- texture.png output.glb
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
        print("Usage: blender --background model.glb --python assign_texture.py -- texture.png output.glb")
        sys.exit(1)

    extra_args = argv[argv.index("--") + 1 :]
    if len(extra_args) < 2:
        print("ERROR: Need texture_path and output_path after '--'")
        sys.exit(1)

    texture_path = extra_args[0]
    output_path = extra_args[1]

    if not os.path.isfile(texture_path):
        print(f"ERROR: Texture not found: {texture_path}")
        sys.exit(1)

    # Create material with texture
    mat = bpy.data.materials.new(name="TextureMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (-400, 0)
    tex.image = bpy.data.images.load(texture_path)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Assign to all meshes
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    # Export
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")
    elif ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=output_path)
    else:
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")

    print(f"Saved with texture: {output_path}")


if __name__ == "__main__":
    main()
