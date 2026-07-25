import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

input_glb = argv[0]
output_fbx = argv[1]

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB
bpy.ops.import_scene.gltf(filepath=input_glb)

# Export FBX
bpy.ops.export_scene.fbx(filepath=output_fbx, use_selection=False)

print(f"Successfully converted {input_glb} to {output_fbx}")
