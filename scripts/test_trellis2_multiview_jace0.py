import os
import sys
import torch

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
trellis2_node_dir = os.path.join(root_dir, "ComfyUI_windows_portable", "ComfyUI", "custom_nodes", "ComfyUI-Trellis2")
comfy_dir = os.path.join(root_dir, "ComfyUI_windows_portable", "ComfyUI")

if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)
if trellis2_node_dir not in sys.path:
    sys.path.insert(0, trellis2_node_dir)

from PIL import Image
import numpy as np

print("Step 0: Importing nodes...")
from nodes import (
    Trellis2LoadModel, 
    Trellis2MeshWithVoxelMultiViewGenerator, 
    Trellis2OvoxelExportToGLB
)

print("Step 1: Loading Trellis2 Model...")
loader = Trellis2LoadModel()
pipeline = loader.process(
    modelname="microsoft/TRELLIS.2-4B",
    backend="sdpa",
    device="cuda",
    low_vram=True,
    keep_models_loaded=True,
    conv_backend="spconv",
    sparse_backend="sdpa",
    use_reconviagen=False
)[0]

print("Step 2: Loading Multi-View Images from C:\\Users\\datam\\Downloads\\jace0_limpios...")
folder_path = r"C:\Users\datam\Downloads\jace0_limpios"

def load_img(name):
    p = os.path.join(folder_path, name)
    if os.path.exists(p):
        img = Image.open(p).convert("RGBA")
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0) # [1, H, W, 4]
        return tensor
    return None

front_tensor = load_img("front.png")
back_tensor = load_img("back.png")
left_tensor = load_img("left.png")
right_tensor = load_img("right.png")

print(f"-> Front: {front_tensor.shape if front_tensor is not None else None}")
print(f"-> Back:  {back_tensor.shape if back_tensor is not None else None}")
print(f"-> Left:  {left_tensor.shape if left_tensor is not None else None}")
print(f"-> Right: {right_tensor.shape if right_tensor is not None else None}")

print("Step 3: Generating Multi-View 3D Asset (8GB Optimized)...")
generator = Trellis2MeshWithVoxelMultiViewGenerator()

mesh_output = generator.process(
    pipeline=pipeline,
    front_image=front_tensor,
    back_image=back_tensor,
    left_image=left_tensor,
    right_image=right_tensor,
    seed=12345,
    pipeline_type="1024",
    sparse_structure_steps=12,
    sparse_structure_guidance_strength=6.5,
    sparse_structure_guidance_rescale=0.05,
    sparse_structure_rescale_t=4.0,
    shape_steps=12,
    shape_guidance_strength=6.5,
    shape_guidance_rescale=0.05,
    shape_rescale_t=4.0,
    texture_steps=12,
    texture_guidance_strength=3.0,
    texture_guidance_rescale=0.2,
    texture_rescale_t=3.0,
    max_num_tokens=999999,
    sparse_structure_resolution=32,
    generate_texture_slat=True,
    sparse_structure_guidance_interval_start=0.1,
    sparse_structure_guidance_interval_end=1.0,
    shape_guidance_interval_start=0.1,
    shape_guidance_interval_end=1.0,
    texture_guidance_interval_start=0.0,
    texture_guidance_interval_end=0.9,
    use_tiled_decoder=True,
    front_axis="z",
    blend_temperature=1.0,
    sampler="euler",
    fill_holes=False,
    hole_iterations=1,
    verbose=True,
    dino_lock=0.0,
    dino_substeps=4,
    hole_fill_algorithm="flood_fill",
    dino_foundation_cap=1.0,
    keep_only_shell=True
)

mesh = mesh_output[0]

print("Step 4: Exporting GLB Asset...")
exporter = Trellis2OvoxelExportToGLB()
glb_mesh = exporter.process(mesh, resolution=1024, texture_size=2048, target_face_num=2000000)[0]

out_glb_path = os.path.join(folder_path, "jace0_3d_asset.glb")
glb_mesh.export(out_glb_path)

print("\n=======================================================")
print(f"🎉 MULTI-VIEW 3D ASSET GENERATED SUCCESSFULLY!")
print(f"Output File: {out_glb_path}")
print(f"Vertices: {len(glb_mesh.vertices)}, Faces: {len(glb_mesh.faces)}")
print("=======================================================")
