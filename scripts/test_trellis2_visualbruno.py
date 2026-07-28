import os
import sys

os.environ["SPARSE_CONV_BACKEND"] = "spconv"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
comfy_dir = os.path.join(root_dir, "ComfyUI_windows_portable", "ComfyUI")
trellis2_dir = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-Trellis2")

if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)
if trellis2_dir not in sys.path:
    sys.path.insert(0, trellis2_dir)

from PIL import Image
import numpy as np

print("Initializing VisualBruno ComfyUI-Trellis2 (8GB VRAM Optimized) Test...", flush=True)

print("Step 0: Importing nodes...", flush=True)
from nodes import Trellis2LoadModel, Trellis2MeshWithVoxelGenerator, Trellis2OvoxelExportToGLB
print("-> Nodes imported successfully!", flush=True)

print("Step 1: Loading Trellis2 Model (VisualBruno)...", flush=True)
loader = Trellis2LoadModel()
model_tuple = loader.process(
    modelname="microsoft/TRELLIS.2-4B",
    backend="sdpa",
    device="cuda",
    low_vram=True,
    keep_models_loaded=True,
    conv_backend="spconv",
    sparse_backend="sdpa",
    use_reconviagen=False
)
model_dict = model_tuple[0]
print("-> Trellis2 Model Loaded Successfully!", flush=True)

print("Step 2: Loading Image from C:\\Users\\datam\\Downloads\\jace0_limpios\\front.png...")
img_path = r"C:\Users\datam\Downloads\jace0_limpios\front.png"
img = Image.open(img_path).convert("RGB")
img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
if img_tensor.ndim == 3:
    img_tensor = img_tensor.unsqueeze(0)

print("Step 3: Generating 3D Mesh with Voxel (8GB Optimized)...")
generator = Trellis2MeshWithVoxelGenerator()
mesh_output = generator.process(
    pipeline=model_dict,
    image=img_tensor,
    seed=42,
    pipeline_type="1024_cascade",
    sparse_structure_steps=12,
    shape_steps=12,
    texture_steps=12,
    max_num_tokens=49152,
    max_views=4,
    sparse_structure_resolution=32,
    generate_texture_slat=True,
    use_tiled_decoder=True,
    sampler="euler",
    fill_holes=True,
    hole_iterations=1,
    hole_fill_algorithm="flood_fill",
    keep_only_shell=True
)

mesh = mesh_output[0]

print("Step 4: Exporting to GLB...")
exporter = Trellis2OvoxelExportToGLB()
glb_res = exporter.process(mesh)[0]
print(f"-> Exported to: {glb_res}")

print("\n=======================================================")
print(f"🎉 VISUALBRUNO TRELLIS2 E2E 3D GENERATION SUCCESSFUL!")
print(f"Output: {glb_res}")
print("=======================================================")
