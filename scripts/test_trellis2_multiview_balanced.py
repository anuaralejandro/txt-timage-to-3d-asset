import os
import sys
import gc

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

print("Initializing VisualBruno ComfyUI-Trellis2 MultiView Balanced AAA Test...", flush=True)

print("Step 0: Importing nodes...", flush=True)
from nodes import Trellis2LoadModel, Trellis2MeshWithVoxelMultiViewGenerator, Trellis2OvoxelExportToGLB
print("-> Nodes imported successfully!", flush=True)

print("Step 1: Loading Trellis2 Model (VisualBruno)...", flush=True)
loader = Trellis2LoadModel()
model_tuple = loader.process(
    modelname="microsoft/TRELLIS.2-4B",
    backend="sdpa",
    device="cuda",
    low_vram=True,
    keep_models_loaded=False, # Free model memory dynamically for 8GB VRAM stability
    conv_backend="spconv",
    sparse_backend="sdpa",
    use_reconviagen=False
)
model_dict = model_tuple[0]
print("-> Trellis2 Model Loaded Successfully!", flush=True)

print("Step 2: Loading Images from C:\\Users\\datam\\Downloads\\jace0_limpios...")
def load_img(path):
    img = Image.open(path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

front = load_img(r"C:\Users\datam\Downloads\jace0_limpios\front.png")
back = load_img(r"C:\Users\datam\Downloads\jace0_limpios\back.png")
left = load_img(r"C:\Users\datam\Downloads\jace0_limpios\left.png")
right = load_img(r"C:\Users\datam\Downloads\jace0_limpios\right.png")

print("Step 3: Generating 3D Mesh with Voxel (8GB VRAM Optimized, MultiView, High Quality)...")
generator = Trellis2MeshWithVoxelMultiViewGenerator()
mesh_output = generator.process(
    pipeline=model_dict,
    front_image=front,
    seed=42,
    pipeline_type="1024_cascade",
    sparse_structure_steps=25,
    shape_steps=25,
    texture_steps=25,
    max_num_tokens=49152,
    sparse_structure_resolution=32,
    generate_texture_slat=True,
    use_tiled_decoder=True,
    sampler="euler",
    fill_holes=True,
    hole_iterations=2,
    hole_fill_algorithm="flood_fill",
    keep_only_shell=True,
    back_image=back,
    left_image=left,
    right_image=right,
    sparse_structure_guidance_strength=7.0,
    sparse_structure_guidance_rescale=0.05,
    sparse_structure_rescale_t=4.0,
    shape_guidance_strength=7.0,
    shape_guidance_rescale=0.05,
    shape_rescale_t=4.0,
    texture_guidance_strength=3.5,
    texture_guidance_rescale=0.20,
    texture_rescale_t=3.0,
    sparse_structure_guidance_interval_start=0.10,
    sparse_structure_guidance_interval_end=1.00,
    shape_guidance_interval_start=0.10,
    shape_guidance_interval_end=1.00,
    texture_guidance_interval_start=0.00,
    texture_guidance_interval_end=0.90,
    front_axis="z",
    blend_temperature=1.0,
    verbose=False,
    dino_lock=0.0,
    dino_substeps=4,
    dino_foundation_cap=1.0
)

mesh = mesh_output[0]

print("Step 4: Exporting to GLB...")
exporter = Trellis2OvoxelExportToGLB()
glb_res = exporter.process(mesh, resolution=1024, texture_size=2048, target_face_num=1000000)[0]

output_path = r"C:\Users\datam\Downloads\jace0_limpios\jace0_3d_asset_hq.glb"
glb_res.export(output_path)
print(f"-> Exported cleanly to: {output_path}")

print("\n=======================================================")
print(f"🎉 VISUALBRUNO TRELLIS2 MULTIVIEW 3D ASSET GENERATED!")
print(f"Output saved to: {output_path}")
print("=======================================================")
