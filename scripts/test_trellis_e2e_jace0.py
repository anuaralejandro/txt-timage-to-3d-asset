import os
import sys
import torch

# Ensure ComfyUI path and IF_Trellis path are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
comfy_dir = os.path.join(root_dir, "ComfyUI_windows_portable", "ComfyUI")
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)

if "ComfyUI-IF_Trellis" not in sys.path:
    sys.path.insert(0, os.path.join(comfy_dir, "custom_nodes", "ComfyUI-IF_Trellis"))

from PIL import Image
import folder_paths

print("Initializing Trellis Pipeline Test...")

from IF_TrellisCheckpointLoader import IF_TrellisCheckpointLoader
from IF_Trellis import IF_TrellisImageTo3D

print("Step 1: Loading Trellis Checkpoint...")
loader = IF_TrellisCheckpointLoader()
model_dict, = loader.load_model(
    model_name="TRELLIS-image-large",
    dinov2_model="dinov2_vitl14_reg",
    attn_backend="sdpa",
    sparse_backend="spconv",
    spconv_algo="implicit_gemm",
    use_fp16=True,
    smooth_k=False
)
print("-> Trellis Checkpoint Loaded Successfully!")

print("Step 2: Loading Multi-View Images from C:\\Users\\datam\\Downloads\\jace0_limpios...")
img_dir = r"C:\Users\datam\Downloads\jace0_limpios"
views = ["front.png", "right.png", "back.png", "left.png"]
loaded_images = []

for v in views:
    v_path = os.path.join(img_dir, v)
    if os.path.exists(v_path):
        img = Image.open(v_path).convert("RGB")
        # Convert PIL Image to PyTorch Tensor [H, W, C] normalized (0..1)
        import numpy as np
        img_np = torch.from_numpy(np.array(img)).float() / 255.0
        loaded_images.append(img_np)
        print(f"-> Loaded view: {v} ({img.size})")

# Stack to [B, H, W, C] tensor
torch_images = torch.stack(loaded_images, dim=0)

trellis_node = IF_TrellisImageTo3D()

print("Step 3: Running Trellis 3D Generation (Multi-View Mode)...")
glb_path, video_path, texture_img = trellis_node.image_to_3d(
    model=model_dict,
    mode="multi",
    images=torch_images,
    seed=42,
    ss_guidance_strength=7.5,
    ss_sampling_steps=12,
    slat_guidance_strength=3.0,
    slat_sampling_steps=12,
    mesh_simplify=0.95,
    texture_size=1024,
    texture_mode="fast",
    fps=15,
    multimode="stochastic",
    project_name="jace0_trellis_multiview",
    render_video=False,
    save_glb=True,
    save_gaussian=False,
    save_texture=True,
    save_wireframe=False
)

print("\n=======================================================")
print(f"🎉 TRELLIS E2E 3D GENERATION SUCCESSFUL!")
print(f"GLB Output Path: {glb_path}")
print("=======================================================")
