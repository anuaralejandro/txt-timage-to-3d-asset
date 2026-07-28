import os
import sys

print("[1/6] Setting up paths...", flush=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
comfy_dir = os.path.join(root_dir, "ComfyUI_windows_portable", "ComfyUI")
trellis2_dir = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-Trellis2")

if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)
if trellis2_dir not in sys.path:
    sys.path.insert(0, trellis2_dir)

print("[2/6] Importing torch...", flush=True)
import torch
print(f"-> PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)

print("[3/6] Importing meshlib & nvdiffrast...", flush=True)
import meshlib.mrmeshnumpy
import nvdiffrast.torch as dr
print("-> meshlib and nvdiffrast imported successfully", flush=True)

print("[4a] Importing triton...", flush=True)
try:
    import triton
    print(f"-> triton version: {triton.__version__}", flush=True)
except Exception as e:
    print(f"-> triton import failed: {e}", flush=True)

print("[4b] Checking flex_gemm...", flush=True)
print("-> flex_gemm bypassed (using PyTorch native grid_sample_3d fallback)", flush=True)

print("[5/6] Importing Trellis2 nodes...", flush=True)
from nodes import Trellis2LoadModel, Trellis2MeshWithVoxelGenerator
print("-> Nodes imported successfully!", flush=True)
