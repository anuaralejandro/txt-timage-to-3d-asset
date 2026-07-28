import os
import sys
import time

print("[1] Initializing paths...", flush=True)
comfy_dir = r"c:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI"
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)

print("[2] Importing PyTorch CUDA...", flush=True)
t0 = time.time()
import torch
print(f"-> PyTorch imported in {time.time() - t0:.2f}s", flush=True)

print("[3] Initializing CUDA context...", flush=True)
t0 = time.time()
print(f"-> CUDA Device: {torch.cuda.get_device_name(0)} (initialized in {time.time() - t0:.2f}s)", flush=True)

print("[4] Importing comfy.model_management...", flush=True)
t0 = time.time()
import comfy.model_management as mm
print(f"-> comfy.model_management imported in {time.time() - t0:.2f}s", flush=True)

print("[5] Checking ComfyUI device...", flush=True)
print(f"-> Selected Device: {mm.get_torch_device()}", flush=True)
