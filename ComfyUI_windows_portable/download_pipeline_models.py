import os
from huggingface_hub import hf_hub_download

# Base ComfyUI models directory
comfy_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI", "models")
checkpoints_dir = os.path.join(comfy_base, "checkpoints")
vae_dir = os.path.join(comfy_base, "vae")
clip_vision_dir = os.path.join(comfy_base, "clip_vision")

os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(clip_vision_dir, exist_ok=True)

models_to_download = [
    {
        "repo_id": "Lykon/DreamShaper",
        "filename": "DreamShaper_8_pruned.safetensors",
        "local_dir": checkpoints_dir,
        "rename": None
    },
    {
        "repo_id": "tencent/Hunyuan3D-2mv",
        "filename": "hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors",
        "local_dir": checkpoints_dir,
        "rename": "hunyuan3d-dit-v2-mv-turbo.safetensors"
    },
    {
        "repo_id": "tencent/Hunyuan3D-2mv",
        "filename": "hunyuan3d-dit-v2-mv/model.fp16.safetensors",
        "local_dir": checkpoints_dir,
        "rename": "hunyuan3d-dit-v2-mv.safetensors"
    },
    {
        "repo_id": "tencent/Hunyuan3D-2",
        "filename": "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors",
        "local_dir": vae_dir,
        "rename": "hunyuan3d-vae-v2-0-turbo.safetensors"
    },
    {
        "repo_id": "comfyanonymous/clip_vision_g",
        "filename": "clip_vision_g.safetensors",
        "local_dir": clip_vision_dir,
        "rename": None
    }
]

print("Starting downloads. This may take a while depending on your connection...")

for item in models_to_download:
    print(f"\nDownloading {item['filename']} from {item['repo_id']}...")
    downloaded_path = hf_hub_download(
        repo_id=item['repo_id'],
        filename=item['filename'],
        local_dir=item['local_dir'],
        local_dir_use_symlinks=False
    )
    
    if item['rename']:
        target_path = os.path.join(item['local_dir'], item['rename'])
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(downloaded_path, target_path)
        
        # Clean up the empty directory structure created by HF Hub
        dir_to_clean = os.path.dirname(downloaded_path)
        try:
            os.rmdir(dir_to_clean)
        except OSError:
            pass
            
        print(f"Saved as: {target_path}")
    else:
        print(f"Saved as: {downloaded_path}")

print("\nAll downloads completed successfully!")
