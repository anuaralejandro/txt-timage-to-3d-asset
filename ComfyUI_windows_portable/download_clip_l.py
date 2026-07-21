import os
import shutil
from huggingface_hub import hf_hub_download

def download():
    models_dir = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI\models\clip_vision"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading clip_vision_l.safetensors (ViT-L-14) from OpenAI (this might take a minute, it's ~1.7GB)...")
    try:
        file_path = hf_hub_download(
            repo_id="openai/clip-vit-large-patch14",
            filename="model.safetensors",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        # Rename it to clip_vision_l.safetensors
        final_path = os.path.join(models_dir, "clip_vision_l.safetensors")
        if os.path.exists(file_path) and file_path != final_path:
            shutil.move(file_path, final_path)
            print(f"Successfully downloaded and renamed to: {final_path}")
        else:
            print(f"File exists at {final_path}")
            
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    download()
