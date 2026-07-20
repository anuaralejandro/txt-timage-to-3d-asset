import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

import json

# ComfyUI Portable URL from API
API_URL = "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest"
TARGET_DIR = r"c:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen"
SEVEN_ZIP_URL = "https://www.7-zip.org/a/7zr.exe"

def get_latest_url():
    req = urllib.request.Request(API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for asset in data.get('assets', []):
            if 'windows_portable_nvidia' in asset['name']:
                return asset['browser_download_url']
    raise Exception("Could not find portable nvidia zip in latest release")

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    os.chdir(TARGET_DIR)
    
    # Download 7zr.exe (standalone 7-Zip extractor)
    seven_z_path = os.path.join(TARGET_DIR, "7zr.exe")
    if not os.path.exists(seven_z_path):
        print("Downloading 7zr.exe for extraction...")
        urllib.request.urlretrieve(SEVEN_ZIP_URL, seven_z_path)

    # Download ComfyUI portable
    archive_path = os.path.join(TARGET_DIR, "ComfyUI_windows_portable.7z")
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 1000:
        print("Downloading ComfyUI Portable (this might take a few minutes as it is ~1.5GB)...")
        # using curl as it shows progress and handles large files better
        url = get_latest_url()
        print(f"URL: {url}")
        run(f"curl -L -o {archive_path} {url}")
    
    print("Extracting ComfyUI...")
    # Extract
    run(f"{seven_z_path} x {archive_path} -y")

    # Move custom node
    print("Installing Custom Node...")
    custom_node_src = os.path.join(TARGET_DIR, "ComfyUI", "custom_nodes", "ComfyUI-LocalAssetFactory")
    comfy_dir = os.path.join(TARGET_DIR, "ComfyUI_windows_portable", "ComfyUI")
    custom_nodes_dest = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-LocalAssetFactory")
    
    if os.path.exists(custom_node_src):
        if not os.path.exists(custom_nodes_dest):
            print(f"Moving plugin to {custom_nodes_dest}")
            shutil.move(custom_node_src, custom_nodes_dest)
        else:
            print("Plugin already exists in target directory.")
    else:
        print("Could not find the original plugin directory. Skipping move.")

    print("\n--- Installation Complete! ---")
    print("To run ComfyUI, double click 'run_nvidia_gpu.bat' in the ComfyUI_windows_portable folder.")
    print(f"Path: {os.path.join(TARGET_DIR, 'ComfyUI_windows_portable')}")

if __name__ == "__main__":
    main()
