"""
Runner script to execute E2E native PyTorch Hunyuan3D-Part GPU 3D Segmentation and generate preview renders.
"""

import sys
import os
import glob
import logging
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "src"
custom_nodes = root / "ComfyUI_windows_portable" / "ComfyUI" / "custom_nodes" / "ComfyUI-LocalAssetFactory"

for p in [root, src, custom_nodes]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from local_asset_factory.segmentation.hunyuan_part_backend import Hunyuan3DPartSegmenter
from local_asset_factory.segmentation.export import SemanticExporter

def main():
    # Find best target GLB
    glb_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI\output\jace0_clean_no_droplets.glb"
    if not os.path.exists(glb_path):
        glb_path = r"C:\Users\datam\Downloads\jace0_clean_no_droplets.glb"

    if not os.path.exists(glb_path):
        glb_files = glob.glob(r'C:\Users\datam\Downloads\**\*.glb', recursive=True)
        if glb_files:
            glb_path = glb_files[0]

    output_dir = os.path.join(root, "output", "hunyuan_part_jace0_e2e")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Running E2E Hunyuan3D-Part GPU 3D Segmentation ===")
    print(f"Input GLB: {glb_path}")
    print(f"Output Directory: {output_dir}")

    segmenter = Hunyuan3DPartSegmenter(low_vram_mode=True)
    face_labels, face_confidences, metrics = segmenter.segment_glb(glb_path)

    exporter = SemanticExporter()
    seg_glb, labels_npz, manifest_json, colored_glb = exporter.export_all(
        source_glb_path=glb_path,
        output_dir=output_dir,
        face_labels=face_labels,
        face_confidences=face_confidences,
        settings={"gpu_metrics": metrics},
    )

    print("=== Hunyuan3D-Part GPU 3D Segmentation Completed ===")
    print(f"Segmented GLB: {seg_glb}")
    print(f"Colored GLB: {colored_glb}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Metrics: {metrics}")

    # Render Blender Previews
    blender_script = os.path.join(custom_nodes, "blender", "render_glb_previews.py")
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
    if not os.path.exists(blender_exe):
        blender_exe = "blender"

    if os.path.exists(blender_script):
        print("=== Rendering 2D Previews with Blender Headless ===")
        cmd = [
            blender_exe, "--background", "--python", blender_script, "--",
            colored_glb, output_dir, "jace0_e2e"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(res.stdout)
        except Exception as err:
            print(f"Blender render error (non-fatal): {err}")

if __name__ == "__main__":
    main()
