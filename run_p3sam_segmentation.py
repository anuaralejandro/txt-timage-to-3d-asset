import sys
import os
import time
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "src"
sys.path.insert(0, str(src))

from local_asset_factory.segmentation.p3sam_backend import P3SAMSonataBackend

def main():
    glb_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI\output\jace0_clean_no_droplets.glb"
    print(f"Starting Native P3-SAM 3D Mesh Segmentation on {glb_path}...")
    
    t0 = time.time()
    backend = P3SAMSonataBackend()
    backend.load()
    
    result = backend.segment_mesh_regions(glb_path, point_count=50000)
    elapsed = time.time() - t0
    
    print(f"Native P3-SAM 3D Segmentation Completed in {elapsed:.2f} seconds!")
    print(f"Detected {result.num_regions} distinct 3D part regions across {len(result.face_region_ids)} faces.")

    # Export colored preview GLB with P3-SAM region colors
    import trimesh
    from local_asset_factory.segmentation.labels import LABEL_COLORS_RGB
    mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
        
    unique_ids = np.unique(result.face_region_ids)
    color_palette = list(LABEL_COLORS_RGB.values())
    
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for f_idx, reg_id in enumerate(result.face_region_ids):
        c_idx = int(reg_id) % len(color_palette)
        r, g, b = color_palette[c_idx]
        face_colors[f_idx] = [int(r * 255), int(g * 255), int(b * 255), 255]
        
    colored_mesh = mesh.copy()
    colored_mesh.visual = trimesh.visual.ColorVisuals(mesh=colored_mesh, face_colors=face_colors)
    
    out_dir = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\p3sam_dynamic_test"
    os.makedirs(out_dir, exist_ok=True)
    colored_glb = os.path.join(out_dir, "p3sam_colored.glb")
    colored_mesh.export(colored_glb)
    print(f"Exported P3-SAM colored mesh to: {colored_glb}")
    
    final_glb = os.path.join(out_dir, "p3sam_godmode_solid.glb")
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
    script = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LocalAssetFactory\blender\separate_and_cap_by_color.py"
    
    cmd = f'"{blender_exe}" --background --python "{script}" -- "{colored_glb}" "{final_glb}"'
    os.system(cmd)
    print(f"🎉 Fully Capped P3-SAM Asset Generated at: {final_glb}")

if __name__ == "__main__":
    main()
