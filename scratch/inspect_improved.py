import trimesh
import numpy as np

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")

print(f"Improved GLB loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
if hasattr(mesh.visual, 'face_colors'):
    unique_colors, counts = np.unique(mesh.visual.face_colors, axis=0, return_counts=True)
    print(f"Unique face colors in improved GLB: {len(unique_colors)}")
    for c, cnt in zip(unique_colors, counts):
        print(f"  RGBA={list(c)}: {cnt} faces ({cnt/len(mesh.faces)*100:.1f}%)")
