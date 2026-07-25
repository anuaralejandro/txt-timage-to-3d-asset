import trimesh
import numpy as np

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")
face_colors = mesh.visual.face_colors[:, :3]

unique_colors, counts = np.unique(face_colors, axis=0, return_counts=True)
print("Unique face colors in character_segmented_colored_improved.glb:")
for color, count in zip(unique_colors, counts):
    pct = count / len(mesh.faces) * 100
    if pct > 0.05:
        print(f"  RGB=({color[0]:3d}, {color[1]:3d}, {color[2]:3d}): {count:5d} faces ({pct:.1f}%)")
