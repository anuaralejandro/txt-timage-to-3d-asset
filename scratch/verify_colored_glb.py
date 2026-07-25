import trimesh
import numpy as np

colored_glb_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored.glb"
mesh = trimesh.load(colored_glb_path, force="mesh")
face_colors = mesh.visual.face_colors

print(f"Colored GLB loaded: {len(mesh.faces)} faces")
unique_colors, counts = np.unique(face_colors, axis=0, return_counts=True)
print(f"Found {len(unique_colors)} unique color components across mesh:")
for color, count in zip(unique_colors, counts):
    print(f"  Color RGBA={list(color)}: {count:5d} faces ({count/len(mesh.faces)*100:.1f}%)")
