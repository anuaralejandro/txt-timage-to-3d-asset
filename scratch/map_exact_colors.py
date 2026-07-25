import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trimesh
import numpy as np
from local_asset_factory.segmentation.labels import SemanticLabel

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")
face_colors = mesh.visual.face_colors[:, :3]
centroids = mesh.triangles_center

# Print all distinct face colors in the user's improved GLB and their spatial centroids (Y, X, Z)
unique_colors, inverse, counts = np.unique(face_colors, axis=0, return_inverse=True, return_counts=True)

print(f"Total faces: {len(mesh.faces)}")
print("Distinct Colors in character_segmented_colored_improved.glb:")
for idx, (color, count) in enumerate(zip(unique_colors, counts)):
    mask = (inverse == idx)
    avg_y = centroids[mask, 1].mean()
    avg_x = centroids[mask, 0].mean()
    min_x = centroids[mask, 0].min()
    max_x = centroids[mask, 0].max()
    print(f"Color {idx:2d}: RGB=({color[0]:3d}, {color[1]:3d}, {color[2]:3d}) | Count={count:5d} ({count/len(mesh.faces)*100:4.1f}%) | Avg_Y={avg_y:+.2f}, Avg_X={avg_x:+.2f}, X_range=[{min_x:+.2f}, {max_x:+.2f}]")
