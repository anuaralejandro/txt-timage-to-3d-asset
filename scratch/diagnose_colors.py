import trimesh
import numpy as np

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")
face_colors = mesh.visual.face_colors[:, :3]
centroids = mesh.triangles_center

# Print RGB of faces at specific spatial locations
print("Forearms (abs(X) > 0.45, Y in [0.2, 0.4]):")
idx_forearm = np.where((np.abs(centroids[:, 0]) > 0.45) & (centroids[:, 1] > 0.2))[0]
print(f"Count: {len(idx_forearm)}")
if len(idx_forearm) > 0:
    print("Sample RGBs:", face_colors[idx_forearm[:10]])

print("\nLower Legs / Feet (Y < -0.6):")
idx_feet = np.where(centroids[:, 1] < -0.6)[0]
print(f"Count: {len(idx_feet)}")
if len(idx_feet) > 0:
    print("Sample RGBs:", face_colors[idx_feet[:10]])

print("\nThighs (Y in [-0.3, -0.1]):")
idx_thighs = np.where((centroids[:, 1] > -0.3) & (centroids[:, 1] < -0.1))[0]
print(f"Count: {len(idx_thighs)}")
if len(idx_thighs) > 0:
    print("Sample RGBs:", face_colors[idx_thighs[:10]])
