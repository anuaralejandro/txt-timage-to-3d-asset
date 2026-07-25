import trimesh
import numpy as np
from sklearn.cluster import KMeans

ref_path = r"C:\Users\datam\Documents\jace_0_improved_color.glb"
mesh_ref = trimesh.load(ref_path, force="mesh")
face_colors = mesh_ref.visual.face_colors[:, :3] # RGB

# Centroids of faces
centroids = mesh_ref.triangles_center

print(f"Face colors shape: {face_colors.shape}")

# Perform K-Means or unique color analysis
# Let's cluster into ~10-13 dominant color centroids
kmeans = KMeans(n_clusters=13, random_state=42).fit(face_colors)
centers = kmeans.cluster_centers_

print("Discovered 13 dominant RGB color centroids in reference model:")
for idx, c in enumerate(centers):
    # Find average height (Y) and side (X) for faces in this cluster
    mask = (kmeans.labels_ == idx)
    avg_y = centroids[mask, 1].mean()
    avg_x = centroids[mask, 0].mean()
    count = mask.sum()
    print(f"Cluster {idx:2d}: RGB=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), Count={count:5d}, Avg_Y={avg_y:.2f}, Avg_X={avg_x:.2f}")
