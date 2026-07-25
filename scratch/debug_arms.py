import trimesh
import numpy as np

glb_path = r"C:\Users\datam\Downloads\jace0_clean_no_droplets.glb"
mesh = trimesh.load(glb_path, force="mesh")
centroids = mesh.triangles_center

X = centroids[:, 0]
Y = centroids[:, 1]

idx = np.where((Y > 0.1) & (Y < 0.5) & (np.abs(X) > 0.2))[0]
print(f"Number of faces with Y in [0.1, 0.5] and abs(X) > 0.2: {len(idx)}")
if len(idx) > 0:
    print("Sample X values:", X[idx[:10]])
    print("Sample Y values:", Y[idx[:10]])
