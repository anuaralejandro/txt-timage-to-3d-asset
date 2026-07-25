import trimesh
import numpy as np

glb_path = r"C:\Users\datam\Downloads\jace0_clean_no_droplets.glb"
mesh = trimesh.load(glb_path, force="mesh")
centroids = mesh.triangles_center

X = centroids[:, 0]
Y = centroids[:, 1]
Z = centroids[:, 2]

print(f"X bounds: min={X.min():.4f}, max={X.max():.4f}")
print(f"Y bounds: min={Y.min():.4f}, max={Y.max():.4f}")
print(f"Z bounds: min={Z.min():.4f}, max={Z.max():.4f}")

# Let's inspect percentiles of Y (height)
print("\nY percentiles:")
for p in [0, 10, 25, 40, 50, 60, 75, 85, 95, 100]:
    print(f"  {p:3d}%: Y = {np.percentile(Y, p):+.4f}")

# Let's inspect percentiles of X (width) for Y > 0.0
y_positive_x = X[Y > 0.0]
print("\nX percentiles for upper body (Y > 0.0):")
for p in [0, 10, 25, 50, 75, 90, 100]:
    print(f"  {p:3d}%: X = {np.percentile(y_positive_x, p):+.4f}")
