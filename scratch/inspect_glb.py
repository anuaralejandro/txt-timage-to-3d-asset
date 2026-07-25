import os
import trimesh
import numpy as np

ref_path = r"C:\Users\datam\Documents\jace_0_improved_color.glb"
clean_path = r"C:\Users\datam\Downloads\jace0_clean_no_droplets.glb"

print(f"Ref exists: {os.path.exists(ref_path)}")
print(f"Clean exists: {os.path.exists(clean_path)}")

if os.path.exists(ref_path):
    mesh_ref = trimesh.load(ref_path, force="mesh")
    print(f"Ref mesh: vertices {len(mesh_ref.vertices)}, faces {len(mesh_ref.faces)}")
    if hasattr(mesh_ref.visual, 'face_colors'):
        print(f"Ref face colors shape: {mesh_ref.visual.face_colors.shape}")
        unique_colors = np.unique(mesh_ref.visual.face_colors, axis=0)
        print(f"Ref unique face colors: {len(unique_colors)}")
        print(unique_colors[:15])
    if hasattr(mesh_ref.visual, 'vertex_colors'):
        print(f"Ref vertex colors shape: {mesh_ref.visual.vertex_colors.shape}")

if os.path.exists(clean_path):
    mesh_clean = trimesh.load(clean_path, force="mesh")
    print(f"Clean mesh: vertices {len(mesh_clean.vertices)}, faces {len(mesh_clean.faces)}")
    bounds = mesh_clean.bounds
    print(f"Clean bounds: min={bounds[0]}, max={bounds[1]}")
