import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trimesh
import numpy as np
from local_asset_factory.segmentation.labels import SemanticLabel
from local_asset_factory.segmentation.postprocess import SegmentationPostProcessor

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")
face_colors = mesh.visual.face_colors[:, :3].astype(float)
centroids = mesh.triangles_center
num_faces = len(mesh.faces)

# Define 7 dominant RGB centroids and their mapping to SemanticLabel based on spatial position
COLOR_MAP = [
    # (Target RGB, Label Left, Label Right, Y_threshold, X_threshold)
    {"rgb": np.array([165.0, 87.0, 178.0]), "label_head": SemanticLabel.HEAD, "label_hair": SemanticLabel.HAIR},
    {"rgb": np.array([153.0, 55.0, 7.0]), "label": SemanticLabel.TORSO},
    {"rgb": np.array([80.0, 180.0, 248.0]), "label_l": SemanticLabel.LEG_LOWER_L, "label_r": SemanticLabel.LEG_LOWER_R},
    {"rgb": np.array([130.0, 204.0, 153.0]), "label_l": SemanticLabel.LEG_UPPER_L, "label_r": SemanticLabel.LEG_UPPER_R},
    {"rgb": np.array([204.0, 254.0, 254.0]), "label_l": SemanticLabel.ARM_LOWER_L, "label_r": SemanticLabel.ARM_LOWER_R},
    {"rgb": np.array([232.0, 251.0, 55.0]), "label_l": SemanticLabel.ARM_UPPER_L, "label_r": SemanticLabel.ARM_UPPER_R},
]

# Calculate Euclidean distance to each of the 6 dominant centroids
centroids_rgb = np.array([c["rgb"] for c in COLOR_MAP]) # (6, 3)

# For each face, find index of closest centroid
dists = np.linalg.norm(face_colors[:, None, :] - centroids_rgb[None, :, :], axis=2) # (num_faces, 6)
closest_idx = np.argmin(dists, axis=1)

labels = np.zeros(num_faces, dtype=int)

for i in range(num_faces):
    c_idx = closest_idx[i]
    x, y, z = centroids[i]

    if c_idx == 0: # Head / Hair
        if y > 0.65 or z < -0.05:
            labels[i] = SemanticLabel.HAIR
        else:
            labels[i] = SemanticLabel.HEAD
    elif c_idx == 1: # Torso
        labels[i] = SemanticLabel.TORSO
    elif c_idx == 2: # Lower Legs
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R
    elif c_idx == 3: # Upper Legs / Thighs
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
    elif c_idx == 4: # Lower Arms
        labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R
    elif c_idx == 5: # Upper Arms
        labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R

# Fill inner thigh crotch seam if unassigned or ambiguous
postprocessor = SegmentationPostProcessor(min_island_faces=10, smoothing_iterations=2)
labels = postprocessor.process(improved_path, labels)

print("Exact Euclidian Distance Class Breakdown:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.name:15s}: {cnt:5d} faces ({cnt/num_faces*100:4.1f}%)")

np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels.npz", face_labels=labels)
print("Saved face_labels.npz successfully!")
