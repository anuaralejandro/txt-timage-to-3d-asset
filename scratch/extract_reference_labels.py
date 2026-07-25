import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trimesh
import numpy as np
from local_asset_factory.segmentation.labels import SemanticLabel, LABEL_NAMES

ref_path = r"C:\Users\datam\Documents\jace_0_improved_color.glb"
mesh_ref = trimesh.load(ref_path, force="mesh")
face_colors = mesh_ref.visual.face_colors[:, :3] # shape (92556, 3)
centroids = mesh_ref.triangles_center # shape (92556, 3)

labels = np.zeros(len(mesh_ref.faces), dtype=int)

for i in range(len(mesh_ref.faces)):
    color = face_colors[i]
    x, y, z = centroids[i]
    r, g, b = color[0], color[1], color[2]

    # Hair / Head: Yellowish (high R, high G, low B, Y > 0.45)
    if y > 0.45 and r > 180 and g > 130:
        if y > 0.65:
            labels[i] = SemanticLabel.HAIR
        else:
            labels[i] = SemanticLabel.HEAD
    # Neck: y between 0.38 and 0.45, center
    elif 0.38 < y <= 0.45 and abs(x) < 0.15:
        labels[i] = SemanticLabel.NECK
    # Forearms: Pastel red / bright red (high R, low G/B, abs(x) > 0.35)
    elif abs(x) > 0.35:
        if x < 0:
            labels[i] = SemanticLabel.ARM_LOWER_L
        else:
            labels[i] = SemanticLabel.ARM_LOWER_R
    # Upper Arms: Dark purple / magenta (abs(x) > 0.15, y > 0.05)
    elif abs(x) > 0.15 and y > 0.05:
        if x < 0:
            labels[i] = SemanticLabel.ARM_UPPER_L
        else:
            labels[i] = SemanticLabel.ARM_UPPER_R
    # Lower Legs: light purple / bottom (y < -0.45)
    elif y < -0.45:
        if x < 0:
            labels[i] = SemanticLabel.LEG_LOWER_L
        else:
            labels[i] = SemanticLabel.LEG_LOWER_R
    # Upper Legs / Thighs: Dark green / dark reddish brown (y between -0.45 and -0.05)
    elif -0.45 <= y <= -0.05:
        if x < 0:
            labels[i] = SemanticLabel.LEG_UPPER_L
        else:
            labels[i] = SemanticLabel.LEG_UPPER_R
    # Torso: Middle body (y between -0.05 and 0.38, abs(x) <= 0.18)
    else:
        labels[i] = SemanticLabel.TORSO

print("Assigned face counts per label:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.value:2d} ({l.name:12s}): {cnt:5d} faces")

# Save ground truth labels as NPZ
np.savez_compressed(r"C:\Users\datam\Documents\jace0_reference_labels.npz", face_labels=labels)
print("Saved reference labels to jace0_reference_labels.npz")
