import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trimesh
import numpy as np
from local_asset_factory.segmentation.labels import SemanticLabel, LABEL_NAMES
from local_asset_factory.segmentation.postprocess import SegmentationPostProcessor

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")
face_colors = mesh.visual.face_colors[:, :3]
centroids = mesh.triangles_center
num_faces = len(mesh.faces)

labels = np.zeros(num_faces, dtype=int)

for i in range(num_faces):
    r, g, b = face_colors[i]
    x, y, z = centroids[i]

    # 1. Hair and Head: Y > 0.45 or (RGB is Head/Hair purple/yellow in reference)
    if y > 0.42 or (r == 165 and g == 87 and b == 178 and y > 0.15):
        if y > 0.65 or z < -0.05:
            labels[i] = SemanticLabel.HAIR
        else:
            labels[i] = SemanticLabel.HEAD
    # 2. Lower Arms / Forearms: RGB=[204, 254, 254] or abs(X) > 0.35
    elif (r == 204 and g == 254 and b == 254) or (abs(x) > 0.35 and y > 0.05):
        labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R
    # 3. Upper Arms: RGB=[232, 251, 55] or abs(X) in [0.15, 0.35]
    elif (r == 232 and g == 251 and b == 55) or (abs(x) > 0.15 and abs(x) <= 0.35 and y > 0.05):
        labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R
    # 4. Lower Legs / Boots: RGB=[226, 232, 145] or Y < -0.42
    elif (r == 226 and g == 232 and b == 145) or (y < -0.42 and abs(x) > 0.02):
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R
    # 5. Upper Legs / Thighs: RGB=[130, 204, 153] or Y in [-0.42, -0.02]
    elif (r == 130 and g == 204 and b == 153) or (-0.42 <= y <= -0.02 and abs(x) > 0.03):
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
    # 6. Neck
    elif 0.38 < y <= 0.42 and abs(x) < 0.15:
        labels[i] = SemanticLabel.NECK
    # 7. Torso: RGB=[153, 55, 7] or central body
    elif (r == 153 and g == 55 and b == 7) or (-0.02 < y <= 0.38 and abs(x) <= 0.16):
        labels[i] = SemanticLabel.TORSO
    else:
        # Inner thigh / crotch seam: assign based on Y
        if y < -0.42:
            labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R
        elif y < -0.02:
            labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
        else:
            labels[i] = SemanticLabel.TORSO

# Apply topological cleanup to fill crotch seam & smooth boundary rings
postprocessor = SegmentationPostProcessor(min_island_faces=15, smoothing_iterations=3)
labels = postprocessor.process(improved_path, labels)

print("Class Breakdown after Perfect Anatomical Mapping:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.name:15s}: {cnt:5d} faces ({cnt/num_faces*100:4.1f}%)")

# Save face labels
np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels.npz", face_labels=labels)
print("Saved face_labels.npz successfully!")
