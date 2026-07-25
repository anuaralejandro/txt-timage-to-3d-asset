import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import trimesh
import numpy as np
from local_asset_factory.segmentation.labels import SemanticLabel

improved_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\character_segmented_colored_improved.glb"
mesh = trimesh.load(improved_path, force="mesh")

face_colors = mesh.visual.face_colors[:, :3] # (92556, 3)
centroids = mesh.triangles_center # (92556, 3)

labels = np.zeros(len(mesh.faces), dtype=int)

# Map colors to 13 classes
for i in range(len(mesh.faces)):
    r, g, b = face_colors[i]
    x, y, z = centroids[i]

    # Hair / Head: Yellow (High R, High G, low B or high overall yellow)
    if r > 180 and g > 160 and b < 140:
        labels[i] = SemanticLabel.HAIR
    elif r > 180 and g > 160 and b >= 140:
        labels[i] = SemanticLabel.HEAD
    # Neck: y in [0.38, 0.45], central
    elif 0.38 < y <= 0.45 and abs(x) < 0.15:
        labels[i] = SemanticLabel.NECK
    # Forearms / Lower Arms: Pastel Red / Reddish
    elif abs(x) > 0.35:
        labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R
    # Upper Arms: Dark Purple
    elif abs(x) > 0.15 and y > 0.05:
        labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R
    # Lower Legs: Light Purple / Feet
    elif y < -0.45:
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R
    # Upper Legs / Thighs: Dark Green / Dark Pink
    elif -0.45 <= y <= -0.05:
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
    # Torso: Grayish / Brown
    else:
        labels[i] = SemanticLabel.TORSO

# Perform topological hair propagation:
# Find all faces connected to HAIR seed faces that do NOT intersect TORSO
face_adjacency = mesh.face_adjacency
adj_graph = {idx: set() for idx in range(len(mesh.faces))}
for f1, f2 in face_adjacency:
    adj_graph[f1].add(f2)
    adj_graph[f2].add(f1)

# Seed hair faces (Y > 0.45 and Yellow)
hair_seeds = set(np.where((centroids[:, 1] > 0.45) & (labels == SemanticLabel.HAIR))[0])
visited_hair = set(hair_seeds)
queue = list(hair_seeds)

while queue:
    curr = queue.pop(0)
    for neighbor in adj_graph[curr]:
        if neighbor not in visited_hair:
            # If neighbor is NOT torso skin surface, propagate HAIR
            if labels[neighbor] != SemanticLabel.TORSO:
                visited_hair.add(neighbor)
                labels[neighbor] = SemanticLabel.HAIR
                queue.append(neighbor)

print("Topological Hair Propagation Complete!")
counts = {l.name: (labels == l.value).sum() for l in SemanticLabel}
for name, cnt in counts.items():
    print(f"  {name:15s}: {cnt:5d} faces")

# Save exact ground truth improved labels
np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels_improved.npz", face_labels=labels)
print("Saved face_labels_improved.npz")
