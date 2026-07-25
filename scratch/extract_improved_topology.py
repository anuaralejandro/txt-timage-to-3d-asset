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

# Build color mapping based on unique RGB centroids
labels = np.zeros(len(mesh.faces), dtype=int)

for i in range(len(mesh.faces)):
    r, g, b = face_colors[i]
    x, y, z = centroids[i]

    # Head and Hair (Top yellow/purple components)
    if y > 0.45 or (r > 150 and g > 150) or (r > 150 and b > 150 and y > 0.10):
        if y > 0.65 or z < -0.05: # Ponytail hanging back
            labels[i] = SemanticLabel.HAIR
        else:
            labels[i] = SemanticLabel.HEAD
    # Forearms / Lower Arms (Pastel red / Cyan RGB=[204, 254, 254])
    elif (r > 190 and g > 240 and b > 240) or abs(x) > 0.35:
        labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R
    # Upper Arms (Bright yellow/green RGB=[232, 251, 55])
    elif (r > 200 and g > 240 and b < 100 and y > 0.05) or (abs(x) > 0.16 and y > 0.05):
        labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R
    # Lower Legs / Feet (Light blue RGB=[80, 180, 248])
    elif (g > 150 and b > 230) or y < -0.42:
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R
    # Upper Legs / Thighs (Dark green RGB=[130, 204, 153])
    elif (g > 180 and b < 180 and y < -0.02) or (-0.42 <= y <= -0.02 and abs(x) > 0.05):
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
    # Neck
    elif 0.38 < y <= 0.45 and abs(x) < 0.15:
        labels[i] = SemanticLabel.NECK
    # Torso (Brown RGB=[153, 55, 7])
    else:
        labels[i] = SemanticLabel.TORSO

# Perform Topological Connected-Component Hair Propagation
# BFS from hair seeds down connected mesh vertices
face_adjacency = mesh.face_adjacency
adj_graph = {idx: set() for idx in range(len(mesh.faces))}
for f1, f2 in face_adjacency:
    adj_graph[f1].add(f2)
    adj_graph[f2].add(f1)

hair_seeds = set(np.where(labels == SemanticLabel.HAIR)[0])
visited = set(hair_seeds)
queue = list(hair_seeds)

while queue:
    curr = queue.pop(0)
    for nbr in adj_graph[curr]:
        if nbr not in visited:
            # If neighbor is NOT part of torso/legs, propagate HAIR
            if labels[nbr] not in (SemanticLabel.TORSO, SemanticLabel.LEG_UPPER_L, SemanticLabel.LEG_UPPER_R, SemanticLabel.LEG_LOWER_L, SemanticLabel.LEG_LOWER_R):
                visited.add(nbr)
                labels[nbr] = SemanticLabel.HAIR
                queue.append(nbr)

print("Label assignment after topological graph propagation:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.name:15s}: {cnt:5d} faces ({cnt/len(mesh.faces)*100:.1f}%)")

np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels.npz", face_labels=labels)
print("Saved face_labels.npz successfully!")
