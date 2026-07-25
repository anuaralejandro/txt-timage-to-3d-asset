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

    # Forearms / Lower Arms (RGB=[204, 254, 254])
    if (r > 190 and g > 240 and b > 240) or (abs(x) > 0.35 and y > 0.05):
        labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R

    # Upper Arms / Shoulders (RGB=[232, 251, 55])
    elif (r > 210 and g > 240 and b < 100) or (abs(x) > 0.16 and abs(x) <= 0.35 and y > 0.05):
        labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R

    # Lower Legs / Boots (RGB=[226, 232, 145])
    elif (r > 200 and g > 210 and b in range(120, 170)) or (y < -0.42):
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R

    # Thighs / Upper Legs (RGB=[130, 204, 153] or RGB=[153, 55, 7] in thigh Y range)
    elif (r in range(110, 150) and g > 180 and b in range(130, 170)) or (-0.42 <= y <= -0.02 and abs(x) > 0.01):
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R

    # Head and Hair (RGB=[165, 87, 178] or Y > 0.45)
    elif y > 0.42 or (r in range(140, 180) and g in range(70, 100) and b in range(160, 200)):
        if y > 0.65 or z < -0.05:
            labels[i] = SemanticLabel.HAIR
        else:
            labels[i] = SemanticLabel.HEAD

    # Neck
    elif 0.38 < y <= 0.42 and abs(x) < 0.15:
        labels[i] = SemanticLabel.NECK

    # Torso (RGB=[153, 55, 7] in central body)
    else:
        if y < -0.02:
            labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R
        else:
            labels[i] = SemanticLabel.TORSO

# Topological BFS Hair Graph Propagation for Ponytail
face_adj = mesh.face_adjacency
adj_graph = {idx: [] for idx in range(num_faces)}
for f1, f2 in face_adj:
    adj_graph[f1].append(f2)
    adj_graph[f2].append(f1)

hair_seeds = set(np.where(labels == SemanticLabel.HAIR)[0])
visited = set(hair_seeds)
queue = list(hair_seeds)

while queue:
    curr = queue.pop(0)
    for nbr in adj_graph[curr]:
        if nbr not in visited:
            # Propagate HAIR to all connected hair strands
            if labels[nbr] not in (SemanticLabel.TORSO, SemanticLabel.LEG_UPPER_L, SemanticLabel.LEG_UPPER_R, SemanticLabel.LEG_LOWER_L, SemanticLabel.LEG_LOWER_R):
                visited.add(nbr)
                labels[nbr] = SemanticLabel.HAIR
                queue.append(nbr)

# Postprocessing: smooth boundaries without joint rings
postprocessor = SegmentationPostProcessor(min_island_faces=10, smoothing_iterations=2)
labels = postprocessor.process(improved_path, labels)

print("Exact 13-Class Breakdown:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.name:15s}: {cnt:5d} faces ({cnt/num_faces*100:4.1f}%)")

np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels.npz", face_labels=labels)
print("Saved face_labels.npz successfully!")
