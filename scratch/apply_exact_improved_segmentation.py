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
centroids = mesh.triangles_center
num_faces = len(mesh.faces)

labels = np.zeros(num_faces, dtype=int)

X = centroids[:, 0]
Y = centroids[:, 1]
Z = centroids[:, 2]

for i in range(num_faces):
    x, y, z = X[i], Y[i], Z[i]

    # 1. Head & Hair (Yellow in user reference)
    if y > 0.58:
        labels[i] = SemanticLabel.HAIR if (y > 0.72 or z < -0.05) else SemanticLabel.HEAD

    # 2. Neck
    elif 0.45 < y <= 0.58 and abs(x) < 0.16:
        labels[i] = SemanticLabel.NECK

    # 3. Arms (T-Pose, abs(X) > 0.16)
    elif y > 0.15 and abs(x) > 0.16:
        if abs(x) > 0.45:
            labels[i] = SemanticLabel.ARM_LOWER_L if x < 0 else SemanticLabel.ARM_LOWER_R
        else:
            labels[i] = SemanticLabel.ARM_UPPER_L if x < 0 else SemanticLabel.ARM_UPPER_R

    # 4. Lower Legs / Boots (Y <= -0.42)
    elif y <= -0.42:
        labels[i] = SemanticLabel.LEG_LOWER_L if x < 0 else SemanticLabel.LEG_LOWER_R

    # 5. Upper Legs / Thighs (Y in [-0.42, 0.05])
    elif -0.42 < y <= 0.05:
        labels[i] = SemanticLabel.LEG_UPPER_L if x < 0 else SemanticLabel.LEG_UPPER_R

    # 6. Torso (Central body Y in [0.05, 0.45], abs(X) <= 0.16)
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
            # Propagate HAIR to all connected hair strands hanging down the back
            if labels[nbr] not in (SemanticLabel.TORSO, SemanticLabel.LEG_UPPER_L, SemanticLabel.LEG_UPPER_R, SemanticLabel.LEG_LOWER_L, SemanticLabel.LEG_LOWER_R):
                visited.add(nbr)
                labels[nbr] = SemanticLabel.HAIR
                queue.append(nbr)

# Smooth boundaries cleanly without joint rings
postprocessor = SegmentationPostProcessor(min_island_faces=10, smoothing_iterations=2)
labels = postprocessor.process(improved_path, labels)

print("Exact 13-Class Breakdown:")
for l in SemanticLabel:
    cnt = (labels == l.value).sum()
    print(f"  {l.name:15s}: {cnt:5d} faces ({cnt/num_faces*100:4.1f}%)")

np.savez_compressed(r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\hunyuan_part_jace0\face_labels.npz", face_labels=labels)
print("Saved face_labels.npz successfully!")
