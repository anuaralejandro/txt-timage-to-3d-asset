# 3D Anatomical Segmentation & Multiview Texture Projection Pipeline

## Overview

This module extends the Hunyuan3D-2mv geometry generation pipeline (`04_hunyuan_multiview_unified.json`) with an 8-class 3D anatomical segmentation and multiview texture projection stage.

Designed specifically for 8 GB VRAM GPUs (NVIDIA RTX 4070 Laptop), the pipeline executes model inferencing sequentially with explicit VRAM unloading (`torch.cuda.empty_cache()`, `gc.collect()`, CPU offloading).

---

## 8 Semantic Anatomical Classes

| Class ID | Name | Description |
|---|---|---|
| `0` | `unassigned` | Background or unclassified face |
| `1` | `torso` | Chest, abdomen, back, pelvis, clothing |
| `2` | `neck` | Neck region derived between jawline and shoulder line |
| `3` | `head` | Head and face excluding hair |
| `4` | `hair` | Hair geometry |
| `5` | `arm_L` | Character's anatomical left arm and hand |
| `6` | `arm_R` | Character's anatomical right arm and hand |
| `7` | `leg_L` | Character's anatomical left leg and foot |
| `8` | `leg_R` | Character's anatomical right leg and foot |

> **Note on Lateralities**: `arm_L`/`arm_R` and `leg_L`/`leg_R` always represent the **character's anatomical left/right**, regardless of camera view or screen position.

---

## Output Artifacts

Running the segmentation pipeline produces the following files:

1. `character_segmented.glb`: Main segmented 3D model containing metadata extras (`face_labels_file`) and per-class debug materials.
2. `face_labels.npz`: Compressed NumPy file containing the 1D array of class labels (0..8) for all mesh faces.
3. `semantic_parts.json`: Schema 1.0 JSON manifest with face counts, confidence scores, models used, and warnings.
4. `character_segmented_colored.glb`: GLB preview model with vertex color palette mapping.
5. `character_textured.glb`: Segmented GLB model with 1024x1024 UV albedo texture baked without cross-class texture bleeding.
6. `albedo_1024.png`: Baked UV albedo texture map with 12px UV edge dilation.

---

## ComfyUI Workflow 05

The new workflow:
```text
ComfyUI/ComfyUI-LocalAssetFactory/workflows/05_hunyuan_multiview_segment_and_texture_8gb.json
```
connects directly to the output of `AssetFactory_BlenderProcessor` from workflow `04`.
