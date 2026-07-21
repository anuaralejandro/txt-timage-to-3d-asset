# Initial State Audit: Hunyuan3D Production Pipeline

Date: 2026-07-21

## 1. Workflows
- **State**: The workflow files (e.g., `04_hunyuan_multiview_unified.json`) are completely empty (0 bytes). There is no source of truth for the ComfyUI node graph.
- **Impact**: The pipeline cannot run natively or be inspected visually in ComfyUI.

## 2. Hunyuan3D Backend (`services/hunyuan3d_2mv/backend.py`)
- **State**: The `generate` method forcibly removes the `right` view if provided, ignoring it completely.
- **State**: Hardcoded inference steps and no exposition of critical parameters like `guidance_scale` or `octree_resolution`.
- **State**: Does not validate the mesh structure before reporting "success" (e.g., it could return a 0-vertex mesh and the pipeline would proceed).

## 3. Configuration & Models (Standard vs Turbo)
- **State**: Handled loosely via a single `variant` string argument in the backend, rather than clearly separated pipelines with distinct guidance and CFG setups.

## 4. Multiview Registration (`canonical_views.py` & `image_alignment.py`)
- **State**: Crops and pads each view independently based on alpha bounds.
- **Impact**: Generates completely mismatched scales and heights across front/left/back views, leading to deformed anatomy in the 3D generation.

## 5. Preflight Runner (`preflight_runner.py`)
- **State**: Only checks for alpha transparency, checkerboard backgrounds, and duplicate files.
- **Missing**: No checks for T-pose, visibility of limbs, or overall anatomical coherence.

## 6. Scoring and Ranking (`mesh_scoring.py` & `ranker.py`)
- **State**: Employs completely simulated placeholder metrics (e.g., hardcoded `silhouette_iou: 0.88`, `passed: len(failed_gates) == 0`).
- **Impact**: Any generated file, regardless of how degenerated it is, will be ranked and approved as a successful 3D character mesh.

## 7. Paint Backend (`services/hunyuan3d_paint/backend.py`)
- **State**: Mocks texture generation by writing a static, tiny valid PNG header string to disk for both base color and normal maps.
- **Impact**: Creates false artifacts and reports fake success instead of explicitly reporting lack of hardware capabilities on 8GB VRAM systems.

## 8. VRAM Scheduler (`vram_scheduler.py`)
- **State**: Relies primarily on static estimates (e.g., `required_mb: 6000`). Uses `torch.cuda.empty_cache()` but does not rigorously force references to be dropped between shape, render, scoring, and paint phases.

## 9. Geometry Post-Processing
- **State**: Lacking a dedicated module for safe topological cleanup, decimation, and GLB verification.

## Conclusion
The current pipeline behaves like a mockup rather than a production system. It simulates successes for downstream processes (scoring, texturing) and does not protect against garbage geometry being produced by the shape generator.
