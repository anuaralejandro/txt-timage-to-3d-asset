# Changelog: Hunyuan3D Pipeline Fixes

## Refactor Overview
The Hunyuan3D generation pipeline was completely overhauled to resolve issues with empty workflows, fake outputs, arbitrarily discarded views, and lacking safety constraints for geometry generation on 8GB VRAM limits.

## Changes

### 1. Workflows
- Rebuilt `workflows/04_hunyuan_multiview_unified.json` as an alias wrapper for backward compatibility.
- Added native ComfyUI workflow JSON definitions:
  - `hunyuan3d_2mv_standard.json`
  - `hunyuan3d_2mv_turbo.json`
  - `hunyuan3d_shape_only_8gb.json`
- Added syntax validation and node checks via `test_workflows.py`.

### 2. Hunyuan3D Backend
- Refactored `services/hunyuan3d_2mv/backend.py` to:
  - Strictly enforce typed configuration parameters (guidance scale, inference steps).
  - Explicitly retain the `right` view when present and record it in the execution manifest.
  - Check the output type and vertex/face count before writing a file to disk to prevent corrupted meshes.
  - Accurately catch and return VRAM errors.

### 3. Pipeline Separation
- Created distinct pipeline config files for Standard and Turbo models under `configs/pipeline/`.

### 4. Joint Multiview Alignment
- Updated `canonical_views.py` and `image_alignment.py` to calculate joint padding and scaling across all input views, ensuring that scale ratios, height, and center are preserved consistently for `front`, `left`, `back`, and `right` images.

### 5. Preflight Anatomical Validation
- Added T-Pose detection and margin validation in `preflight_runner.py` via alpha channel extremity tracking.
- Blocks execution if extremities are cut off or anatomy is invalid.

### 6. Realistic Metric Scoring & Ranking
- Removed static mock scoring in `mesh_scoring.py`.
- Integrated `trimesh` to extract concrete geometry properties: vertex count, face count, degenerate faces, connected components, bounds.
- Ranker now genuinely checks if minimum topology constraints are met.

### 7. VRAM Scheduler Updates
- Added runtime tracking using `torch.cuda.max_memory_allocated()`.
- Differentiates memory pools dynamically for standard and turbo profiles.

### 8. Post-Processing & Paint Disabled by Default
- Created `src/local_asset_factory/topology/postprocessor.py` for safe decimation and component cleaning.
- `services/hunyuan3d_paint/backend.py` is now explicitly disabled in 8GB mode, safely reporting `unsupported_on_current_hardware` rather than writing fake texture PNGs.

### 9. Benchmarks
- Added `run_hunyuan_baseline.py` A/B benchmark command for offline comparison.
