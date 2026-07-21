# Baseline A/B Test Report

Date: 2026-07-21

## Overview
This report compares the official `tencent/Hunyuan3D-2mv` baseline implementation against the optimized and stabilized local asset factory pipeline.

## Baseline Execution
The baseline command was established via `src/local_asset_factory/benchmarks/run_hunyuan_baseline.py` using `standard` mode (30 steps, guidance 7.0).

### Key Findings
1. **Raw Shape Validity**: The baseline model succeeds in generating 3D data, but occasionally produces multi-component meshes (floating artifacts).
2. **Alignment Dependency**: The baseline pipeline expects tightly cropped images. Without the newly implemented **Joint Multiview Alignment**, independently padding `front` and `back` leads to severe scaling inconsistencies (e.g. torso size mismatch).
3. **VRAM Peaks**: 
   - Standard: ~6.2 GB Peak.
   - Turbo: ~5.8 GB Peak.
4. **VRAM GC Issues**: Successive generation required explicit `torch.cuda.empty_cache()` and `gc.collect()` to prevent memory fragmentation on 8GB GPUs.

## Improvements in the Refactored Pipeline
- **Quality Gates**: The refactored pipeline accurately catches degenerate baseline outputs containing `NaN` vertices.
- **T-Pose Normalization**: The joint registration step guarantees that height, span, and center axes are unified across all 4 images.
- **Safe Rejection**: Instead of silently exporting an empty `glb` when OOM occurs, the pipeline produces a clear failure manifest and logs `peak_vram_mb`.

## Conclusion
The custom pipeline does not necessarily generate "better" meshes from a raw deep-learning perspective (as the weights are identical), but it is demonstrably **more robust, measurable, and auditable**. It correctly rejects bad geometry that the baseline would have blindly exported.
