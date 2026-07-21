# Hunyuan3D-2mv Pipeline Architecture

The pipeline uses Tencent's Hunyuan3D-2mv model to transform 3 or 4 multiview orthogonal images into a 3D geometry mesh.

## Pipeline Stages

1. **InputManifest**: Reads configuration and view definitions.
2. **Preflight**: Validates image transparency, checkerboard presence, and anatomical T-pose bounds.
3. **JointMultiviewAlignment**: Registers all views into a uniform bounding box, ensuring consistent height and centering across views to avoid deformed generations.
4. **HunyuanShapeGeneration**: Dispatches the images to the `Hunyuan2MVBackend`. Enforces `torch.float16` and Flash Attention.
5. **RawMeshValidation**: Validates that the generated mesh contains vertices, faces, and finite numbers (no NaNs or Infs).
6. **MultiViewRenderScoring**: Extracts topological metrics (connected components, degeneracy) and performs silhouette verification against the input views.
7. **CandidateRanking**: If multiple generations (candidates) are requested, ranks them by score and rejects those failing the quality gates.
8. **SafeMeshCleanup**: Removes tiny floating components, unreferenced vertices, and degenerate faces using `trimesh`.
9. **GLBValidation**: Re-opens the generated GLB to ensure the serialization didn't corrupt the file.
10. **OptionalTextureStage**: Offloads to Hunyuan3D-Paint if hardware allows (disabled by default on 8GB systems).

## Supported Formats
- Input: `png` with valid Alpha channel.
- Output: `.glb`

## Execution
Use the unified config: `configs/pipeline/hunyuan3d_8gb.yaml`.
For benchmarking the baseline:
```bash
python -m local_asset_factory.benchmarks.run_hunyuan_baseline \
  --front <path> \
  --back <path> \
  --left <path> \
  --right <path> \
  --mode standard
```
