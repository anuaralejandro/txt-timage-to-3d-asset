# Hunyuan3D 8GB VRAM Limitations

Running the full Hunyuan3D generation pipeline (Shape + Paint) significantly exceeds 8GB VRAM limit on consumer laptops. This document explains the explicit constraints enforced in `configs/pipeline/hunyuan3d_8gb.yaml`.

## Geometry Generation (Hunyuan3D-2mv)
- The model itself uses ~6 GB of VRAM during inference.
- **CPU Offloading** is enforced by default (`enable_cpu_offload=true`).
- **Flash Attention** is required.
- Standard steps (30) or Turbo steps (10) are supported, but must be run sequentially without overlapping models in memory.

## Texture Generation (Hunyuan3D-Paint)
- **Status: Disabled by Default**
- The Paint stage requires loading the Hunyuan3D-2.1 DiT model, which takes ~6-8GB independently. Running it sequentially requires aggressive memory swapping and often leads to out-of-memory errors on 8GB devices because PyTorch cannot always release fragmented VRAM in Windows fast enough.
- For 8GB devices, the pipeline generates a `validated_base_mesh.glb` (untextured or vertex-colored only) and flags the texturing step as `unsupported_on_current_hardware`.

## Candidate Generation
- Instead of batch generation, multiple candidates (if `candidate_count > 1`) are generated sequentially in a loop, cleaning the VRAM explicitly between each pass.

## Fallback Options
To get textured meshes:
1. Use an external cloud instance with 16GB+ VRAM for the `paint` stage.
2. Use simpler heuristics (e.g., Tripo3D API or traditional UV unwrapping).
