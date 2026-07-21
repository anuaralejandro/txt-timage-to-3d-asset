# ComfyUI-LocalAssetFactory

> **Local-first game asset generation pipeline for ComfyUI.**
>
> Transform text descriptions into game-ready 3D assets using only local backends:
> Ollama · ComfyUI · TRELLIS · Blender

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Nodes Reference](#nodes-reference)
- [Workflows](#workflows)
- [Texture Generation](#texture-generation)
- [Verification](#verification)
- [Resuming from Failed Stages](#resuming-from-failed-stages)
- [Disabling TRELLIS](#disabling-trellis)
- [VRAM Management](#vram-management)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Overview

**LocalAssetFactory** is a ComfyUI custom node package that creates game assets for mobile/Unity from a text description. The entire pipeline runs locally — no cloud APIs required.

### Pipeline Stages

```
Asset Brief → Ollama Prompt Architect → Concept Image → Textures → TRELLIS 3D → Blender → Manifest
```

Each stage is independently skippable. The system supports partial success — if TRELLIS fails due to VRAM, the pipeline continues with what it has.

### Pipeline Modes

| Mode | Stages |
|------|--------|
| `spec_only` | Brief → Spec |
| `concept` | Brief → Spec → Concept Image |
| `concept_textures` | Brief → Spec → Concept Image → Textures |
| `full` | Brief → Spec → Concept → Textures → TRELLIS |
| `full_blender` | All stages including Blender processing |

---

## Architecture

```
ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/
├── __init__.py              # ComfyUI node registration
├── nodes.py                 # 7 custom nodes
├── config.py                # Configuration (env vars + .env)
├── schemas.py               # Pydantic data models
├── requirements.txt
├── .env.example
├── services/                # Backend integrations
│   ├── ollama_client.py     # Local Ollama HTTP client
│   ├── comfy_bridge.py      # ComfyUI API wrapper
│   ├── trellis_adapter.py   # TRELLIS image-to-3D
│   ├── texture_generator.py # Texture generation logic
│   ├── blender_runner.py    # Headless Blender execution
│   ├── manifest_writer.py   # Manifest JSON creation
│   └── output_manager.py    # Output directory management
├── utilities/               # Shared helpers
│   ├── environment.py       # Environment checks
│   ├── image_conversion.py  # PIL ↔ ComfyUI tensor
│   ├── file_safety.py       # Path validation & sanitisation
│   ├── retry.py             # Retry decorator
│   └── logging_utils.py     # Centralised logging
├── blender/                 # Blender headless scripts
│   ├── process_asset.py     # Full processing pipeline
│   ├── assign_texture.py    # Texture assignment
│   ├── generate_preview_renders.py
│   └── optimize_mesh.py     # Mesh optimization + decimation
├── workflows/               # ComfyUI workflow templates
│   ├── local_asset_factory_v1.json
│   ├── local_asset_factory_v1_api.json
│   ├── concept_image_v1.json
│   ├── texture_generation_v1.json
│   └── trellis_local_v1.json
├── scripts/                 # Verification scripts
│   ├── verify_environment.py
│   ├── verify_ollama.py
│   ├── verify_comfy_models.py
│   └── verify_blender.py
└── tests/                   # Unit tests
    ├── test_environment.py
    ├── test_schemas.py
    ├── test_output_paths.py
    ├── test_manifest.py
    └── test_file_safety.py
```

---

## Requirements

### Hardware

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX (8 GB VRAM) |
| RAM | 16 GB |
| Storage | 20 GB free (for models) |

### Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| ComfyUI | Latest | Image generation |
| Ollama | Latest | Local LLM |
| Blender | 3.6+ / 4.x | Mesh processing |

### Optional

| Software | Purpose |
|----------|---------|
| ComfyUI-IF_Trellis | Image-to-3D generation |
| SDXL / SD 1.5 / FLUX | Checkpoint for image generation |

---

## Installation

### Step 1: Clone into ComfyUI custom_nodes

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/YOUR_USER/ComfyUI-LocalAssetFactory.git
```

Or copy the folder directly.

### Step 2: Install Python dependencies

```powershell
cd ComfyUI\custom_nodes\ComfyUI-LocalAssetFactory
pip install -r requirements.txt
```

### Step 3: Install Ollama

Download from [ollama.com](https://ollama.com) and pull a model:

```powershell
ollama pull mistral
```

Recommended models for 8 GB VRAM:
- `mistral` (7B) — good balance of quality and speed
- `llama3.1:8b` — strong instruction following
- `gemma2:9b` — excellent structured output

### Step 4: Verify

```powershell
# Start Ollama (if not running as a service)
ollama serve

# Verify everything
python scripts\verify_environment.py
python scripts\verify_ollama.py
python scripts\verify_comfy_models.py
python scripts\verify_blender.py
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and edit:

```powershell
copy .env.example .env
notepad .env
```

**System environment variables take precedence over `.env` values.**

### Setting variables on Windows

```powershell
# Temporary (current session)
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:BLENDER_EXECUTABLE = "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

# Permanent (user)
[System.Environment]::SetEnvironmentVariable("BLENDER_EXECUTABLE", "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe", "User")
```

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI server URL |
| `BLENDER_EXECUTABLE` | `blender` | Path to Blender executable |
| `ASSET_FACTORY_OUTPUT_DIR` | `ComfyUI/output/local_asset_factory` | Output directory |
| `ENABLE_TRELLIS` | `true` | Enable/disable TRELLIS |
| `ENABLE_TEXTURE_GENERATION` | `true` | Enable/disable textures |
| `ENABLE_BLENDER_PROCESSING` | `true` | Enable/disable Blender |
| `ALLOW_PARTIAL_SUCCESS` | `true` | Continue on non-fatal errors |

---

## Quick Start

### 1. Start services

```powershell
# Terminal 1: Ollama
ollama serve

# Terminal 2: ComfyUI
cd C:\path\to\ComfyUI
python main.py
```

### 2. Load the workflow

Open ComfyUI in your browser (`http://localhost:8188`) and load:

```
ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/workflows/local_asset_factory_v1.json
```

### 3. Configure the Asset Brief node

- Set the asset name, type, description, style, etc.
- The example workflow comes pre-configured with a **Royal Training Sword**.

### 4. Queue the prompt

Click **Queue Prompt**. The pipeline will:
1. Generate the structured spec via Ollama
2. Create concept art via ComfyUI
3. Generate textures via ComfyUI
4. Attempt TRELLIS 3D generation (if enabled)
5. Process in Blender (if model was generated)
6. Write `manifest.json`

### 5. Check output

```
ComfyUI/output/local_asset_factory/<asset_id>/
├── concept/           ← concept art
├── textures/source/   ← generated textures
├── raw_3d/            ← TRELLIS output
├── processed/         ← Blender output (.glb)
├── previews/          ← preview renders
└── manifest.json      ← complete metadata
```

---

## Nodes Reference

### 1. Asset Factory · Asset Brief

Creates a validated asset request from user inputs.

**Inputs:** asset_name, asset_type, description_es, visual_style, gameplay_role, target_platform, triangle_budget, texture_resolution, seed

**Outputs:** ASSET_REQUEST_JSON, STATUS

### 2. Asset Factory · Ollama Prompt Architect

Calls local Ollama to produce structured specification and optimized prompts.

**Inputs:** ASSET_REQUEST_JSON
**Controls:** model, temperature, max_tokens, generate_texture_prompts, generate_variants
**Outputs:** ASSET_SPEC_JSON, CONCEPT_IMAGE_PROMPT, CONCEPT_NEGATIVE_PROMPT, TEXTURE_PROMPT_JSON, STATUS

### 3. Asset Factory · Local Concept Image

Generates concept art via a local ComfyUI workflow.

**Inputs:** CONCEPT_IMAGE_PROMPT, CONCEPT_NEGATIVE_PROMPT
**Controls:** workflow_path, width, height, steps, cfg, seed, enabled
**Outputs:** IMAGE, IMAGE_PATH, STATUS

### 4. Asset Factory · Local Texture Generator

Generates textures locally from structured prompts.

**Inputs:** TEXTURE_PROMPT_JSON, ASSET_SPEC_JSON
**Controls:** workflow_path, generate_base_color, generate_ornament_variant, tileable_mode, width, height, steps, cfg, seed, enabled
**Outputs:** TEXTURE_PATHS_JSON, PREVIEW_TEXTURE_IMAGE, STATUS

### 5. Asset Factory · Local TRELLIS Image-to-3D

Runs TRELLIS for image-to-3D conversion (optional, fault-tolerant).

**Inputs:** IMAGE, IMAGE_PATH, ASSET_SPEC_JSON
**Controls:** workflow_path, generate_3d, quality_mode, seed, timeout_seconds
**Outputs:** MODEL_PATH, ALL_OUTPUT_PATHS_JSON, STATUS

### 6. Asset Factory · Blender Processor

Processes the 3D model in headless Blender.

**Inputs:** MODEL_PATH, TEXTURE_PATHS_JSON, ASSET_SPEC_JSON
**Controls:** blender_executable, optimize_mesh, assign_texture, generate_previews, create_glb, create_fbx, enabled
**Outputs:** PROCESSED_MODEL_PATH, PREVIEW_RENDER_PATHS_JSON, BLENDER_REPORT_JSON, STATUS

### 7. Asset Factory · Save Manifest

Writes the complete manifest.json with all metadata.

**Inputs:** All outputs from previous nodes
**Outputs:** MANIFEST_PATH, MANIFEST_JSON, STATUS

---

## Workflows

| Workflow | Purpose |
|----------|---------|
| `local_asset_factory_v1.json` | Full pipeline with all 7 nodes |
| `local_asset_factory_v1_api.json` | API-format for programmatic use |
| `concept_image_v1.json` | Standalone concept art generation |
| `texture_generation_v1.json` | Standalone texture generation |
| `trellis_local_v1.json` | Standalone TRELLIS image-to-3D |

### Customizing workflows

The concept and texture workflows use standard ComfyUI nodes (CheckpointLoaderSimple, CLIPTextEncode, KSampler, etc.). You can:

1. Replace the checkpoint with your preferred model
2. Add LoRA nodes for style control
3. Adjust sampler settings for quality vs. speed
4. Change the resolution for your VRAM budget

---

## Texture Generation

### Surface Textures (tileable)

For walls, floors, materials. Enable `tileable_mode`:
- Prompts include "seamless tileable" prefix
- Best for modular_piece and architecture assets

### Object Textures (unique)

For hero assets like swords, shields, chests:
- Clean albedo without tiling constraints
- Focused on readability at mobile resolutions

### Prompt Construction

The system automatically adds game-texture-appropriate prefixes:

```
Surface: "flat albedo game texture, seamless tileable, stylized hand-painted, ..."
Object:  "flat albedo game texture, stylized hand-painted, anime low poly style, ..."
```

And a standard negative prompt avoiding text, watermarks, lighting artifacts, etc.

---

## Verification

Run all verification scripts before first use:

```powershell
cd ComfyUI\custom_nodes\ComfyUI-LocalAssetFactory

# Full environment check
python scripts\verify_environment.py

# Ollama connectivity + model check
python scripts\verify_ollama.py

# ComfyUI connectivity + node check
python scripts\verify_comfy_models.py

# Blender executable + headless mode
python scripts\verify_blender.py
```

---

## Resuming from Failed Stages

Since outputs are saved to disk at each stage, you can resume manually:

1. **If Ollama failed:** Fix Ollama, re-run only the Ollama Prompt Architect node.
2. **If concept image failed:** The spec JSON is already saved. Disconnect the Ollama node and feed the spec directly.
3. **If TRELLIS failed:** If `ALLOW_PARTIAL_SUCCESS=true`, the manifest is still created. You can re-run TRELLIS later with the saved concept image.
4. **If Blender failed:** The raw 3D model is saved. You can re-run Blender processing independently.

---

## Disabling TRELLIS

### Method 1: Environment variable

```powershell
$env:ENABLE_TRELLIS = "false"
```

### Method 2: Node control

In the TRELLIS node, set `generate_3d` to `false`.

### Method 3: Disconnect

Simply disconnect the TRELLIS node from the workflow. The Blender node will skip when it receives an empty MODEL_PATH.

---

## VRAM Management

### Designed for 8 GB VRAM

The pipeline runs stages sequentially. Each stage uses GPU independently:

1. **Ollama** — LLM inference (uses its own VRAM management)
2. **Concept Image** — Loads checkpoint, generates, ComfyUI unloads
3. **Textures** — Same checkpoint, reused if possible
4. **TRELLIS** — Separate model, needs its own VRAM
5. **Blender** — CPU-based, no GPU required

### Tips for low VRAM

- Use **SD 1.5** instead of SDXL (less VRAM)
- Set concept image to **512×512** instead of 768×768
- Set TRELLIS to **fast** quality mode
- Disable TRELLIS if you don't need 3D
- Close other GPU applications during generation
- Use `--lowvram` or `--novram` flags when starting ComfyUI

### If you get CUDA out-of-memory

1. The system catches VRAM errors and reports them clearly
2. If `ALLOW_PARTIAL_SUCCESS=true`, the pipeline continues
3. Check the manifest's `errors` field for details
4. Consider disabling TRELLIS or reducing resolution

---

## Troubleshooting

### "Ollama is not available"

```powershell
# Check if Ollama is running
ollama list
# If not running:
ollama serve
```

### "ComfyUI not available"

Make sure ComfyUI is running on the configured port (default: 8188).

### "Blender NOT found"

Set the full path:

```powershell
$env:BLENDER_EXECUTABLE = "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
```

### "No image generated"

- Check that a checkpoint model is loaded in ComfyUI
- Verify the workflow file matches your ComfyUI setup
- Check ComfyUI's console for errors

### "Invalid JSON from Ollama"

The system automatically:
1. Tries to repair the JSON locally
2. Makes a second call asking for corrected JSON
3. Fails cleanly with an actionable error if both attempts fail

Try a different model or lower the temperature.

---

## Limitations

### v1 Limitations

1. **No PBR textures** — v1 generates base color/albedo only, not normal maps, roughness, etc.
2. **TRELLIS dependency** — Requires ComfyUI-IF_Trellis to be installed and working; the workflow template may need adjustment for your specific version
3. **Single-object focus** — The pipeline is designed for individual assets, not scenes
4. **No UV unwrapping** — Blender processing does basic cleanup but does not create custom UV maps
5. **No automatic triangle decimation** — The pipeline reports triangle counts but does not enforce budgets automatically in v1
6. **Workflow templates are generic** — You may need to adjust checkpoint names and node configurations for your specific ComfyUI setup
7. **No Unity importer yet** — Output files (.glb, .fbx) are Unity-compatible but there's no automatic import script

### Technical Notes

- The ComfyUI bridge uses polling (not WebSocket) for simplicity
- Blender scripts run as subprocesses, not as a persistent server
- TRELLIS quality depends heavily on the concept image quality
- LLM output quality varies by model; `mistral` and `llama3.1:8b` are recommended

---

## Running Tests

```powershell
cd ComfyUI\custom_nodes\ComfyUI-LocalAssetFactory
python -m pytest tests/ -v
```

Or individual tests:

```powershell
python -m pytest tests/test_schemas.py -v
python -m pytest tests/test_file_safety.py -v
python -m pytest tests/test_manifest.py -v
```

---

## License

MIT

---

## Credits

Built with:
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Ollama](https://ollama.com)
- [Blender](https://www.blender.org)
- [Pydantic](https://docs.pydantic.dev)
