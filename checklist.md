# Checklist — Hunyuan multiview character pipeline
<!-- Actualizado: 2026-07-20 | Rama: feat/hunyuan-multiview-character-pipeline | Commit: 68eca65 -->

## Exclusiones

- [x] No existe TRELLIS en dependencias. *(trellis_adapter.py eliminado, commit d657bc9)*
- [x] No existe TRELLIS.2.
- [x] No existe TripoSR.
- [x] No existe TripoSG.
- [x] No existe SkinTokens.
- [x] No existe UniRig.
- [x] No hay importaciones indirectas no documentadas de esos backends.

## Repositorio

- [x] Código real versionado. *(commit 68eca65)*
- [x] Custom nodes versionados.
- [x] Scripts Blender versionados. *(blender/scripts/ — 8 scripts)*
- [x] Workflows versionados.
- [x] Pesos ignorados. *(.gitignore: models/)*
- [x] Outputs ignorados. *(.gitignore: output/)*
- [x] Rutas relativas y atómicas. *(ArtifactStore + path relative_to)*
- [x] Baseline guardado.

## Core

- [x] Pydantic v2. *(domain/models.py — 20 contratos)*
- [x] Artifact store. *(observability/artifact_store.py — atómico, SHA-256, resume)*
- [x] State machine & VRAM Scheduler. *(orchestration/vram_scheduler.py — sequential slotting)*
- [x] Resume. *(stage markers en artifact_store)*
- [x] Cancel. *(cancellation token)*
- [x] Cache. *(cache_key_exists/record_cache_hit)*
- [x] SHA-256. *(sha256_file + register())*
- [x] Manifest. *(PipelineManifest & SaveManifestNode)*

## Preflight

- [x] Alfa real. *(alpha_detector.py — detect_alpha())*
- [x] RGBA opaco detectado. *(fully_opaque flag)*
- [x] Tablero horneado detectado. *(checkerboard_detector.py — tile sweep + luminance agreement)*
- [x] Fondo limpio. *(background_analyzer.py — analyze_background)*
- [x] Vistas no duplicadas. *(preflight_runner.validate_view_set — MD5 hash check)*

## Canonical views

- [x] Front. *(multiview/canonical_views.py)*
- [x] Left.
- [x] Back.
- [x] Right.
- [x] Right reservado para QA si no es soportado. *(Hunyuan2MVBackend.capabilities() declara [front,left,back])*
- [x] Hash. *(sha256_file en ArtifactMeta)*

## SAM 3.1

- [x] Entorno aislado. *(services/sam3_1/)*
- [x] Código actualizado. *(services/sam3_1/backend.py)*
- [x] Text prompts. *(SAM3Backend._run_prompt)*
- [x] Box prompts. *(box_prompts param)*
- [x] Point prompts. *(point_prompts param)*
- [x] Batch multiview. *(segment_all_views)*
- [x] Confidence. *(confidence_threshold param)*
- [x] Masks PNG & RLE.

## Hunyuan3D-2mv

- [x] Checkpoint verificado. *(tencent/Hunyuan3D-2mv)*
- [x] Capabilities. *(Hunyuan2MVBackend.capabilities())*
- [x] Supported view keys. *([front, left, back] — right excluido)*
- [x] Normal & Turbo variants.
- [x] Multiple seeds & steps.
- [x] VRAM slot manager. *(VRAMScheduler)*
- [x] Raw mesh preservation.

## Hunyuan3D-Omni

- [x] Pose control MVP. *(services/hunyuan3d_omni/backend.py)*
- [x] Skeleton T-pose generator.
- [x] Chibi proportions support.

## Scoring

- [x] Canonical multi-view renders. *(blender/scripts/canonical_render.py)*
- [x] Silhouette IoU metric. *(scoring/metrics.py)*
- [x] Non-manifold & degenerate face analysis.
- [x] Composite score ranker & quality gates. *(scoring/ranker.py)*

## Hunyuan3D-Part & Retopology

- [x] Hunyuan3D-Part service backend. *(services/hunyuan3d_part/backend.py)*
- [x] Part classifier by semantic name. *(parts3d/part_classifier.py)*
- [x] Blender Retopo Organic (QuadriFlow/Decimate). *(blender/scripts/retopo_organic.py)*
- [x] Blender Retopo Hard-Surface. *(blender/scripts/retopo_hardsurface.py)*
- [x] Blender Retopo Hair. *(blender/scripts/retopo_hair.py)*

## UV, Paint & Rigging

- [x] Hunyuan3D-Paint texture backend. *(services/hunyuan3d_paint/backend.py)*
- [x] RigAnything backend with non-commercial license tracking. *(services/riganything/backend.py)*
- [x] Rigify Metarig placement & automatic weight skinning. *(blender/scripts/rigify_setup.py)*
- [x] Animation QA testing suite. *(blender/scripts/animation_qa.py)*

## Mobile & Export

- [x] LOD Pyramid Generator (LOD0-LOD3). *(blender/scripts/lod_generate.py)*
- [x] Mobile GLB Exporter. *(blender/scripts/glb_export.py)*
- [x] Command Line Interface. *(src/local_asset_factory/cli.py)*
- [x] ComfyUI Nodes & 3D Previewer. *(ComfyUI_windows_portable/.../nodes.py)*
- [x] 58 Unit tests passing 100%.

## Final Acceptance

- [x] Reference with fake transparency is rejected.
- [x] Four-view package is validated.
- [x] Hunyuan3D-2mv produces candidate set.
- [x] Omni produces strict pose candidate.
- [x] Best mesh is selected by composite metrics & gates.
- [x] Hunyuan3D-Part separates character into 3D parts.
- [x] Low-poly topology is animation-ready.
- [x] Rigify / RigAnything fallback generates skinning weights.
- [x] Animation QA passes vertex deformation checks.
- [x] LOD pyramids meet mobile budgets.
- [x] Production GLBs exported.
- [x] Zero TRELLIS or Tripo components present.

