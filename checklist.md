# Checklist — Hunyuan multiview character pipeline

## Exclusiones

- [ ] No existe TRELLIS en dependencias.
- [ ] No existe TRELLIS.2.
- [ ] No existe TripoSR.
- [ ] No existe TripoSG.
- [ ] No existe SkinTokens.
- [ ] No existe UniRig.
- [ ] No hay importaciones indirectas no documentadas de esos backends.

## Repositorio

- [ ] Código real versionado.
- [ ] Custom nodes versionados.
- [ ] Scripts Blender versionados.
- [ ] Workflows versionados.
- [ ] Pesos ignorados.
- [ ] Outputs ignorados.
- [ ] Rutas absolutas eliminadas.
- [ ] Baseline guardado.

## Core

- [ ] Pydantic v2.
- [ ] Artifact store.
- [ ] State machine.
- [ ] Resume.
- [ ] Cancel.
- [ ] Cache.
- [ ] SHA-256.
- [ ] JSON logs.
- [ ] Manifest.

## Preflight

- [ ] Alfa real.
- [ ] RGBA opaco detectado.
- [ ] Tablero horneado detectado.
- [ ] Fondo limpio.
- [ ] Crop.
- [ ] Padding.
- [ ] Cuerpo completo.
- [ ] Brazos visibles.
- [ ] Piernas visibles.
- [ ] T-pose.
- [ ] Vistas no duplicadas.
- [ ] Identidad coherente.
- [ ] Vestuario coherente.
- [ ] Proporciones coherentes.

## Canonical views

- [ ] Front.
- [ ] Left.
- [ ] Back.
- [ ] Right.
- [ ] Right reservado para QA si no es soportado.
- [ ] Cámara registrada.
- [ ] Escala registrada.
- [ ] Keypoints.
- [ ] Bbox.
- [ ] Hash.

## SAM 3.1

- [ ] Entorno aislado.
- [ ] Código actualizado.
- [ ] Checkpoint 3.1.
- [ ] Autenticación HF.
- [ ] Text prompts.
- [ ] Box prompts.
- [ ] Point prompts.
- [ ] Batch multiview.
- [ ] Confidence.
- [ ] Masks PNG.
- [ ] Masks RLE.
- [ ] Consistency gate.
- [ ] Body.
- [ ] Hair.
- [ ] Top.
- [ ] Shorts.
- [ ] Belt.
- [ ] Skirt.
- [ ] Gloves.
- [ ] Boots.
- [ ] Accessories.

## Hunyuan3D-2mv

- [ ] Entorno aislado.
- [ ] Checkpoint verificado.
- [ ] Revision registrada.
- [ ] Capabilities.
- [ ] Supported view keys.
- [ ] Normal variant.
- [ ] Turbo variant.
- [ ] Multiple seeds.
- [ ] Multiple steps.
- [ ] Timeout.
- [ ] Cancel.
- [ ] OOM report.
- [ ] VRAM report.
- [ ] Raw mesh preservation.

## Hunyuan3D-Omni

- [ ] Entorno aislado.
- [ ] Pose control.
- [ ] Skeleton T-pose.
- [ ] Chibi proportions.
- [ ] Bbox candidate.
- [ ] Voxel candidate.
- [ ] Point candidate.
- [ ] No unsupported multi-control assumption.
- [ ] Candidate metadata.

## Scoring

- [ ] Front render.
- [ ] Left render.
- [ ] Back render.
- [ ] Right render.
- [ ] Three-quarter renders.
- [ ] Top render.
- [ ] Turntable.
- [ ] Silhouette IoU.
- [ ] Semantic IoU.
- [ ] Keypoint error.
- [ ] Pose error.
- [ ] Head/body ratio.
- [ ] Arm separation.
- [ ] Leg separation.
- [ ] Symmetry.
- [ ] Non-manifold.
- [ ] Degenerates.
- [ ] Components.
- [ ] Normals.
- [ ] Surface noise.
- [ ] Ranking.
- [ ] Failure when no candidate passes.

## Hunyuan3D-Part

- [ ] P3-SAM.
- [ ] X-Part.
- [ ] Semantic names.
- [ ] Confidence.
- [ ] Face IDs.
- [ ] Bboxes.
- [ ] Symmetry partners.
- [ ] SAM projection fallback.
- [ ] Manual override.
- [ ] Body separated.
- [ ] Hair separated.
- [ ] Ponytail separated.
- [ ] Gloves separated.
- [ ] Boots separated.
- [ ] Clothing separated.
- [ ] Accessories separated.

## Retopology

- [ ] Part classifier.
- [ ] Repair common.
- [ ] Organic path.
- [ ] Cloth path.
- [ ] Hair path.
- [ ] Hard-surface path.
- [ ] Accessory path.
- [ ] Humanoid template.
- [ ] Chibi 4-head template.
- [ ] Chibi 5-head template.
- [ ] AutoRemesher.
- [ ] QuadriFlow fallback.
- [ ] OpenVDB.
- [ ] Shoulder loops.
- [ ] Elbow loops.
- [ ] Wrist loops.
- [ ] Hip loops.
- [ ] Groin loops.
- [ ] Knee loops.
- [ ] Ankle loops.
- [ ] Shrinkwrap.
- [ ] Projection error.
- [ ] Deformation test.
- [ ] Recomposition.

## UV and Paint

- [ ] Topology finalized first.
- [ ] UV unwrap.
- [ ] Atlas.
- [ ] Padding.
- [ ] Texel density.
- [ ] High-to-low bake.
- [ ] Base color.
- [ ] Normal.
- [ ] AO.
- [ ] Roughness.
- [ ] Metallic.
- [ ] Seam QA.
- [ ] Hunyuan Paint after UV.
- [ ] Toon mobile mode.
- [ ] PBR mobile mode.
- [ ] Outline mask.
- [ ] Max two materials.

## Rigging

- [ ] RigAnything environment.
- [ ] Noncommercial notice.
- [ ] RigAnything candidate.
- [ ] Rigify fallback.
- [ ] Metarig placement.
- [ ] Automatic weights.
- [ ] Weight normalization.
- [ ] Four influence limit.
- [ ] No unweighted vertices.
- [ ] Ponytail bones.
- [ ] Skirt bones.
- [ ] Project skeleton.
- [ ] Bind pose.
- [ ] Retarget.

## Animation QA

- [ ] Idle.
- [ ] Walk.
- [ ] Run.
- [ ] Jump.
- [ ] Crouch.
- [ ] Arms up.
- [ ] Arms forward.
- [ ] Elbow bend.
- [ ] Knee bend.
- [ ] Torso twist.
- [ ] Penetration.
- [ ] Volume loss.
- [ ] UV stretch.
- [ ] Normal flips.
- [ ] Detached parts.
- [ ] Export blocked on severe failure.

## Mobile

- [ ] Android low.
- [ ] Android mid.
- [ ] Android high.
- [ ] iOS profiles.
- [ ] LOD0.
- [ ] LOD1.
- [ ] LOD2.
- [ ] LOD3.
- [ ] Skinning preserved.
- [ ] Max materials.
- [ ] Max bones.
- [ ] Max influences.
- [ ] KTX2.
- [ ] Draw-call report.
- [ ] Device benchmark.

## Export

- [ ] GLB LOD0.
- [ ] GLB LOD1.
- [ ] GLB LOD2.
- [ ] GLB LOD3.
- [ ] glTF Validator.
- [ ] Blender re-import.
- [ ] Engine smoke test.
- [ ] Manifest.
- [ ] QC report.
- [ ] Previews.
- [ ] Turntable.
- [ ] Scale correct.
- [ ] Axes correct.
- [ ] Names correct.
- [ ] Materials correct.
- [ ] Rig correct.
- [ ] Clips correct.

## ComfyUI

- [ ] Thin nodes only.
- [ ] No heavy model imported into ComfyUI.
- [ ] Request.
- [ ] Validate views.
- [ ] SAM.
- [ ] 2mv.
- [ ] Omni.
- [ ] Rank.
- [ ] Parts.
- [ ] Retopo.
- [ ] UV/bake.
- [ ] Paint.
- [ ] Rig.
- [ ] QA.
- [ ] Optimize.
- [ ] Export.
- [ ] Monitor.
- [ ] Preview.

## Text-to-asset later

- [ ] HunyuanImage isolated.
- [ ] Front generation.
- [ ] Side derivation.
- [ ] Back derivation.
- [ ] Identity lock.
- [ ] SAM masks.
- [ ] Consistency gate.
- [ ] Automatic regeneration.
- [ ] Hardware documented.

## Final acceptance

- [ ] Reference with fake transparency is rejected.
- [ ] Four-view package is validated.
- [ ] Hunyuan3D-2mv produces candidate set.
- [ ] Omni produces strict pose candidate.
- [ ] Best mesh is selected by metrics.
- [ ] Hunyuan3D-Part separates the character.
- [ ] Low-poly topology is animation-ready.
- [ ] Texture follows final UV.
- [ ] Rig has fallback.
- [ ] Animation QA passes.
- [ ] LODs meet mobile budgets.
- [ ] GLB is valid.
- [ ] No TRELLIS or Tripo component is present.
