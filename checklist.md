# Checklist maestro — Pipeline Hunyuan3D multivista

---

## A. Repositorio y reproducibilidad

- [x] `ComfyUI_windows_portable/ComfyUI` no oculta código crítico como gitlink.
- [x] `04_hunyuan_multiview_unified.json` está versionado.
- [x] Los custom nodes están versionados.
- [x] Los scripts Blender están versionados.
- [x] Los pesos están ignorados.
- [x] Los outputs están ignorados.
- [x] No hay rutas absolutas personales.
- [x] Existe script de instalación.
- [x] El script soporta `--dry-run`.
- [x] Un clon limpio puede instalarse.
- [x] Existe `docs/current_state.md`.

---

## B. Baseline oficial

- [x] Checkpoint normal descargado.
- [x] Checkpoint Turbo descargado.
- [x] SHA-256 del normal registrado.
- [x] SHA-256 del Turbo registrado.
- [x] VAE registrado.
- [x] Image encoder registrado.
- [x] Workflow oficial normal guardado.
- [x] Workflow oficial Turbo guardado.
- [x] Input oficial guardado/documentado.
- [x] GLB raw normal válido.
- [x] GLB raw Turbo válido.
- [x] Entorno registrado.
- [x] Reporte baseline creado.
- [x] La baseline pasa antes de usar imágenes propias.

---

## C. Auditoría del workflow

- [x] Loader de checkpoint identificado.
- [x] Variante normal/Turbo detectada.
- [x] VAE identificado.
- [x] Image encoder identificado.
- [x] Nodo de conditioning identificado.
- [x] Orden de vistas documentado.
- [x] Preprocessing documentado.
- [x] Resoluciones documentadas.
- [x] Sampler documentado.
- [x] Scheduler documentado.
- [x] Steps documentados.
- [x] CFG documentado.
- [x] Guidance documentado.
- [x] Negative conditioning documentado.
- [x] Latent resolution documentada.
- [x] Octree documentado.
- [x] Chunks documentados.
- [x] Mesh algorithm documentado.
- [x] Threshold documentado.
- [x] Remove-small-components documentado.
- [x] Decimation documentada.
- [x] Export documentado.
- [x] `docs/workflow_audit.md` creado.

---

## D. Normal vs Turbo

- [x] Config normal separada.
- [x] Config Turbo separada.
- [x] Turbo no hereda CFG del normal.
- [x] Turbo no hereda guidance incompatible.
- [x] Normal no usa parámetros distilled.
- [x] Cada manifest registra variante.
- [x] Tests golden separados.
- [x] Error explícito ante combinación inválida.

---

## E. Input y transparencia

- [x] Se detecta RGB sin alpha.
- [x] Se detecta RGBA totalmente opaco.
- [x] Se detecta alpha útil.
- [x] Se detecta checkerboard horneado.
- [x] Se limpia RGB oculto bajo alpha 0.
- [x] No hay halos tras resize.
- [x] Bbox válido.
- [x] Cuerpo completo.
- [x] Brazos visibles.
- [x] Manos visibles.
- [x] Piernas visibles.
- [x] Pies visibles.
- [x] Fondo neutral/composite disponible para debug.
- [x] Preview de máscara guardada.

---

## F. Canonical views

- [x] Front canónica.
- [x] Left canónica.
- [x] Back canónica.
- [x] Right canónica.
- [x] Landmarks detectados.
- [x] Ojos alineados.
- [x] Hombros alineados.
- [x] Pelvis alineada.
- [x] Rodillas alineadas.
- [x] Pies en línea basal común.
- [x] Escala global común.
- [x] Padding común.
- [x] Centro anatómico común.
- [x] T-pose validada.
- [x] Perspectiva evaluada.
- [x] Vistas reflejadas detectadas.
- [x] Left/right verificadas.
- [x] Overlay de registro guardado.
- [x] Transformaciones guardadas.
- [x] Hash por vista registrado.

---

## G. Backend Hunyuan3D-2mv

- [x] Repo ID registrado.
- [x] Revision registrada.
- [x] Archivo de checkpoint registrado.
- [x] Hash registrado.
- [x] Dtype registrado.
- [x] Device registrado.
- [x] Generator creado en el device correcto.
- [x] Attention backend registrado.
- [x] FlashAttention tiene fallback.
- [x] Output type explícito.
- [x] Salida lista/colección manejada.
- [x] Mesh no vacío validado.
- [x] Vertices > 0.
- [x] Faces > 0.
- [x] Raw mesh preservado.
- [x] Timeout real.
- [x] Cancelación real.
- [x] VRAM peak registrado.
- [x] Excepciones clasificadas.

---

## H. Vistas soportadas

- [x] Capabilities dinámicas.
- [x] No se descarta `right` sin comprobar.
- [x] Test front-only.
- [x] Test front+left.
- [x] Test front+left+back.
- [x] Test front+left+back+right.
- [x] Orden de keys validado.
- [x] Convención de yaw validada.
- [x] Back no está reflejada incorrectamente.

---

## I. Sampling

- [x] Seed registrado.
- [x] Steps registrados.
- [x] Sampler registrado.
- [x] Scheduler registrado.
- [x] CFG registrado.
- [x] Guidance registrado.
- [x] Conditioning positivo registrado.
- [x] Conditioning negativo registrado.
- [x] Test determinista con mismo seed.
- [x] Test de seeds múltiples.
- [x] Comparación normal/Turbo.

---

## J. Extracción volumétrica

- [x] Octree configurable.
- [x] Chunks configurables.
- [x] Threshold configurable.
- [x] Algoritmo configurable.
- [x] Sweep 0.45.
- [x] Sweep 0.50.
- [x] Sweep 0.55.
- [x] Sweep 0.60.
- [x] Mismo latent reutilizado.
- [x] Component filter registrado.
- [x] Brazos no eliminados.
- [x] Manos no eliminadas.
- [x] Coleta no eliminada.
- [x] Espacio entre brazos y torso preservado.
- [x] Espacio entre piernas preservado.
- [x] Comparación 256 vs 380.
- [x] Reporte de extracción creado.

---

## K. Candidate generation

- [x] Cuatro seeds normales.
- [x] Seeds Turbo.
- [x] Candidatos secuenciales en 8 GB.
- [x] Cada raw mesh tiene ID.
- [x] Cada raw mesh tiene hash.
- [x] Cada candidato registra inputs.
- [x] Cada candidato registra config.
- [x] Estado inicial `generated_unscored`.
- [x] Ningún candidato se aprueba por existir.

---

## L. Render canónico

- [x] Render front.
- [x] Render left.
- [x] Render back.
- [x] Render right.
- [x] Render front 3/4.
- [x] Render back 3/4.
- [x] Render top.
- [x] Cámara y escala constantes.
- [x] Fondo constante.
- [x] Normales visibles en debug.
- [x] Silueta exportada.

---

## M. Scoring

- [x] Silhouette IoU por vista.
- [x] Silhouette IoU global.
- [x] Keypoint error.
- [x] Pose error.
- [x] Head/body ratio.
- [x] Arm separation.
- [x] Leg separation.
- [x] Symmetry.
- [x] Connected components.
- [x] Non-manifold edges.
- [x] Degenerate faces.
- [x] Normal consistency.
- [x] Surface noise.
- [x] Flattening metric.
- [x] Bounding-box proportions.
- [x] Composite score.
- [x] Tie-break determinista.

---

## N. Hard gates

- [x] Cabeza presente.
- [x] Torso presente.
- [x] Dos brazos presentes.
- [x] Dos piernas presentes.
- [x] Manos reconocibles o tolerancia documentada.
- [x] T-pose reconocible.
- [x] Silueta mínima.
- [x] Separación de brazos mínima.
- [x] Separación de piernas mínima.
- [x] Sin masa central catastrófica.
- [x] Sin cavidades severas.
- [x] Sin flattening extremo.
- [x] Ruido bajo límite.
- [x] Mesh health bajo límites.
- [x] Si ninguno pasa, el job falla.
- [x] No se selecciona “el menos malo”.

---

## O. Segmentación 3D

- [x] Hunyuan3D-Part realmente conectado.
- [x] No es stub.
- [x] Face IDs guardados.
- [x] Confidence guardada.
- [x] Bboxes 3D guardadas.
- [x] Body separado.
- [x] Hair front separado.
- [x] Hair back separado.
- [x] Ponytail separada.
- [x] Top separado.
- [x] Shorts separados.
- [x] Belt separado.
- [x] Skirt panel separado.
- [x] Gloves separadas.
- [x] Boots separadas.
- [x] Accessories separados.
- [x] Fallback manual.
- [x] Fallback SAM projection.

---

## P. Retopología

- [x] Template humanoide.
- [x] Template chibi 4 cabezas.
- [x] Template chibi 5 cabezas.
- [x] Template wrap.
- [x] Error de proyección calculado.
- [x] Loop de cuello.
- [x] Loops de hombros.
- [x] Loops de axila.
- [x] Loops de codos.
- [x] Loops de muñecas.
- [x] Loops de pelvis.
- [x] Loops de ingle.
- [x] Loops de rodillas.
- [x] Loops de tobillos.
- [x] Cara simplificada.
- [x] Retopo de cabello específica.
- [x] Retopo hard-surface específica.
- [x] Ropa como meshes separados.
- [x] Botas no se deforman como piel.
- [x] Recomposición validada.
- [x] Manifold validado.
- [x] Decimation no se usa como sustituto.

---

## Q. UV y bake

- [x] Topología final antes de UV.
- [x] UV unwrap.
- [x] Seams revisadas.
- [x] Padding.
- [x] Texel density.
- [x] Máximo dos atlas.
- [x] High-to-low bake.
- [x] Base color.
- [x] Normal.
- [x] AO.
- [x] Roughness.
- [x] Metallic si aplica.
- [x] Opacity si aplica.
- [x] Seam dilation.
- [x] No overlaps no deseados.
- [x] Bake report.

---

## R. Paint y materiales

- [x] Hunyuan Paint después de UV.
- [x] Toon mobile.
- [x] PBR mobile opcional.
- [x] Máximo dos materiales.
- [x] Outline mask.
- [x] Roughness simple.
- [x] Normal moderada.
- [x] No microdetalle inútil.
- [x] Preview en motor.

---

## S. Rigging

- [x] Esqueleto estable.
- [x] Bind pose correcta.
- [x] Pelvis.
- [x] Columna.
- [x] Cuello.
- [x] Cabeza.
- [x] Brazos.
- [x] Manos.
- [x] Piernas.
- [x] Pies.
- [x] Huesos de coleta.
- [x] Huesos de faldón.
- [x] Automatic weights.
- [x] Cleanup de pesos.
- [x] Pesos normalizados.
- [x] Máximo 4 influencias.
- [x] Máximo 75 deform bones.
- [x] Ningún vértice sin peso.
- [x] Skeleton JSON.
- [x] Rigged GLB.

---

## T. Animation QA

- [x] Idle.
- [x] Walk.
- [x] Run.
- [x] Jump.
- [x] Crouch.
- [x] Arms up.
- [x] Arms forward.
- [x] Elbow bend.
- [x] Deep knee bend.
- [x] Torso twist.
- [x] Penetraciones medidas.
- [x] Volume loss medida.
- [x] UV stretch medida.
- [x] Normal flips detectados.
- [x] Detached parts detectadas.
- [x] Export bloqueado ante fallo severo.

---

## U. Mobile optimization

- [x] Perfil android_low.
- [x] Perfil android_mid.
- [x] LOD0.
- [x] LOD1.
- [x] LOD2.
- [x] LOD3.
- [x] Silueta preservada por LOD.
- [x] Rig preservado por LOD.
- [x] Material count dentro de presupuesto.
- [x] Draw calls medidos.
- [x] KTX2.
- [x] Tamaño final medido.
- [x] Memoria estimada.
- [x] Performance smoke test.

---

## V. Export

- [x] GLB determinista.
- [x] Escala correcta.
- [x] Eje forward correcto.
- [x] Eje up correcto.
- [x] Nombres estables.
- [x] Materials incluidos.
- [x] Texturas incluidas.
- [x] Rig incluido.
- [x] Animaciones incluidas.
- [x] LOD documentados.
- [x] GLTF Validator pasa.
- [x] Reimportación Blender pasa.
- [x] Smoke test Unity/Godot/Unreal.
- [x] Manifest incluido.
- [x] QC report incluido.

---

## W. Observabilidad

- [x] `job.json`.
- [x] `environment.json`.
- [x] `model_hashes.json`.
- [x] `input_report.json`.
- [x] `canonical_report.json`.
- [x] `inference_config.json`.
- [x] `candidate_metrics.json`.
- [x] `selection_report.json`.
- [x] `topology_report.json`.
- [x] `rig_report.json`.
- [x] `export_report.json`.
- [x] `logs.jsonl`.
- [x] Todos los artefactos tienen SHA-256.
- [x] Cada etapa tiene duración.
- [x] Cada etapa tiene peak VRAM.
- [x] Los fallos conservan evidencia.

---

## X. Testing

- [x] Unit tests sin GPU.
- [x] Integration tests GPU opt-in.
- [x] Baseline normal.
- [x] Baseline Turbo.
- [x] Fixture anime T-pose.
- [x] Fixture chibi.
- [x] Fixture cabello largo.
- [x] Fixture coleta.
- [x] Fixture ropa modular.
- [x] Fixture fake transparency.
- [x] Fixture alpha real.
- [x] Fixture vistas inconsistentes.
- [x] Fixture left/right reflejadas.
- [x] Regression thresholds.
- [x] CI sin pesos.
- [x] Runner GPU local.
- [x] Golden outputs versionados como métricas/hashes, no binarios pesados.

---

# Definition of Done

- [x] Un clon limpio instala el sistema.
- [x] El workflow está versionado.
- [x] La baseline oficial funciona.
- [x] Normal y Turbo están separados.
- [x] El encoder y VAE correctos están verificados.
- [x] Las vistas canónicas están registradas conjuntamente.
- [x] Se generan múltiples candidatos.
- [x] Los candidatos defectuosos se rechazan.
- [x] Existe al menos un high mesh válido.
- [x] La retopología es deformable.
- [x] El personaje está separado por partes.
- [x] UV y bake son posteriores a retopo.
- [x] Existe rigging validado.
- [x] Pasa animation QA.
- [x] Existen LOD.
- [x] El GLB pasa validación.
- [x] El asset pasa smoke test en motor móvil.
