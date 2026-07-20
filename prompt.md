# Prompt maestro para agente de terminal — Hunyuan multiview local asset factory

## Rol

Actúa como principal engineer especializado en:

- generación 3D;
- Hunyuan3D;
- procesamiento de mallas;
- Blender headless;
- retopología;
- rigging;
- MLOps local;
- assets móviles.

Trabaja sobre:

```text
anuaralejandro/txt-timage-to-3d-asset
```

Reconstruye el sistema como una fábrica local de personajes 3D estilizados, multivista, segmentados, low-poly, riggeados y exportables.

---

## Decisión de stack no negociable

No debes instalar, integrar, sugerir ni usar:

```text
TRELLIS
TRELLIS.2
TripoSR
TripoSG
SkinTokens
UniRig
```

El stack de ML permitido es:

```text
tencent/Hunyuan3D-2mv
Tencent-Hunyuan/Hunyuan3D-Omni
Tencent-Hunyuan/Hunyuan3D-2.1
Hunyuan3D-Paint
Tencent-Hunyuan/Hunyuan3D-Part
facebookresearch/sam3 con checkpoints SAM 3.1
HunyuanImage-3.0-Instruct o Distil, opcional
RigAnything, opcional y no comercial
```

Tooling permitido:

```text
Blender
OpenVDB
Voxel Remesh
AutoRemesher
QuadriFlow
Rigify
glTF Validator
KTX2/BasisU
```

---

## Objetivo

Implementar:

```text
texto o paquete multivista
-> preflight
-> vistas canónicas
-> SAM 3.1
-> Hunyuan3D-2mv candidates
-> Hunyuan3D-Omni pose candidate
-> scoring
-> Hunyuan3D-Part
-> retopología por parte
-> UV/bake
-> Hunyuan3D-Paint
-> RigAnything/Rigify
-> animation QA
-> LOD móvil
-> GLB
```

---

## Hechos técnicos que debes respetar

1. `Hunyuan3D-2mv` es el checkpoint multivista principal.
2. El ejemplo oficial documenta `front`, `left` y `back`; no inventes soporte para otras claves.
3. Conserva `right` en el contrato interno para validación y scoring.
4. Consulta `capabilities()` del adaptador antes de enviar vistas.
5. Hunyuan3D-Omni acepta controles `pose`, `bbox`, `voxel` y `point`.
6. No asumas que Omni combina todos los controles simultáneamente; genera candidatos separados.
7. SAM 3.1 es segmentación 2D, no segmentación de malla.
8. Hunyuan3D-Part se ejecuta después de seleccionar una malla.
9. Hunyuan3D-Paint se ejecuta después de retopología y UV.
10. Decimation no equivale a retopología.
11. Un mesh quad no equivale a animation-ready.
12. ComfyUI debe ser una capa delgada.

---

## Paso 1 — Inspección

Ejecuta:

```bash
git status --short
git branch --show-current
git log --oneline --decorate -20
find . -maxdepth 6 -type f \
  -not -path './.git/*' \
  -not -path '*/models/*' \
  -not -path '*/output/*' \
  -not -path '*/input/*' \
  -not -path '*/python_embeded/*' \
  | sort
```

Comprueba si existen realmente:

```text
nodes.py
schemas.py
services/
blender/
workflows/
```

Si el README los declara pero no están versionados:

1. crea `docs/recovery_required.md`;
2. no inventes código ausente;
3. localiza la copia local;
4. incorpórala sin pesos ni outputs;
5. congela un baseline.

Crea rama:

```bash
git switch -c feat/hunyuan-multiview-character-pipeline
```

---

## Paso 2 — Estructura

Crea:

```text
src/local_asset_factory/
├── domain/
├── orchestration/
├── preflight/
├── multiview/
├── segmentation2d/
├── geometry/
├── scoring/
├── parts3d/
├── topology/
├── texturing/
├── rigging/
├── animation_qa/
├── optimization/
├── export/
└── observability/
```

Servicios:

```text
services/
├── hunyuan3d_2mv/
├── hunyuan3d_omni/
├── hunyuan3d_part/
├── hunyuan3d_paint/
├── sam3_1/
├── hunyuan_image/
└── riganything/
```

No instales los modelos en el Python de ComfyUI.

---

## Paso 3 — Contratos tipados

Implementa con Pydantic v2:

```text
AssetRequest
InputImage
CanonicalView
CanonicalViewSet
SemanticMask
SemanticMaskSet
GeometryRequest
GeometryCandidate
CandidateMetrics
SelectedGeometry
Part3D
PartSet
TopologyResult
UVResult
TextureResult
RigResult
AnimationQAResult
LODResult
ExportResult
PipelineManifest
```

Cada artefacto debe incluir:

```text
id
relative_path
sha256
producer
checkpoint
revision
seed
created_at
metadata
```

---

## Paso 4 — Artifact store

```text
artifacts/<job_id>/
├── request.json
├── input/
├── normalized/
├── views/
├── masks_2d/
├── candidates/
│   ├── hunyuan2mv/
│   └── omni/
├── selected/
├── parts_3d/
├── topology/
├── uv/
├── textures/
├── rig/
├── animation_qa/
├── lod/
├── exports/
├── previews/
├── metrics.json
├── manifest.json
└── logs/
```

Requisitos:

- escritura atómica;
- rutas relativas;
- hashes;
- resume;
- cancelación;
- cache por hash;
- conservación de candidatos fallidos.

---

## Paso 5 — Preflight

Implementa detectores para:

```text
no_true_alpha
baked_checkerboard_background
dominant_background
subject_cropped
missing_limb
invalid_pose
inconsistent_views
duplicate_views
```

La imagen de prueba con tablero horneado debe fallar.

CLI:

```bash
asset-factory preflight input/front.png
asset-factory validate-views views.yaml
```

---

## Paso 6 — SAM 3.1

Crea un servicio aislado.

Requisitos:

- código actualizado del repo oficial;
- checkpoints SAM 3.1;
- acceso Hugging Face configurable;
- text prompts;
- boxes;
- points;
- batch por vistas;
- salida PNG y RLE;
- confidence;
- runtime metrics.

Taxonomía inicial:

```text
body
hair
face
eyes
top
shorts
belt
skirt_panel
glove_left
glove_right
boot_left
boot_right
red_accessories
```

Implementa consistencia de máscaras entre vistas.

---

## Paso 7 — Canonical multiview

Contrato interno:

```yaml
front: ...
left: ...
back: ...
right: ...
```

Usa `right` para QA aunque el checkpoint no lo consuma.

Normaliza:

- fondo;
- altura del personaje;
- centro de pelvis;
- escala;
- orientación;
- crop;
- pose;
- cámara ortográfica aproximada.

No permitas generar 3D si las vistas no son coherentes.

---

## Paso 8 — Hunyuan3D-2mv

Interfaz:

```python
class Hunyuan2MVBackend:
    def healthcheck(self): ...
    def capabilities(self): ...
    def generate(self, request): ...
```

`capabilities()` debe declarar:

```text
supported_views
supported_variants
required_vram
supported_output_formats
```

Config inicial:

```yaml
checkpoint: tencent/Hunyuan3D-2mv
subfolder: hunyuan3d-dit-v2-mv
seeds: [11, 29, 47, 83]
steps: [30, 40]
variants:
  - normal
  - turbo
```

Guarda cada malla raw sin modificar.

---

## Paso 9 — Hunyuan3D-Omni

Implementa candidatos separados:

```text
pose
voxel
bbox
point
```

MVP obligatorio:

```text
pose
```

Genera una pose T canónica con proporciones configurables:

```yaml
character_proportions:
  total_heads: 4.5
  head_height_ratio: 0.32
  shoulder_width_ratio: 0.28
  arm_abduction_deg: 90
  elbow_flexion_deg: 0
  leg_separation_ratio: 0.12
```

No reemplaces 2mv: Omni es candidato complementario.

---

## Paso 10 — Render y scoring

Blender headless debe generar:

```text
front
left
back
right
front_3q
back_3q
top
turntable
```

Métricas:

```text
silhouette_iou
semantic_part_iou
keypoint_error
pose_error
head_body_ratio_error
arm_separation
leg_separation
symmetry
non_manifold_edges
degenerate_faces
connected_components
normal_consistency
surface_noise
```

No seleccionar ningún candidato por nombre de modelo. Seleccionar por score y gates.

---

## Paso 11 — Hunyuan3D-Part

Servicio aislado con:

```text
P3-SAM
X-Part
```

Salida:

```text
PartSet
```

Cada parte:

```text
semantic_name
confidence
mesh_path
source_face_ids
bbox
symmetry_partner
part_class
```

Si la salida semántica es incorrecta, permitir:

- override manual;
- proyección de máscaras SAM 3.1;
- separación por conectividad;
- reintento con prompts.

---

## Paso 12 — Clasificación y topología

Clasifica cada parte:

```text
organic_deforming
organic_rigid
cloth_deforming
cloth_rigid
hair_rigid
hair_secondary_motion
hard_surface
accessory
```

### Orgánico

```text
repair
-> symmetry
-> humanoid template wrap
-> AutoRemesher/QuadriFlow
-> enforce loops
-> shrinkwrap
-> deformation QA
```

### Hard-surface

```text
OpenVDB/voxel repair
-> planar cleanup
-> limited dissolve
-> bevel
-> weighted normals
-> triangulation
```

### Hair

```text
preserve silhouette
-> simplify hidden surfaces
-> separate ponytail
-> pivots
-> optional bones
```

No decimes el personaje completo como una sola malla.

---

## Paso 13 — UV y bake

Orden obligatorio:

```text
final topology
-> UV
-> atlas
-> bake
-> seam validation
-> paint/material
```

Produce:

```text
base_color
normal
AO
roughness
metallic
emissive optional
opacity optional
```

---

## Paso 14 — Hunyuan3D-Paint

Implementa dos modos:

```text
toon_mobile
pbr_mobile
```

`toon_mobile` debe priorizar:

- colores limpios;
- pocos materiales;
- legibilidad;
- atlas;
- outline mask;
- roughness simple.

No dejes que Paint cambie la geometría final.

---

## Paso 15 — Rigging

### RigAnything

Uso opcional para investigación no comercial.

Entrada:

```text
final low-poly mesh
```

Salida:

```text
skeleton
weights
rigged GLB
```

### Rigify fallback

1. crear metarig;
2. ajustar huesos con keypoints 3D;
3. generar rig;
4. automatic weights;
5. corregir pesos;
6. añadir ponytail/skirt bones;
7. reducir a huesos deformantes;
8. retarget al esqueleto del proyecto.

La exportación no depende de que RigAnything funcione.

---

## Paso 16 — Animation QA

Usa:

```text
idle
walk
run
jump
crouch
arms_up
arms_forward
elbow_bend
deep_knee_bend
torso_twist
```

Mide:

```text
unweighted_vertices
max_influences
weight_sum_error
self_intersections
part_penetrations
volume_loss
uv_stretch
normal_flips
detached_parts
```

Bloquea exportación en fallo severo.

---

## Paso 17 — Mobile optimization

Perfiles:

```yaml
android_low:
  lod0: 16000
  lod1: 8000
  lod2: 3500
  lod3: 1200

android_mid:
  lod0: 28000
  lod1: 14000
  lod2: 6500
  lod3: 2200
```

Límites:

```text
max_materials: 2
max_vertex_influences: 4
max_deform_bones: 75
texture: 1024/2048
compression: KTX2
```

---

## Paso 18 — Export

Genera:

```text
asset_lod0.glb
asset_lod1.glb
asset_lod2.glb
asset_lod3.glb
manifest.json
qc_report.json
turntable.mp4
previews/
```

Validaciones:

- glTF Validator;
- reimport Blender;
- materiales;
- rig;
- animaciones;
- LOD;
- nombres;
- escala;
- ejes.

---

## Paso 19 — ComfyUI

Nodos delgados:

```text
Asset Request
Validate Multiview
SAM 3.1 Segment
Generate Hunyuan 2mv Candidates
Generate Omni Pose Candidate
Rank Candidates
Hunyuan Part
Retopology
UV and Bake
Hunyuan Paint
Rig
Animation QA
Mobile Optimize
Export
Job Monitor
3D Preview
```

El nodo no debe cargar directamente todos los pesos.

---

## Paso 20 — Texto a asset

No es el primer milestone.

Fase posterior:

```text
text
-> HunyuanImage-3.0-Instruct/Distil front concept
-> image-to-image derived views
-> SAM 3.1 masks
-> multiview consistency gate
-> Hunyuan3D-2mv
```

Si no hay hardware para HunyuanImage, el MVP debe aceptar un paquete multivista preparado externamente.

---

## Tests obligatorios

### Unitarios

- contratos;
- state machine;
- checkerboard;
- alfa;
- views;
- masks;
- scoring;
- budgets;
- manifest.

### Integración

- SAM 3.1;
- Hunyuan3D-2mv;
- Omni pose;
- Hunyuan3D-Part;
- Hunyuan3D-Paint;
- Blender;
- RigAnything;
- Rigify.

### Golden

- personaje anime T-pose;
- chibi 4 cabezas;
- cabello largo;
- coleta;
- faldón;
- botas;
- fake transparency;
- vistas inconsistentes.

---

## Criterios de aceptación

- [ ] No existe ninguna dependencia TRELLIS o Tripo.
- [ ] Hunyuan3D-2mv consume vistas válidas.
- [ ] La vista derecha se usa para QA.
- [ ] SAM 3.1 produce máscaras semánticas.
- [ ] Se generan varios candidatos.
- [ ] Omni preserva T-pose.
- [ ] El selector usa métricas.
- [ ] Hunyuan3D-Part separa la malla.
- [ ] Cada parte usa una estrategia de topología.
- [ ] La UV se crea después de retopología.
- [ ] Hunyuan Paint se ejecuta después de UV.
- [ ] Existe rigging con fallback.
- [ ] Pasa animation QA.
- [ ] Existen LOD.
- [ ] Exporta GLB validado.
- [ ] El manifest registra modelos, seeds y licencias.
- [ ] ComfyUI es solo una interfaz.

---

## Forma de trabajo

Por milestone:

1. inspecciona;
2. documenta plan;
3. implementa;
4. prueba;
5. genera evidencia;
6. commit.

Commits sugeridos:

```text
feat(preflight): reject baked checkerboard references
feat(multiview): add canonical four-view contracts
feat(sam3): add isolated SAM 3.1 segmentation service
feat(hunyuan): add Hunyuan3D-2mv candidate backend
feat(omni): add strict T-pose controlled generation
feat(parts): integrate Hunyuan3D-Part segmentation
feat(topology): add per-part retopology strategies
feat(paint): add post-UV Hunyuan3D-Paint pipeline
feat(rigging): add RigAnything candidate and Rigify fallback
feat(mobile): add LOD and KTX2 export
test(e2e): add anime T-pose golden pipeline
```

No abras PR, no hagas merge y no subas pesos sin autorización.
