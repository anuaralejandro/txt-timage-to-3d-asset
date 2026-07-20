# Task backlog — Pipeline Hunyuan multiview

## P0 — Recuperación

### HY-0001 Inventariar código real
- Verificar archivos publicados y locales.
- Documentar módulos ausentes.
- Entregable: `docs/current_state.md`.

### HY-0002 Versionar implementación local
- Custom nodes.
- Services.
- Blender scripts.
- Workflows.
- Excluir pesos y outputs.

### HY-0003 Baseline
- Conservar input defectuoso.
- Conservar renders actuales.
- Guardar parámetros y logs.

---

## P0 — Core

### HY-0101 Crear paquete Python
### HY-0102 Contratos Pydantic
### HY-0103 Artifact store
### HY-0104 State machine
### HY-0105 Logging estructurado
### HY-0106 CLI base
### HY-0107 API local

---

## P0 — Preflight

### HY-0201 Detectar alfa real
### HY-0202 Detectar tablero horneado
### HY-0203 Normalizar fondo
### HY-0204 Detectar cuerpo cortado
### HY-0205 Validar T-pose
### HY-0206 Validar consistencia multivista
### HY-0207 Crear `CanonicalViewSet`

Aceptación:

```text
front required
left required
back required
right required for QA
```

---

## P1 — SAM 3.1

### HY-0301 Crear entorno aislado
### HY-0302 Descargar checkpoints autorizados
### HY-0303 Servicio healthcheck
### HY-0304 Segmentación por texto
### HY-0305 Refinamiento por box/puntos
### HY-0306 Batch de cuatro vistas
### HY-0307 Consistencia semántica
### HY-0308 Export masks y confidence

Partes iniciales:

```text
body
hair
face
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

---

## P1 — Hunyuan3D-2mv

### HY-0401 Entorno aislado
### HY-0402 Model manager
### HY-0403 Adapter capabilities
### HY-0404 Validar vistas soportadas
### HY-0405 Generar normal candidates
### HY-0406 Generar turbo candidates
### HY-0407 Seeds múltiples
### HY-0408 Cancelación y timeout
### HY-0409 Registrar VRAM
### HY-0410 Conservar raw meshes

---

## P1 — Hunyuan3D-Omni

### HY-0501 Entorno aislado
### HY-0502 Pose control
### HY-0503 Skeleton T-pose generator
### HY-0504 Proporciones chibi
### HY-0505 Bbox candidate
### HY-0506 Voxel candidate
### HY-0507 Point-cloud candidate
### HY-0508 Candidate metadata

MVP:

```text
pose control
```

---

## P1 — Scoring

### HY-0601 Blender canonical renderer
### HY-0602 Silhouette IoU
### HY-0603 Semantic part IoU
### HY-0604 Keypoint error
### HY-0605 Pose error
### HY-0606 Head/body ratio
### HY-0607 Arm separation
### HY-0608 Leg separation
### HY-0609 Mesh health
### HY-0610 Surface noise
### HY-0611 Candidate ranking
### HY-0612 No-candidate-passed failure

---

## P1 — Hunyuan3D-Part

### HY-0701 Entorno aislado
### HY-0702 P3-SAM adapter
### HY-0703 X-Part adapter
### HY-0704 Semantic naming
### HY-0705 Mask projection fallback
### HY-0706 Manual override
### HY-0707 Part confidence
### HY-0708 Recomposition metadata

---

## P1 — Retopología

### HY-0801 Part classifier
### HY-0802 Common mesh repair
### HY-0803 Humanoid template library
### HY-0804 Chibi 4-head template
### HY-0805 Chibi 5-head template
### HY-0806 AutoRemesher adapter
### HY-0807 QuadriFlow fallback
### HY-0808 Joint loop enforcement
### HY-0809 Shrinkwrap/projection
### HY-0810 OpenVDB hard-surface path
### HY-0811 Hair path
### HY-0812 Ponytail separation
### HY-0813 Topology QA
### HY-0814 Recompose low-poly character

---

## P1 — UV y textura

### HY-0901 UV profiles
### HY-0902 Atlas packing
### HY-0903 High-to-low bake
### HY-0904 Seam validation
### HY-0905 Hunyuan3D-Paint environment
### HY-0906 Toon mobile paint
### HY-0907 PBR mobile paint
### HY-0908 Material budget
### HY-0909 Outline mask
### HY-0910 Texture compression source

---

## P1 — Rigging

### HY-1001 RigAnything environment
### HY-1002 RigAnything adapter
### HY-1003 Rigify metarig placement
### HY-1004 Automatic weights
### HY-1005 Weight normalization
### HY-1006 Four-influence pruning
### HY-1007 Ponytail bones
### HY-1008 Skirt bones
### HY-1009 Project skeleton normalization
### HY-1010 Rig QA
### HY-1011 Rigify fallback path

---

## P1 — Animation QA

### HY-1101 Animation fixture library
### HY-1102 Idle
### HY-1103 Walk
### HY-1104 Run
### HY-1105 Jump
### HY-1106 Crouch
### HY-1107 Arms up
### HY-1108 Arms forward
### HY-1109 Elbow bend
### HY-1110 Deep knee bend
### HY-1111 Torso twist
### HY-1112 Penetration metrics
### HY-1113 Volume loss
### HY-1114 UV stretch
### HY-1115 Export gate

---

## P1 — Mobile

### HY-1201 Android low profile
### HY-1202 Android mid profile
### HY-1203 Android high profile
### HY-1204 iOS profiles
### HY-1205 LOD0
### HY-1206 LOD1
### HY-1207 LOD2
### HY-1208 LOD3
### HY-1209 Preserve skinning
### HY-1210 KTX2/BasisU
### HY-1211 Draw-call report
### HY-1212 Bone budget
### HY-1213 Device benchmark

---

## P1 — Export

### HY-1301 GLB exporter
### HY-1302 glTF Validator
### HY-1303 Blender re-import
### HY-1304 Engine smoke test
### HY-1305 Manifest
### HY-1306 QC report
### HY-1307 Turntable
### HY-1308 Preview renders

---

## P1 — ComfyUI

### HY-1401 Request node
### HY-1402 View validator
### HY-1403 SAM node
### HY-1404 Hunyuan2mv candidates
### HY-1405 Omni pose
### HY-1406 Candidate ranking
### HY-1407 Hunyuan Part
### HY-1408 Retopology
### HY-1409 UV/bake
### HY-1410 Hunyuan Paint
### HY-1411 Rig
### HY-1412 Animation QA
### HY-1413 Mobile optimize
### HY-1414 Export
### HY-1415 Job monitor
### HY-1416 3D preview

---

## P2 — Texto a multiview

### HY-1501 HunyuanImage environment
### HY-1502 Front concept generation
### HY-1503 Image-to-image side views
### HY-1504 Back view derivation
### HY-1505 SAM consistency
### HY-1506 Automatic regeneration
### HY-1507 Hardware profile
### HY-1508 Distilled checkpoint support

---

## P1 — Testing

### HY-1601 Unit tests
### HY-1602 Integration markers
### HY-1603 Golden anime T-pose
### HY-1604 Golden chibi
### HY-1605 Golden ponytail
### HY-1606 Golden skirt
### HY-1607 Golden boots
### HY-1608 Fake checkerboard
### HY-1609 Inconsistent views
### HY-1610 End-to-end smoke
### HY-1611 Quality regression report
### HY-1612 CI without GPU
### HY-1613 Local GPU benchmark

---

## Milestones

### M0 Recuperado
HY-0001 a HY-0003.

### M1 Input fiable
HY-0101 a HY-0308.

### M2 Multiview geometry
HY-0401 a HY-0612.

### M3 Segmented low-poly
HY-0701 a HY-0814.

### M4 Textured and rigged
HY-0901 a HY-1115.

### M5 Mobile export
HY-1201 a HY-1416.

### M6 Text-to-asset
HY-1501 a HY-1508.

---

## Definition of Done

- Sin TRELLIS.
- Sin Tripo.
- Hunyuan3D-2mv funcional.
- Omni pose funcional.
- SAM 3.1 funcional.
- Hunyuan3D-Part funcional.
- Retopología por parte.
- UV antes de Paint.
- Rig con fallback.
- Animation QA.
- LOD.
- GLB validado.
- Manifest reproducible.
