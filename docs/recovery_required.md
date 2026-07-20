# Recovery Required — State vs Declaration

**Fecha de auditoría:** 2026-07-20  
**Rama:** `feat/hunyuan-multiview-character-pipeline`

---

## Estado del repositorio antes de esta rama

| Componente | Declarado en README | Estado real | Acción |
|---|---|---|---|
| `nodes.py` | ✓ | ✓ Existe | Docstring actualizado |
| `schemas.py` | ✓ | ✓ Existe | Ampliación pendiente (M1) |
| `services/` | ✓ | ✓ Existe (parcial) | Reestructurado |
| `blender/` | ✓ | ✓ Existe (6 scripts) | Ampliación pendiente |
| `workflows/` | ✓ | ✓ Existe | Limpiado |

---

## Problemas encontrados y resueltos en M0

### 1. TRELLIS presente — ELIMINADO

Los siguientes archivos existían e introducían una dependencia en TRELLIS, que está en la lista de exclusiones:

- `services/trellis_adapter.py` → **ELIMINADO**
- `workflows/trellis_local_v1.json` → **ELIMINADO**
- `config.py: ENABLE_TRELLIS, TRELLIS_WORKFLOW_PATH` → **ELIMINADOS**
- `__init__.py: AssetFactory_LocalTrellis alias` → **ELIMINADO**
- `nodes.py: docstring referenciando LocalTrellisNode` → **ACTUALIZADO**

### 2. Stack incompleto vs objetivo

El sistema original implementaba:
```
texto -> imagen -> Hunyuan3D -> decimate -> UV -> GLB
```

El sistema objetivo implementa:
```
multiview validado
-> SAM 3.1 máscaras 2D
-> Hunyuan3D-2mv candidatos (múltiples seeds)
-> Hunyuan3D-Omni pose candidato
-> scoring multivista
-> Hunyuan3D-Part segmentación
-> retopología por tipo de parte
-> UV y bake
-> Hunyuan3D-Paint
-> rigging (RigAnything + Rigify fallback)
-> animation QA
-> LOD móvil
-> GLB validado
```

### 3. Contratos insuficientes

`schemas.py` original cubría props/weapons. Los contratos de personaje (CanonicalView, GeometryCandidate, Part3D, etc.) no existían. → **Implementación pendiente M1**.

### 4. Sin preflight

No existía detección de:
- alfa falso (RGBA opaco)
- tablero horneado en RGB
- cuerpo cortado
- T-pose inválida
- vistas inconsistentes

La imagen de referencia con tablero horneado habría entrado al pipeline sin rechazo. → **Implementación M2**.

### 5. Sin segmentación semántica

No existía SAM 3.1 ni ningún equivalente. → **Implementación M2**.

---

## Módulos a crear (milestones)

```
M1: src/local_asset_factory/     — core framework, contratos, artifact store
M2: preflight + SAM 3.1 service  — validación de entrada
M3: geometry generation          — Hunyuan3D-2mv + Omni + scoring
M4: parts + topology             — Hunyuan3D-Part + retopo por parte
M5: texturing + rigging          — Hunyuan3D-Paint + RigAnything + Rigify
M6: mobile export                — LOD + KTX2 + GLB + ComfyUI thin nodes
```

---

## Restricciones de stack confirmadas

Los siguientes modelos NO están presentes y NO serán añadidos:
- TRELLIS / TRELLIS.2
- TripoSR / TripoSG
- SkinTokens
- UniRig

Stack autorizado:
- `tencent/Hunyuan3D-2mv` — backbone multiview principal
- `Tencent-Hunyuan/Hunyuan3D-Omni` — control de pose/bbox/voxel/point
- `Tencent-Hunyuan/Hunyuan3D-2.1` — runtime base + Paint
- `Hunyuan3D-Paint` — texturizado post-UV
- `Tencent-Hunyuan/Hunyuan3D-Part` — segmentación 3D
- `facebookresearch/sam3` con checkpoints SAM 3.1 — segmentación 2D
- `HunyuanImage-3.0-Instruct` o Distil — fase posterior
- `RigAnything` — no comercial, opcional
- Blender + AutoRemesher + QuadriFlow + OpenVDB + Rigify + glTF Validator + KTX2

---

## Hardware objetivo

- GPU: RTX 4070 Laptop 8 GB VRAM
- Estrategia: carga/descarga secuencial de modelos ML
- Sin inferencia simultánea de múltiples modelos
