# Análisis técnico integral — Pipeline Hunyuan3D multivista para personajes

**Repositorio:** `anuaralejandro/txt-timage-to-3d-asset`  
**Rama auditada:** `feat/hunyuan-multiview-character-pipeline`  
**Objetivo declarado:** convertir vistas ortográficas de un personaje anime en T-pose en un asset 3D de alta calidad, optimizado para Android/iOS, con geometría, textura, retopología, rigging, LOD y exportación GLB.

---

## 1. Conclusión ejecutiva

El resultado gris y deformado no se explica por una sola causa. El sistema presenta una combinación de problemas de **versionado**, **reproducibilidad**, **configuración del checkpoint**, **conditioning multivista**, **muestreo**, **extracción volumétrica**, **normalización de cámaras**, **control de calidad inexistente o incompleto** y **expectativas arquitectónicas incompatibles con una malla generativa raw**.

La transparencia falsa fue una causa inicial válida, pero ya no debe considerarse el principal sospechoso si el nuevo preflight realmente genera alpha útil y las imágenes canónicas muestran el personaje recortado correctamente. El foco actual debe desplazarse a los siguientes puntos:

1. El workflow real de ComfyUI no está versionado de forma reproducible en el repositorio padre.
2. El instalador descarga el checkpoint **Turbo**, pero no garantiza que el workflow use el conditioning y guidance exigidos por Turbo.
3. La reconstrucción probablemente utiliza una resolución volumétrica y un threshold que destruyen brazos, manos, cabello y espacios negativos.
4. Existe riesgo de usar un image encoder incompatible, especialmente si se conecta `clip_vision_g` donde el checkpoint MV espera su conditioning DINO/multivista.
5. El sistema descarta la vista derecha por una restricción no demostrada.
6. Las cuatro imágenes se recortan y escalan por separado, sin registro anatómico multivista conjunto.
7. Cualquier GLB generado puede marcarse como aprobado aunque sea una masa inutilizable.
8. La mayor parte de la arquitectura “AAA mobile-ready” todavía es contrato, scaffolding o backlog, no una implementación end-to-end validada.
9. La malla generativa de Hunyuan3D-2mv debe tratarse como **high mesh / candidato geométrico**, no como el asset final de producción.

---

## 2. Alcance y limitaciones de esta auditoría

Se revisaron los archivos visibles en la rama, entre ellos:

- `services/hunyuan3d_2mv/backend.py`
- `src/local_asset_factory/geometry/hunyuan2mv_client.py`
- `src/local_asset_factory/multiview/canonical_views.py`
- `src/local_asset_factory/domain/models.py`
- `src/local_asset_factory/domain/enums.py`
- `configs/pipeline/default.yaml`
- `ComfyUI_windows_portable/download_pipeline_models.py`
- `ComfyUI_windows_portable/run_nvidia_gpu.bat`
- `analysis.md`
- `task.md`
- `checklist.md`
- scripts Blender y servicios auxiliares visibles en la rama

### Limitación crítica

La ruta:

```text
ComfyUI_windows_portable/ComfyUI
```

está registrada como un repositorio anidado o gitlink. El repositorio padre no expone de forma normal el contenido de:

```text
ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/
ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/workflows/
```

Por ello, `04_hunyuan_multiview_unified.json` no puede auditarse de forma reproducible desde el repositorio padre. Hasta corregir esto, cualquier diagnóstico del grafo ComfyUI será necesariamente parcial.

---

# 3. Problemas confirmados

## P0.1 — El workflow y los custom nodes no están versionados de forma reproducible

### Evidencia

La rama agrega `ComfyUI_windows_portable/ComfyUI` como una sola entrada, no como archivos individuales. Esto es consistente con un directorio que contiene su propio `.git`.

### Impacto

- No se puede revisar el workflow exacto.
- No se puede reproducir la inferencia desde un clon limpio.
- No se conocen conexiones, nodos, widgets, valores o versiones reales.
- Los cambios locales pueden no estar en GitHub aunque el desarrollador crea que sí.
- Es imposible asociar un GLB con el workflow que lo produjo.

### Verificación local

```powershell
git ls-files --stage ComfyUI_windows_portable/ComfyUI
```

Si el primer campo es `160000`, es un gitlink.

### Corrección

Versionar fuera del repositorio interno:

```text
workflows/04_hunyuan_multiview_unified.json
comfyui/ComfyUI-LocalAssetFactory/
```

Añadir un script de instalación o sincronización:

```text
scripts/install_comfy_nodes.ps1
scripts/install_comfy_nodes.py
```

El script debe copiar o enlazar los nodos y workflows hacia la instalación local de ComfyUI.

---

## P0.2 — El instalador descarga Turbo sin asegurar un workflow Turbo correcto

### Evidencia

`download_pipeline_models.py` descarga:

```text
tencent/Hunyuan3D-2mv/
hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors
```

y lo renombra como checkpoint de ComfyUI. No descarga el modelo normal MV en la misma ruta de instalación.

### Riesgo técnico

Los checkpoints distilled/Turbo suelen requerir una configuración distinta a la del modelo normal. En particular, no debe asumirse que funcionan correctamente con:

- CFG tradicional alto.
- Scheduler arbitrario.
- Guidance no compatible.
- Número de pasos heredado del modelo normal.
- Conditioning negativo convencional no previsto por el workflow Turbo.

Una configuración incorrecta puede producir:

- masa central amorfa;
- pérdida de extremidades;
- siluetas fusionadas;
- geometría casi plana o perforada;
- ruido volumétrico;
- resultados extremadamente sensibles al seed.

### Corrección

Mantener dos baselines separados:

```text
hunyuan3d-dit-v2-mv.safetensors
hunyuan3d-dit-v2-mv-turbo.safetensors
```

Los workflows deben estar separados o parametrizados explícitamente:

```text
04a_hunyuan_multiview_normal.json
04b_hunyuan_multiview_turbo.json
```

Nunca permitir que un checkpoint Turbo use silenciosamente los parámetros del normal.

---

## P0.3 — No existe una baseline oficial congelada

### Problema

No hay un paquete de evidencia que demuestre que la instalación local de Hunyuan3D-2mv funciona correctamente con:

- workflow oficial;
- imágenes oficiales;
- checkpoint correcto;
- VAE correcto;
- encoder correcto;
- sampler y scheduler correctos;
- extracción de malla correcta.

### Impacto

Actualmente no puede distinguirse entre:

1. instalación dañada;
2. pesos equivocados;
3. workflow equivocado;
4. imágenes difíciles;
5. threshold inadecuado;
6. bug de postprocesamiento.

### Corrección

Crear:

```text
benchmarks/baseline_official/
├── input/
├── workflow.json
├── environment.json
├── model_hashes.json
├── output_raw.glb
├── renders/
└── report.md
```

La baseline debe ejecutarse antes de probar el personaje Jace.

---

## P0.4 — Resolución volumétrica y threshold potencialmente destructivos

### Síntoma compatible

La malla observada tiene:

- planos grandes y triangulados;
- brazos o huecos colapsados;
- masa central gruesa;
- ruido escalonado;
- cavidades;
- pérdida de partes delgadas.

Esto es típico de una extracción a baja resolución o threshold demasiado alto.

### Parámetros que deben registrarse

- latent resolution;
- VAE;
- `octree_resolution`;
- `num_chunks`;
- algoritmo de extracción;
- `mesh_threshold`;
- simplificación posterior;
- weld/merge distance;
- remove small components;
- decimation ratio.

### Baseline recomendada

```yaml
octree_resolution: 380
num_chunks: 20000
mesh_algorithm: surface_net
threshold_sweep:
  - 0.45
  - 0.50
  - 0.55
  - 0.60
```

El threshold debe evaluarse usando el mismo latent para aislar la etapa de extracción.

---

## P0.5 — Posible image encoder incompatible

### Evidencia de riesgo

El instalador descarga:

```text
clip_vision_g.safetensors
```

El checkpoint Hunyuan3D-2mv utiliza una arquitectura de conditioning multivista específica. Si el workflow sustituye dicho encoder por CLIP Vision G genérico, puede aceptar tensores sin producir un conditioning semánticamente correcto.

### Consecuencia

- el modelo no interpreta correctamente las vistas;
- la forma se aproxima a un volumen promedio;
- las vistas pueden mezclarse o perder correspondencia;
- la calidad no mejora al agregar imágenes.

### Acción obligatoria

Auditar el JSON y documentar:

```text
loader exacto
image encoder exacto
node de conditioning exacto
orden de vistas
normalización esperada
resolución de entrada
dtype
device
```

No aceptar una cadena “funciona porque no arroja error”.

---

## P0.6 — El backend Python no replica una inferencia de referencia completa

En `services/hunyuan3d_2mv/backend.py`, la llamada central solo pasa:

```python
image=pil_views
num_inference_steps=steps
generator=...
```

No fija explícitamente:

- `octree_resolution`;
- `num_chunks`;
- `output_type`;
- device del generator;
- revision del modelo;
- dtype;
- offload policy;
- parámetros de extracción;
- hash de las imágenes canónicas;
- hash de los pesos.

Además, el código exporta directamente el objeto devuelto sin validar de forma explícita si el pipeline retorna una lista o colección de meshes.

### Corrección

Crear un contrato de inferencia completo y serializable:

```yaml
checkpoint:
revision:
variant:
dtype:
device:
views:
seed:
steps:
guidance:
sampler:
scheduler:
octree_resolution:
num_chunks:
mesh_algorithm:
mesh_threshold:
output_type:
```

El manifest debe contener todos estos valores.

---

## P0.7 — Se marca el candidato como aprobado solo porque existe un GLB

En `hunyuan2mv_client.py`, un resultado exitoso del backend crea un `GeometryCandidate` con:

```python
passed_gates=True
```

aunque todavía no se hayan calculado las métricas.

### Impacto

Una roca, masa, malla vacía o personaje sin brazos puede avanzar a etapas posteriores.

### Corrección

El estado correcto después de generación debe ser:

```text
generated_unscored
```

y no aprobado.

Flujo requerido:

```text
generated
→ rendered
→ metrics_computed
→ hard_gates_evaluated
→ ranked
→ selected/rejected
```

`passed_gates` solo puede establecerse después de los hard gates.

---

# 4. Problemas de vistas y conditioning

## P1.1 — Normalización individual en lugar de registro multivista conjunto

`canonical_views.py`:

1. obtiene bbox por alpha;
2. recorta cada imagen;
3. crea un canvas cuadrado;
4. centra;
5. redimensiona.

El procedimiento se ejecuta independientemente por vista.

### Qué no corrige

- ojos a diferente altura;
- cabeza a diferente escala;
- hombros desalineados;
- pelvis desplazada;
- pies en distinta línea basal;
- torso más largo en una lateral;
- brazos a distinta altura;
- coleta con diferente origen;
- cámara con perspectiva desigual.

### Solución

Crear un `MultiviewRegistrationResult` basado en landmarks:

```text
head_top
chin
neck_center
shoulder_left/right
elbow_left/right
wrist_left/right
pelvis_center
knee_left/right
ankle_left/right
foot_baseline
```

La transformación debe calcularse como conjunto, no maximizar cada vista por separado.

---

## P1.2 — Las vistas se etiquetan como ortográficas sin demostrarlo

El código guarda:

```python
camera_type=ORTHOGRAPHIC
camera_yaw_deg=...
```

pero esto es metadata declarativa. No demuestra que la imagen haya sido generada con cámara ortográfica.

### Riesgo

Si las vistas laterales usan perspectiva o diferente distancia focal:

- el pecho y pelvis cambian de volumen;
- la cabeza cambia de proporción;
- manos y botas se escalan de forma desigual;
- la reconstrucción multivista se vuelve contradictoria.

### Solución

El preflight debe calcular indicadores de consistencia y permitir:

```text
PASS orthographic_like
WARN weak_perspective
FAIL incompatible_perspective
```

---

## P1.3 — La política de descartar `right` es demasiado rígida

El proyecto define que `right` es solo para QA y el backend la elimina.

### Problema

El soporte exacto debe determinarse por la implementación/checkpoint realmente instalado, no por una interpretación restrictiva del ejemplo mínimo.

### Solución

Implementar capabilities dinámicas:

```json
{
  "supported_views": ["front", "left", "back", "right"],
  "required_views": ["front"],
  "recommended_view_sets": [
    ["front", "left", "back"],
    ["front", "left", "back", "right"]
  ]
}
```

Ejecutar A/B con tres y cuatro vistas.

---

## P1.4 — No hay verificación de orden semántico de vistas

Una vista mal conectada puede ser válida como imagen, pero incorrecta como orientación:

- left/right intercambiadas;
- back conectada como front;
- imagen reflejada;
- convención de yaw opuesta;
- manos o accesorios asimétricos confundidos.

### Solución

Añadir validaciones:

- face visible en front;
- face ausente o mínima en back;
- nariz orientada correctamente en left/right;
- embeddings de front/back distintos;
- detección de espejo;
- hash y preview etiquetados.

---

## P1.5 — El fondo transparente no basta: el RGB de píxeles transparentes puede contaminar bordes

Aunque alpha sea real, los píxeles con alpha 0 pueden conservar checkerboard, negro o blanco en RGB.

### Riesgo

Al redimensionar con Lanczos, el RGB oculto puede mezclarse en el borde y crear halos.

### Solución

Antes de resize:

```python
rgb[alpha == 0] = neutral_background
```

o premultiplicar alpha correctamente.

Guardar dos artefactos:

```text
canonical_rgba.png
canonical_composited_neutral.png
```

y usar el formato exacto esperado por el preprocesador.

---

# 5. Problemas del modelo y entorno

## P1.6 — Dos rutas de inferencia sin una fuente única de configuración

Existen:

1. workflow ComfyUI nativo;
2. backend Python `Hunyuan2MVBackend`.

No está demostrado que compartan:

- checkpoint;
- VAE;
- encoder;
- steps;
- guidance;
- scheduler;
- octree;
- threshold;
- preprocessing.

### Impacto

Un resultado de ComfyUI no puede compararse directamente con el backend Python.

### Solución

Crear una especificación única:

```text
configs/models/hunyuan3d_2mv_normal.yaml
configs/models/hunyuan3d_2mv_turbo.yaml
```

ComfyUI y Python deben leer o exportar los mismos valores.

---

## P1.7 — Falta pinning de revisiones y hashes

El código registra el nombre del checkpoint, pero no siempre su revisión real.

### Riesgo

Una actualización silenciosa puede cambiar resultados.

### Solución

Registrar:

- repo ID;
- revision/commit;
- archivo;
- SHA-256;
- tamaño;
- dtype;
- versión de `hy3dgen`;
- versión de ComfyUI;
- versión de PyTorch/CUDA;
- driver;
- GPU.

---

## P1.8 — FlashAttention se habilita incondicionalmente

El backend llama:

```python
self._model.enable_flashattn()
```

sin comprobar:

- compatibilidad;
- versión;
- dtype;
- fallback;
- equivalencia numérica.

### Solución

Usar una capability:

```yaml
attention_backend: auto
allow_flash_attention: true
fallback: sdpa
```

Registrar cuál se usó realmente.

---

## P1.9 — El generador de PyTorch no especifica dispositivo

El código usa:

```python
torch.Generator().manual_seed(seed)
```

### Riesgo

Dependiendo de la implementación, el generator CPU puede no corresponder al device esperado o producir diferencias.

### Solución

```python
torch.Generator(device=device).manual_seed(seed)
```

---

# 6. Problemas de scoring y selección

## P1.10 — Las métricas existen como modelos, pero no como circuito de rechazo demostrado

El repositorio define:

- silhouette IoU;
- semantic IoU;
- keypoint error;
- pose error;
- head/body ratio;
- arm separation;
- leg separation;
- mesh health;
- surface noise.

Sin embargo, no está demostrado que el workflow actual:

1. renderice cada candidato;
2. calcule métricas;
3. aplique hard gates;
4. rechace todos si ninguno pasa;
5. seleccione el mejor;
6. preserve el raw de cada candidato.

### Corrección

Hard gates mínimos:

```yaml
head_present: true
both_arms_present: true
both_legs_present: true
min_silhouette_iou: 0.55
min_arm_separation: 0.08
min_leg_separation: 0.05
max_non_manifold_edges: 500
max_degenerate_faces: 100
max_surface_noise: 0.35
```

No seleccionar “el menos malo” si todos fallan.

---

## P1.11 — No se diferencia fallo de forma y fallo de extracción

Un mesh malo puede venir de:

- latent malo;
- VAE decode malo;
- threshold malo;
- componente pequeño eliminado;
- postprocesamiento Blender.

### Solución

Guardar artefactos intermedios:

```text
conditioning_preview/
latent_metadata.json
decoded_volume_metadata.json
mesh_threshold_045.glb
mesh_threshold_050.glb
mesh_threshold_055.glb
mesh_threshold_060.glb
raw_unmodified.glb
postprocessed.glb
```

---

# 7. Problemas de postprocesamiento y expectativas

## P1.12 — Decimation no equivale a retopología

Reducir triángulos de una malla generativa:

- no crea loops de hombro;
- no crea loops de codo;
- no mejora axila o ingle;
- no separa ropa, cuerpo y cabello;
- no crea topología facial;
- no preserva deformación.

### Solución

Para personajes humanoides:

```text
high mesh seleccionado
→ segmentación por partes
→ template wrap corporal
→ retopología por parte
→ recomposición
→ UV
→ bake
→ rigging
```

---

## P1.13 — Hunyuan3D-2mv no genera por sí solo un asset AAA listo para móvil

La salida raw debe considerarse:

```text
geometric candidate / high mesh
```

No:

```text
final production asset
```

Un asset listo para Android/iOS requiere:

- topología deformable;
- rig;
- skin weights;
- LOD;
- UV estable;
- texturas;
- materiales;
- límites de influencias;
- compresión;
- colliders;
- validación GLTF;
- pruebas en motor.

---

## P1.14 — El sistema no garantiza modularidad de prendas

Una reconstrucción completa desde vistas vestidas tiende a fusionar:

- piel;
- top;
- shorts;
- cinturón;
- paneles;
- cabello;
- botas.

### Solución estratégica

Mantener un cuerpo base canónico con:

- topología fija;
- UV fija;
- rig fijo;
- proporciones parametrizables.

Generar o reconstruir por separado:

```text
body
hair
top
shorts
boots
gloves
accessories
```

---

# 8. Problemas de implementación incompleta

## P2.1 — Gran parte del pipeline es scaffolding

El repositorio contiene contratos, dataclasses, configuraciones, READMEs y scripts iniciales, pero no demuestra una ejecución completa de:

- SAM semántico real;
- consistency gate;
- múltiples candidatos;
- scoring real;
- Hunyuan3D-Part real;
- retopología real;
- UV/bake real;
- Hunyuan Paint real;
- rigging real;
- animation QA real;
- export mobile validado.

### Consecuencia

La arquitectura documentada es más madura que la implementación ejecutable.

---

## P2.2 — `allow_partial_success: true` puede ocultar fallos

En configuración se permite éxito parcial.

### Riesgo

Un job puede producir un GLB aunque fallen:

- scoring;
- part segmentation;
- retopology;
- rigging;
- QA.

### Solución

Definir perfiles:

```yaml
research_debug:
  allow_partial_success: true

production_mobile:
  allow_partial_success: false
```

---

## P2.3 — Rutas absolutas y entorno local

Hay scripts con rutas como:

```text
C:\Users\datam\Videos\...
```

### Impacto

- no reproducible;
- no portable;
- falla en otra máquina;
- complica CI;
- impide instalación limpia.

### Solución

Usar:

- `.env`;
- paths relativos;
- configuración;
- detección automática;
- CLI `doctor`.

---

# 9. Arquitectura recomendada

```text
INPUT VIEWS
    ↓
Preflight técnico
    ↓
Limpieza alpha/RGB
    ↓
Registro anatómico conjunto
    ↓
Canonical view set
    ↓
Baseline de conditioning
    ↓
Candidate generation:
  - Hunyuan3D-2mv normal
  - Hunyuan3D-2mv turbo
  - seeds múltiples
  - thresholds múltiples
    ↓
Raw artifact preservation
    ↓
Canonical renders
    ↓
Hard gates + scoring
    ↓
Selected high mesh
    ↓
3D part segmentation
    ↓
Body template wrap + per-part retopo
    ↓
Recompose
    ↓
UV + high-to-low bake
    ↓
Paint/materials
    ↓
Rig + weights
    ↓
Animation QA
    ↓
LOD + KTX2
    ↓
GLB validation + engine smoke test
```

---

# 10. Matriz de diagnóstico obligatoria

## Etapa A — Instalación

| Caso | Input | Modelo | Workflow | Resultado esperado |
|---|---|---|---|---|
| A1 | oficial | normal | oficial | mesh correcto |
| A2 | oficial | turbo | oficial turbo | mesh correcto |
| A3 | oficial | normal | workflow propio | comparable a A1 |
| A4 | oficial | turbo | workflow propio | comparable a A2 |

Si A1 falla, no continuar con Jace.

## Etapa B — Vistas

| Caso | Vistas |
|---|---|
| B1 | front |
| B2 | front + left |
| B3 | front + left + back |
| B4 | front + left + back + right |

## Etapa C — Extracción

Usar el mismo latent:

| Caso | Octree | Threshold |
|---|---:|---:|
| C1 | 256 | 0.60 |
| C2 | 380 | 0.60 |
| C3 | 380 | 0.55 |
| C4 | 380 | 0.50 |
| C5 | 380 | 0.45 |

## Etapa D — Checkpoint

| Caso | Variante | Steps | Guidance |
|---|---|---:|---|
| D1 | normal | oficial | oficial |
| D2 | normal | 30 | documentado |
| D3 | turbo | oficial | oficial turbo |
| D4 | turbo | configuración actual | configuración actual |

---

# 11. Definition of Done técnica

El problema de “GLB de mierda” se considera resuelto únicamente cuando:

- el workflow está versionado y puede clonarse;
- la baseline oficial genera una malla válida;
- los hashes de modelos están registrados;
- el conditioning usa los componentes correctos;
- las vistas canónicas están alineadas;
- se preservan brazos, piernas, cabeza y cabello;
- existe separación espacial suficiente en T-pose;
- el mesh raw se guarda antes de modificarlo;
- el pipeline rechaza candidatos defectuosos;
- se genera más de un candidato;
- el mejor se selecciona con métricas;
- la retopología no depende de decimation simple;
- existe rigging validado;
- existen LOD;
- el GLB pasa validación y smoke test en motor.

---

# 12. Prioridad recomendada

## P0 — Antes de seguir generando

1. Sacar workflow/custom nodes del gitlink.
2. Congelar baseline oficial.
3. Separar normal y turbo.
4. Auditar image encoder y conditioning.
5. Registrar octree, chunks, threshold y algoritmo.
6. Desactivar postproceso y guardar raw.
7. Cambiar `passed_gates=True` por estado no evaluado.

## P1 — Calidad de reconstrucción

1. Registro anatómico conjunto.
2. A/B de 3 vs 4 vistas.
3. Barrido de seeds y thresholds.
4. Renders canónicos.
5. Hard gates.
6. Candidate ranking.

## P2 — Producción

1. Segmentación 3D.
2. Template wrap corporal.
3. Retopología por parte.
4. UV/bake.
5. Paint.
6. Rigging.
7. QA.
8. LOD y export.

---

## Veredicto

La transparencia ya no es suficiente para explicar el fallo. El sistema sigue vulnerable en el **núcleo de inferencia y extracción**, y además acepta cualquier GLB como si fuera un candidato válido. Antes de invertir en rigging, pintura o optimización móvil, debe demostrarse que la etapa Hunyuan3D-2mv produce consistentemente una silueta humana completa con el workflow oficial y con parámetros registrados.
