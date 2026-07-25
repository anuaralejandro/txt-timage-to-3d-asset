# TASK — Segmentación semántica 3D y texturizado multivista para Hunyuan3D en 8 GB

## Estado

`Planned`

## Prioridad

Alta

## Contexto

El repositorio ya dispone de un pipeline ComfyUI funcional que convierte cuatro imágenes de un personaje en un GLB mediante Hunyuan3D-2mv Turbo. El workflow actual:

```text
04_hunyuan_multiview_unified.json
```

realiza, de forma resumida:

```text
Load 4 views
→ remove background
→ DINOv2 encodings
→ Hunyuan3D multiview conditioning
→ sampling
→ voxel decode
→ mesh
→ save GLB
→ Blender smoothing/optimization
→ 3D preview
```

La geometría generada y su postprocesamiento ya son aceptables. El objetivo de esta tarea es extender la salida limpia de Blender con segmentación anatómica consistente y texturizado multivista controlado semánticamente.

## Objetivo

Implementar una etapa modular de segmentación híbrida 2D/3D que produzca etiquetas por cara para:

```text
torso
neck
head
hair
arm_L
arm_R
leg_L
leg_R
```

y usar dichas etiquetas para proyectar y hornear las cuatro imágenes originales sobre el modelo sin contaminación entre partes.

La solución debe ser operable en una RTX 4070 Laptop con 8 GB de VRAM.

---

# Alcance

## Incluido

- segmentación del GLB limpio;
- creación de proxy para high-poly;
- renders semánticos multivista;
- Face ID por píxel;
- human parsing;
- pose y lateralidad;
- refinamiento SAM2;
- P3-SAM como prior geométrico;
- fusión de evidencia 2D→3D;
- postprocesamiento topológico;
- inferencia de cuello;
- exportación de GLB semántico;
- preview coloreado;
- proyección/bake de texturas multivista;
- administración explícita de VRAM;
- workflow ComfyUI nuevo;
- pruebas y documentación.

## No incluido en el MVP

- rigging automático;
- skin weights;
- separación destructiva obligatoria del mesh;
- materiales PBR generativos completos;
- entrenamiento o fine-tuning de modelos;
- reemplazo del pipeline Hunyuan existente;
- integración obligatoria de Hunyuan3D Paint;
- garantía de segmentación perfecta para personajes no humanoides.

---

# Definición de clases

| ID | Clase | Definición |
|---:|---|---|
| 0 | `unassigned` | Cara sin evidencia suficiente |
| 1 | `torso` | Pecho, abdomen, espalda, pelvis central y ropa correspondiente |
| 2 | `neck` | Región entre cabeza y línea de hombros |
| 3 | `head` | Cara, orejas y cráneo sin cabello |
| 4 | `hair` | Masa geométrica del cabello |
| 5 | `arm_L` | Brazo y mano izquierdos anatómicos |
| 6 | `arm_R` | Brazo y mano derechos anatómicos |
| 7 | `leg_L` | Pierna y pie izquierdos anatómicos |
| 8 | `leg_R` | Pierna y pie derechos anatómicos |

Regla invariante:

```text
L/R = lateralidad anatómica del personaje
```

---

# Arquitectura

```text
04 workflow
    │
    ▼
cleaned GLB from BlenderProcessor
    │
    ├── proxy mesh if high-poly
    │       │
    │       └── P3-SAM geometric regions
    │
    └── Blender semantic render
            ├── RGB
            ├── depth
            ├── world normals
            ├── Face ID
            └── camera matrices
                    │
                    ├── human parser
                    ├── DWPose/OpenPose
                    └── SAM2 refinement
                            │
                            ▼
                    CPU fusion by face
                            │
                            ▼
                    topology postprocess
                            │
                            ├── semantic GLB
                            ├── face_labels.npz
                            ├── semantic_parts.json
                            └── colored preview
                                    │
                                    ▼
                        semantic texture projection
                                    │
                                    ▼
                         UV bake + textured GLB
```

---

# Fases de implementación

## Fase 0 — Auditoría del repositorio

Inspeccionar:

- registro de custom nodes;
- estructura `src/local_asset_factory`;
- `AssetFactory_BlenderProcessor`;
- Blender runner y scripts;
- output manager;
- mesh IO actual;
- preview node;
- requirements;
- tests;
- convenciones de logs;
- workflow JSON.

Entregable: nota breve en la descripción del PR o `docs/segmentation-architecture.md`.

## Fase 1 — Contratos y configuración

Crear:

- enum/constantes de labels;
- configuración serializable;
- dataclasses o Pydantic models;
- esquema `semantic_parts.json`;
- paths de outputs;
- configuración de modelos;
- configuración low-VRAM;
- seed y determinismo donde aplique.

Defaults recomendados:

```yaml
low_vram_mode: true
render_resolution: 768
render_views: 8
batch_size: 1
sam2_model: sam2.1_hiera_small
dtype: float16
proxy_triangle_threshold: 250000
proxy_target_triangles: 150000
p3sam_point_count: 50000
texture_resolution: 1024
uv_dilation_px: 12
```

## Fase 2 — Administración de memoria

Crear un `ModelLifecycleManager` o equivalente.

Funciones mínimas:

- lazy model construction;
- load one backend;
- unload backend;
- move to CPU;
- delete references;
- garbage collection;
- CUDA cache clear;
- peak VRAM reset/read;
- context manager de inferencia;
- log por fase.

No asumir CUDA disponible. Soportar mensajes claros para CPU, aunque algunos modelos sean imprácticos.

## Fase 3 — Proxy mesh

Input:

```text
cleaned_glb_path
```

Output:

```text
segmentation_mesh_path
proxy_mapping_metadata
proxy_used
```

Requisitos:

- preservar transformaciones;
- aplicar transformaciones de manera consistente;
- triangular;
- decimar solo si supera umbral;
- conservar high-poly;
- construir BVH;
- transferir labels proxy→original;
- detectar caras sin correspondencia;
- guardar métricas de distancia.

## Fase 4 — Blender semantic renderer

Crear un script Blender headless reutilizando el runner existente.

Vistas:

```text
front
front_left
left
back_left
back
back_right
right
front_right
```

Cada vista debe guardar:

- RGB neutro;
- depth lineal;
- normals en world space;
- Face ID;
- camera intrinsics/extrinsics;
- view direction;
- bounding box;
- transformación mesh↔world.

Requisitos:

- cámara ortográfica encuadrada automáticamente;
- iluminación neutra;
- sin antialiasing en Face ID;
- oclusión correcta;
- IDs exactos;
- soporte >65k caras;
- prueba de round-trip píxel→face.

## Fase 5 — Human parser

Interfaz:

```python
class HumanParserBackend(Protocol):
    def load(self, config): ...
    def predict(self, image) -> ParserResult: ...
    def unload(self): ...
```

Backend principal:

```text
SCHP-LIP
```

Mapear clases del parser a labels internos. Conservar confidence maps cuando el backend lo permita.

Fallbacks:

- permitir desactivar;
- backend alternativo detrás de la misma interfaz;
- no impedir que ComfyUI inicie sin pesos.

## Fase 6 — Pose y lateralidad

Interfaz equivalente para DWPose/OpenPose.

Keypoints mínimos:

- shoulders L/R;
- elbows L/R;
- wrists L/R;
- hips L/R;
- knees L/R;
- ankles L/R;
- neck/nose/head hints.

Requisitos:

- transformar keypoints a la convención anatómica;
- considerar si la vista es frontal, posterior o lateral;
- asociar pose con cámaras conocidas;
- producir prompts positivos y negativos para SAM2;
- confidence threshold configurable;
- fallback geométrico si faltan keypoints.

## Fase 7 — SAM2 refinement

Backend:

```text
SAM2.1 Hiera Small FP16
```

Uso correcto:

- recibe máscara/bounding box del parser;
- recibe puntos de pose;
- refina bordes;
- procesa una vista y una región a la vez;
- no asigna nombres de clase por sí solo;
- libera embeddings entre vistas cuando low-VRAM lo requiera.

Outputs:

- máscara binaria refinada;
- confidence;
- warnings;
- debug overlay.

## Fase 8 — P3-SAM

Input:

- proxy o mesh de segmentación;
- point count;
- point prompts automáticos o manuales.

Output:

- regiones geométricas;
- confidence/IoU estimada;
- mapping puntos/caras;
- preview de regiones.

Requisitos:

- backend lazy y opcional;
- Sonata y P3-SAM no deben convivir con los demás modelos GPU;
- surface sampling reproducible;
- punto count configurable;
- cache de features opcional;
- no convertir automáticamente region ID en clase anatómica;
- usar regiones como regularizador durante la fusión.

## Fase 9 — Fusión 2D→3D

Crear arrays:

```text
face_scores[F, 9]
face_observation_count[F]
face_best_confidence[F]
```

Por cada píxel válido:

1. leer Face ID;
2. leer label y confidence;
3. validar depth;
4. calcular normal-view weight;
5. aplicar confiabilidad por vista;
6. acumular score;
7. registrar observación.

Peso inicial configurable:

```python
weight = (
    parser_confidence
    * sam_confidence
    * visibility_weight
    * max(0.0, dot(normal, camera_to_surface))
    * view_reliability
)
```

Añadir:

- pose prior;
- P3-SAM region consistency;
- penalización por conflicto anatómico;
- score minimum;
- confidence margin mínimo entre primera y segunda clase.

## Fase 10 — Postprocesamiento topológico

Implementar sobre la adyacencia de caras:

- majority smoothing;
- connected components;
- small-island removal;
- hole filling;
- unassigned propagation;
- left/right spatial and pose constraints;
- one-main-component preference por extremidad;
- preservación de discontinuidades fuertes;
- P3-SAM super-region consistency.

Todos los thresholds deben ser configurables y testeables.

## Fase 11 — Cuello

Pipeline:

1. estimar cabeza;
2. obtener hombros;
3. delimitar banda vertical;
4. seleccionar región central;
5. intersectar con geometría conectada;
6. excluir cabello, torso y brazos;
7. suavizar límite;
8. fallback con proporciones del bounding box.

Debe evitar clasificar pecho superior como cuello.

## Fase 12 — Exportación

Outputs:

```text
character_segmented.glb
character_segmented_debug.glb
face_labels.npz
semantic_parts.json
```

GLB debug:

- material por clase;
- colores distinguibles;
- metadata de labels;
- sin depender de texturas externas para visualizar.

GLB semántico:

- conservar mesh final;
- metadata en `extras`;
- materials/attributes cuando sea técnicamente apropiado;
- no destruir UV existentes;
- recargable mediante trimesh y Blender.

## Fase 13 — Texturizado multivista

### UV

- crear UV si no existe;
- conservar UV válida si ya existe;
- configurar island margin;
- bake a 1024 por defecto;
- 2048 opcional.

### Proyección

Fuentes:

```text
front
left
right
back
```

Cada texel/cara debe escoger o mezclar vistas según:

- visibilidad;
- depth consistency;
- normal angle;
- distancia al borde de la máscara;
- confidence semántica;
- calidad de la vista;
- parte anatómica.

Regla:

```text
una muestra de una clase no debe pintar otra clase
```

Ejemplos:

- hair no pinta head;
- torso no pinta arms;
- arm_L no recibe arm_R;
- leg_L no recibe leg_R.

### Bake

- combinar proyecciones;
- corregir seams;
- dilatar 8–16 px;
- guardar coverage mask;
- guardar confidence texture;
- identificar texels sin observación;
- ofrecer hook opcional de inpainting;
- exportar GLB con albedo horneado.

## Fase 14 — Nodos ComfyUI

Registrar:

```text
AssetFactory_CreateSegmentationProxy
AssetFactory_P3SAMSegment
AssetFactory_RenderSemanticViews
AssetFactory_HumanParseViews
AssetFactory_PoseSemanticHints
AssetFactory_SAM2RefineParts
AssetFactory_FuseSemanticParts
AssetFactory_WriteSemanticGLB
AssetFactory_ProjectMultiviewTexture
AssetFactory_SegmentationPreview
```

Cada nodo debe tener:

- `INPUT_TYPES`;
- `RETURN_TYPES`;
- `RETURN_NAMES`;
- `FUNCTION`;
- `CATEGORY`;
- manejo de errores;
- docstring;
- defaults low-VRAM;
- status legible;
- outputs persistentes.

## Fase 15 — Workflow

Crear:

```text
05_hunyuan_multiview_segment_and_texture_8gb.json
```

Requisitos:

- no modificar destructivamente `04`;
- partir del output limpio de Blender;
- incluir notes/groups;
- exponer ajustes clave;
- incluir bypass de P3-SAM;
- incluir bypass de texturas;
- incluir modo solo segmentación;
- preview intermedio y final;
- rutas correctas entre nodos.

## Fase 16 — Instalación y documentación

Crear o actualizar:

```text
requirements-segmentation.txt
scripts/install_segmentation_backends.*
docs/segmentation-pipeline.md
docs/segmentation-models.md
docs/low-vram-troubleshooting.md
```

Documentar:

- checkpoints;
- directorios;
- licencias;
- instalación Windows;
- instalación Linux;
- cómo abrir workflow;
- parámetros 8 GB;
- errores comunes;
- OOM;
- incompatibilidades;
- cómo ejecutar por etapas;
- cómo depurar máscaras.

---

# Contratos de nodos sugeridos

## CreateSegmentationProxy

Inputs:

```text
MODEL_PATH: STRING
triangle_threshold: INT = 250000
target_triangles: INT = 150000
preserve_boundaries: BOOLEAN = true
```

Outputs:

```text
SEGMENTATION_MODEL_PATH: STRING
PROXY_METADATA_PATH: STRING
STATUS: STRING
```

## RenderSemanticViews

Inputs:

```text
MODEL_PATH: STRING
view_count: 6|8
resolution: INT = 768
orthographic: BOOLEAN = true
```

Outputs:

```text
RENDER_MANIFEST_PATH: STRING
PREVIEW_IMAGES: IMAGE
STATUS: STRING
```

## FuseSemanticParts

Inputs:

```text
MODEL_PATH: STRING
RENDER_MANIFEST_PATH: STRING
PARSER_RESULTS_PATH: STRING
POSE_RESULTS_PATH: STRING
SAM_RESULTS_PATH: STRING
P3SAM_RESULTS_PATH: STRING optional
FUSION_CONFIG_JSON: STRING optional
```

Outputs:

```text
FACE_LABELS_PATH: STRING
SEMANTIC_MANIFEST_PATH: STRING
COLORED_GLB_PATH: STRING
STATUS: STRING
```

## ProjectMultiviewTexture

Inputs:

```text
SEGMENTED_MODEL_PATH: STRING
SEMANTIC_MANIFEST_PATH: STRING
FRONT_IMAGE: IMAGE
LEFT_IMAGE: IMAGE
RIGHT_IMAGE: IMAGE
BACK_IMAGE: IMAGE
texture_resolution: 512|1024|2048
```

Outputs:

```text
TEXTURED_GLB_PATH: STRING
ALBEDO_PATH: STRING
COVERAGE_PATH: STRING
CONFIDENCE_PATH: STRING
STATUS: STRING
```

---

# Rendimiento y VRAM

Configuración objetivo:

| Parámetro | Default |
|---|---:|
| Resolución de segmentación | 768×768 |
| Vistas | 8 |
| Batch | 1 |
| P3-SAM points | 50,000 |
| Proxy target | 150,000 tris |
| SAM2 | Hiera Small FP16 |
| Texture bake | 1024 |
| Paralelismo GPU | 1 |
| Fusión mesh | CPU |

Registrar por fase:

```text
start time
end time
elapsed
GPU allocated
GPU reserved
GPU peak allocated
RAM estimate when possible
warnings
```

---

# Pruebas

## Unitarias

- labels;
- parser mapping;
- L/R convention;
- camera transforms;
- Face ID encoding;
- score accumulation;
- normal weights;
- adjacency;
- connected components;
- island removal;
- neck inference;
- proxy mapping;
- manifests.

## Integración sin modelos pesados

Usar backends deterministas de prueba para verificar:

```text
render → fake parser → fake pose → fake SAM → fusion → export
```

Esto no sustituye las pruebas reales, pero debe cubrir la infraestructura.

## Integración real

Cuando los pesos estén disponibles:

- un personaje T-pose;
- un personaje con cabello separado;
- mesh con manos cercanas al torso;
- high-poly;
- vista posterior;
- ejecución low-VRAM.

## Smoke test

Comando documentado que:

1. toma un GLB existente;
2. ejecuta solo segmentación;
3. produce outputs;
4. recarga el GLB;
5. valida número de caras y labels.

## Regresión

- ComfyUI inicia sin extras;
- workflow `04` abre;
- workflow `05` abre;
- nodos anteriores siguen registrados;
- no hay imports pesados al startup.

---

# Criterios de aceptación medibles

- [ ] 100 % de las caras poseen label válido `0..8`.
- [ ] Al menos 95 % de las caras visibles relevantes no quedan `unassigned` en el fixture principal.
- [ ] `arm_L` y `arm_R` no se intercambian entre front y back.
- [ ] `leg_L` y `leg_R` no se intercambian entre front y back.
- [ ] Hair y head se exportan como labels distintos cuando existe separación visual/geométrica.
- [ ] El GLB segmentado se abre en Blender y trimesh.
- [ ] El número de caras se conserva salvo cambios explícitos documentados.
- [ ] Proxy→high-poly cubre todas las caras.
- [ ] El workflow puede ejecutar segmentación desde un GLB ya generado.
- [ ] El modo low-VRAM procesa vistas secuencialmente.
- [ ] No se cargan dos backends GPU grandes simultáneamente.
- [ ] El pipeline registra peak VRAM.
- [ ] El texturizado no cruza labels en pruebas sintéticas.
- [ ] Existe coverage map para zonas sin textura.
- [ ] No se comitean pesos ni outputs pesados.
- [ ] Todas las pruebas ejecutables pasan.

---

# Entregables

1. módulos de segmentación;
2. scripts Blender;
3. nodos ComfyUI;
4. workflow `05`;
5. dependencias opcionales;
6. scripts de instalación;
7. pruebas;
8. documentación;
9. ejemplo de configuración 8 GB;
10. resumen de resultados y limitaciones.

---

# Definition of Done

La tarea se considera terminada únicamente cuando existe una ruta reproducible:

```text
cleaned_character.glb
+ front/left/right/back images
→ ComfyUI workflow 05
→ character_segmented.glb
→ semantic_parts.json
→ face_labels.npz
→ debug preview
→ textured_character.glb
```

y cuando la etapa puede ejecutarse sin volver a generar la geometría Hunyuan.
