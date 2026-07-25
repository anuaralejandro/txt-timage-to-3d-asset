# Prompt maestro para Codex

Trabaja directamente en este repositorio:

- Repositorio: `anuaralejandro/txt-timage-to-3d-asset`
- Rama base actual: `feat/hunyuan-multiview-character-pipeline`
- Workflow funcional que debe conservarse:
  `ComfyUI/ComfyUI-LocalAssetFactory/workflows/04_hunyuan_multiview_unified.json`

## Misión

Implementa una extensión production-grade del pipeline ComfyUI existente para segmentar anatómicamente personajes 3D generados por Hunyuan3D multiview y preparar un pipeline de texturizado multivista con control semántico.

El pipeline existente de geometría ya funciona y no debe ser reescrito innecesariamente. Actualmente recibe cuatro vistas, genera un mesh mediante Hunyuan3D-2mv Turbo, guarda un GLB, lo limpia/suaviza con Blender y finalmente lo muestra en el preview. La nueva funcionalidad debe comenzar a partir del **GLB limpio producido por `AssetFactory_BlenderProcessor`**.

La implementación debe funcionar de manera realista en:

- GPU: NVIDIA RTX 4070 Laptop
- VRAM: 8 GB
- ComfyUI local
- Batch de inferencia: 1
- Windows como plataforma primaria, sin impedir Linux
- Modelos de personajes estilizados/anime, smooth y potencialmente high-poly
- Cuatro imágenes fuente: frontal, izquierda, derecha y posterior

## Resultado obligatorio

Producir un GLB segmentado con estas ocho clases semánticas, usando siempre lateralidad anatómica del personaje:

```text
0 background/unassigned
1 torso
2 neck
3 head
4 hair
5 arm_L
6 arm_R
7 leg_L
8 leg_R
```

`L` y `R` representan izquierda y derecha anatómicas del personaje, nunca izquierda y derecha de la pantalla.

El sistema debe exportar, como mínimo:

1. `character_segmented.glb`
2. `face_labels.npy` o `face_labels.npz`
3. `semantic_parts.json`
4. preview GLB coloreado por clase
5. máscaras 2D por vista para depuración
6. métricas básicas y log de VRAM
7. GLB texturizado o, en el MVP, una interfaz completamente funcional para proyectar y hornear las cuatro vistas originales sin contaminación entre clases

## Arquitectura requerida

Implementa una arquitectura híbrida y modular:

1. **P3-SAM / Hunyuan3D-Part**
   - Segmentación geométrica 3D sobre point cloud o proxy mesh.
   - Debe ser un backend opcional cargado de manera lazy.
   - No asumir que sus segmentos ya tienen nombres anatómicos.

2. **Human parser basado en SCHP-LIP o backend equivalente**
   - Debe diferenciar izquierda/derecha para brazos y piernas.
   - Debe reconocer cabello, cara/cabeza y torso/ropa.
   - Implementarlo detrás de una interfaz de backend para poder sustituirlo.

3. **DWPose u OpenPose**
   - Resolver lateralidad anatómica.
   - Proporcionar hombros, codos, muñecas, caderas, rodillas y tobillos.
   - Usar keypoints como pistas, no como única fuente de segmentación.

4. **SAM2.1 Hiera Small FP16**
   - Refinar máscaras 2D.
   - No usar SAM2 como clasificador semántico.
   - Ejecutar una vista o región por vez.

5. **Blender semantic renderer**
   - Renderizar el GLB limpio en 6 u 8 vistas ortográficas.
   - Generar RGB, depth, world normals, Face ID y metadata de cámara.
   - Face ID debe permitir mapear cada píxel visible a una cara del mesh.

6. **Fusión 2D→3D**
   - Acumular votos por cara utilizando:
     - confianza del human parser;
     - confianza de SAM;
     - visibilidad;
     - profundidad;
     - orientación de la normal respecto a la cámara;
     - consistencia entre vistas;
     - regiones propuestas por P3-SAM.
   - Resolver conflictos y caras sin etiqueta mediante conectividad del mesh.

7. **Postprocesamiento**
   - Eliminar islas pequeñas.
   - Majority voting por adyacencia.
   - Mantener el componente conectado principal por extremidad.
   - Evitar cruces izquierda/derecha.
   - Inferir cuello mediante cabeza, hombros y geometría.
   - Mantener cabello separado de cabeza cuando la geometría lo permita.

8. **Texturizado multivista**
   - Reutilizar las cuatro imágenes originales sin fondo.
   - Proyectarlas sobre el GLB segmentado.
   - Evitar texture bleeding entre partes semánticas.
   - Mezclar por visibilidad, profundidad, ángulo de cámara y confianza.
   - Crear UV y bake con Blender.
   - Dilatar bordes UV.
   - Dejar una interfaz opcional de inpainting para texels sin observación.
   - No integrar Hunyuan3D Paint completo como dependencia obligatoria.

## Restricción crítica de VRAM

Nunca mantengas simultáneamente en VRAM Hunyuan3D, DINOv2, VAE, P3-SAM/Sonata, human parser, DWPose y SAM2.

El flujo debe dividirse en fases:

```text
Fase A:
Hunyuan3D → VoxelToMesh → GLB → Blender cleanup → guardar → descargar modelos

Fase B:
cargar GLB limpio → proxy opcional → P3-SAM → guardar resultados → descargar

Fase C:
renders semánticos → human parsing → pose → SAM2 por vista → descargar

Fase D:
fusión y limpieza en CPU → exportación GLB → proyección/bake de texturas
```

Implementa explícitamente:

- carga lazy de modelos;
- `torch.inference_mode()`;
- autocast FP16 donde sea seguro;
- `model.to("cpu")` antes de eliminar;
- `del model`;
- `gc.collect()`;
- `torch.cuda.empty_cache()`;
- registro de `torch.cuda.max_memory_allocated()`;
- batch 1;
- procesamiento secuencial de vistas;
- fallback CPU para operaciones de mesh;
- configuración `low_vram_mode=True` por defecto.

Objetivo de aceptación: el pipeline de segmentación no debe producir OOM en una GPU de 8 GB bajo la configuración recomendada. Si un backend externo no cabe, debe fallar con un mensaje accionable y permitir continuar con un backend más ligero.

## Proxy para meshes high-poly

Si el mesh supera un umbral configurable, por defecto 250,000 triángulos:

1. crear `segmentation_proxy.glb` de 100k–250k triángulos;
2. segmentar el proxy;
3. transferir etiquetas al mesh original mediante BVH nearest-surface o nearest-triangle;
4. validar que cada cara del high-poly reciba una etiqueta válida;
5. conservar el high-poly original para el resultado final.

No destruyas ni reemplaces silenciosamente el GLB limpio original.

## Nodos ComfyUI mínimos

Implementa o adapta estos nodos, conservando las convenciones existentes del proyecto:

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

Puedes combinar nodos únicamente cuando reduzca complejidad sin volver opaca la depuración.

Cada nodo debe:

- validar entradas;
- devolver errores legibles en ComfyUI;
- registrar progreso;
- no bloquear indefinidamente;
- usar rutas dentro del output manager del proyecto;
- devolver paths estables;
- documentar inputs, outputs y defaults;
- poder desactivarse o ejecutarse en modo debug.

## Contratos de datos

Define contratos tipados y versionados.

### `semantic_parts.json`

Debe contener al menos:

```json
{
  "schema_version": "1.0",
  "source_glb": "...",
  "segmented_glb": "...",
  "coordinate_system": "...",
  "labels": {
    "0": "unassigned",
    "1": "torso",
    "2": "neck",
    "3": "head",
    "4": "hair",
    "5": "arm_L",
    "6": "arm_R",
    "7": "leg_L",
    "8": "leg_R"
  },
  "face_count": 0,
  "face_labels_path": "...",
  "proxy_used": false,
  "per_label_face_counts": {},
  "confidence_summary": {},
  "warnings": [],
  "models": {},
  "settings": {}
}
```

### Máscaras 2D

Por vista:

```text
views/<view_name>/
  rgb.png
  depth.exr o depth.npy
  normals.exr o normals.npy
  face_id.png o face_id.npy
  parser_labels.png
  refined_labels.png
  pose.json
  camera.json
```

No uses PNG RGB de 8 bits para Face IDs cuando el número de caras pueda exceder el rango representable. Prefiere `.npy`, EXR entero o codificación RGB24 documentada.

## Fusión requerida

Implementa un acumulador de puntuación por cara y clase. La lógica base debe ser equivalente a:

```python
score[face_id, label] += (
    parser_confidence
    * sam_confidence
    * visibility_weight
    * normal_view_weight
    * view_reliability
)
```

Requisitos:

- ignorar píxeles sin Face ID;
- resolver oclusiones con depth;
- no asignar una cara desde una máscara que no la ve;
- combinar múltiples vistas;
- usar P3-SAM como prior geométrico;
- guardar confidence por cara cuando sea posible;
- resolver `unassigned` mediante vecinos y priors;
- permitir configurar pesos.

## Inferencia de cuello

El cuello no puede depender de una clase directa del parser.

Derívalo mediante:

- head mask;
- línea de hombros de pose;
- región central bajo la mandíbula;
- límites verticales configurables;
- conectividad y diámetro local del mesh;
- exclusión de torso, cabello y brazos.

Debe existir fallback geométrico cuando falle la pose.

## Exportación semántica

El GLB final debe conservar la geometría y contener una representación útil de las partes:

- materiales o colores de debug por clase;
- opcionalmente objetos separados;
- opcionalmente vertex groups o atributos;
- metadata `extras` con el mapa de clases;
- ruta externa a `face_labels.npz`.

No dupliques vértices de manera innecesaria. Documenta cualquier duplicación causada por límites de material o exportación.

## Workflow ComfyUI

No rompas ni sobrescribas el workflow existente.

Crea un workflow nuevo:

```text
ComfyUI/ComfyUI-LocalAssetFactory/workflows/
05_hunyuan_multiview_segment_and_texture_8gb.json
```

Debe reutilizar la salida de `AssetFactory_BlenderProcessor` y conectar la nueva etapa:

```text
SaveMeshToGLB
→ BlenderProcessor
→ CreateSegmentationProxy
→ P3SAMSegment
→ RenderSemanticViews
→ HumanParseViews
→ PoseSemanticHints
→ SAM2RefineParts
→ FuseSemanticParts
→ WriteSemanticGLB
→ ProjectMultiviewTexture
→ SegmentationPreview
```

También conserva una ruta rápida de solo segmentación que omita texturizado.

## Dependencias

No descargues pesos al repositorio ni los comitees.

Implementa:

- `requirements-segmentation.txt` o extras equivalentes;
- script de instalación verificable;
- rutas configurables para checkpoints;
- descarga explícita y con consentimiento mediante script/documentación;
- comprobación de hashes cuando sea viable;
- auditoría de licencias de P3-SAM, Sonata, SCHP, DWPose y SAM2;
- mensajes claros cuando falte un modelo;
- importaciones lazy para que ComfyUI pueda iniciar aunque no estén instalados los backends opcionales.

Evita conflictos con las versiones ya fijadas de PyTorch, CUDA, NumPy, trimesh y Blender Python.

## Estructura sugerida

Adáptala a la estructura real del repositorio después de inspeccionarlo:

```text
src/local_asset_factory/segmentation/
  __init__.py
  config.py
  labels.py
  contracts.py
  memory.py
  mesh_io.py
  proxy.py
  p3sam_backend.py
  human_parser_backend.py
  pose_backend.py
  sam2_backend.py
  semantic_renderer.py
  fusion.py
  postprocess.py
  export.py
  texture_projection.py

tests/
  segmentation/
```

No fuerces esta estructura si el proyecto tiene una convención mejor; mantén separación equivalente de responsabilidades.

## Pruebas obligatorias

Implementa:

### Unitarias

- mapping de clases del parser;
- convención anatómica L/R;
- cálculo de pesos;
- acumulación por Face ID;
- resolución de conflictos;
- propagación por adyacencia;
- eliminación de islas;
- inferencia de cuello;
- serialización de contratos;
- transferencia proxy→high-poly.

### Integración

- GLB simple con partes conocidas;
- personaje en T-pose;
- personaje con cabello;
- front/back/left/right;
- ausencia de uno de los backends opcionales;
- mesh high-poly con proxy;
- exportación y recarga del GLB;
- ejecución del workflow en low-VRAM.

### Regresión

- el workflow `04_hunyuan_multiview_unified.json` sigue abriendo;
- la geometría existente sigue generándose;
- `AssetFactory_BlenderProcessor` conserva su comportamiento;
- ComfyUI inicia sin instalar todos los modelos opcionales.

## Criterios de aceptación

No marques la tarea como terminada hasta que:

1. el workflow anterior continúe válido;
2. exista el workflow `05`;
3. los nodos nuevos aparezcan registrados en ComfyUI;
4. un GLB limpio pueda segmentarse sin volver a ejecutar Hunyuan;
5. se produzcan las ocho clases;
6. L/R sean anatómicamente correctos en front y back;
7. el preview coloreado permita inspección visual;
8. el GLB exportado pueda reabrirse;
9. el proxy transfiera etiquetas al high-poly;
10. los modelos se descarguen/carguen de manera opcional y lazy;
11. se documenten parámetros para 8 GB;
12. existan pruebas y un comando reproducible para ejecutarlas;
13. se registre pico de VRAM;
14. no se comiteen checkpoints ni outputs grandes;
15. la documentación incluya troubleshooting de OOM.

## Orden de trabajo

1. Inspecciona primero la arquitectura real, registro de nodos, Blender runner, output manager y workflow.
2. Resume brevemente lo encontrado.
3. Implementa contratos, labels, config y memory manager.
4. Implementa el semantic renderer y Face ID.
5. Implementa fusión CPU y exportación usando backends simulados.
6. Añade human parser/pose/SAM2 detrás de adaptadores.
7. Añade P3-SAM y proxy.
8. Añade proyección/bake de texturas.
9. Construye el workflow `05`.
10. Ejecuta pruebas y corrige fallos.
11. Actualiza documentación.
12. Presenta un resumen final exacto.

## Reglas de ejecución para Codex

- No te limites a escribir un plan: implementa el código.
- No reescribas módulos funcionales sin necesidad.
- No cambies nombres públicos existentes salvo que mantengas compatibilidad.
- No uses mocks como resultado final; solo en pruebas.
- No ocultes excepciones.
- No hagas commits de checkpoints o assets pesados.
- No afirmes que una prueba pasó si no se ejecutó.
- Si una dependencia impide una prueba real, implementa la capa completa, añade pruebas con backend falso y documenta exactamente la prueba pendiente.
- Usa decisiones razonables sin interrumpir por detalles menores.
- Conserva un registro claro de archivos modificados.
- Al final entrega:
  - resumen;
  - arquitectura implementada;
  - archivos cambiados;
  - comandos de instalación;
  - comandos de prueba;
  - resultados de pruebas;
  - VRAM medida o motivo por el cual no pudo medirse;
  - limitaciones reales;
  - siguientes pasos estrictamente necesarios.
