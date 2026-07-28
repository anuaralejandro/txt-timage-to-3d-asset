# Arquitectura Completa de Generación 3D: Trellis 2 (VisualBruno 4B) & Blender 4.4

Este documento detalla la arquitectura técnica, representaciones 3D, canalización de inferencia (pipeline), optimización de memoria VRAM (8GB) y el post-procesamiento de topología en Blender para la suite **ComfyUI-Trellis2**.

---

## 1. Visión General del Sistema

Trellis 2 (`microsoft/TRELLIS.2-4B`) es un modelo de difusiones rectificadas (Flow Matching) de última generación diseñado para generar assets 3D de alta calidad a partir de imágenes individuales (Single-View) o múltiples vistas (Multi-View: Frontal, Trasera, Izquierda, Derecha).

El flujo completo se compone de dos grandes subsistemas:

```
[Imágenes Multi-View / Single-View]
                │
                ▼
  [Trellis2 Image-to-3D Pipeline]
  ├── DINOv3 Feature Extractor
  ├── Sparse Structure Flow (Estructura Dispersa)
  ├── Hole Filling (flood_fill)
  ├── Shape SLat Flow (LR & HR)
  └── Texture SLat Flow (PBR Textures)
                │
         (MeshWithVoxel Raw)
                │
                ▼
  [O-Voxel Decoder & Trimesh Export]
                │
       (Raw Mesh: 1.7M - 4.2M Tris)
                │
                ▼
  [Blender 4.4 Headless Retopology & UV]
  ├── Manifold Voxel Remesh
  ├── C++ Decimation (Exactamente 30,000 Tris)
  ├── Laplacian & Smooth Modifiers
  ├── Weighted Normals & Shade Smooth
  └── Smart UV Unwrapping
                │
                ▼
  [Asset Final GLB (30k Polígonos - AAA Ready)]
```

---

## 2. Componentes de la Arquitectura Trellis 2

### 2.1 Representaciones 3D Internas
* **`SparseTensor`:** Representación tensorial dispersa optimizada mediante `spconv` y `sdpa` (Scaled Dot-Product Attention).
* **`Shape SLat` (Structured Latents):** Latentes estructurados que codifican la geometría tridimensional en dos niveles de resolución (LR 512 y HR 1024/1536 cascade).
* **`Texture SLat`:** Latentes que codifican la información cromática PBR (Base Color y Alpha).
* **`MeshWithVoxel`:** Representación intermedia de voxel y malla derivada de FlexiCubes / Marching Cubes.

### 2.2 Nodos de ComfyUI (`custom_nodes/ComfyUI-Trellis2/nodes.py`)
1. **`Trellis2LoadModel`:** 
   - Carga diferida (*lazy loading*) de los pesos en formato `.safetensors`.
   - Configurado en `sdpa` para atención y `spconv` para convoluciones dispersas.
   - Incluye colectores de basura explícitos (`gc.collect()` y `torch.cuda.empty_cache()`) en `trellis2/models/__init__.py` para evitar errores de archivo de paginación en Windows (Error OS 1455).

2. **`Trellis2MeshWithVoxelMultiViewGenerator`:**
   - Procesa simultáneamente hasta 4 vistas RGBA (`front`, `back`, `left`, `right`).
   - Fusión de características visuales mediante **DINOv3**.
   - Integra la función `flood_fill` para rellenar de forma iterativa los orificios topológicos (*bullet holes*) en la estructura dispersa antes de la decodificación de forma.

3. **`Trellis2OvoxelExportToGLB`:**
   - Convierte el objeto `MeshWithVoxel` en un objeto de malla `trimesh.Trimesh`.
   - Incorpora búsqueda binaria GPU acelerada con `torch.searchsorted()` en `o_voxel/postprocess.py` para mapear colores de vértices directamente en memoria VRAM cuando `cumesh` nativo no está disponible.

---

## 3. Estrategia de Optimización VRAM para GPUs de 8GB

Para ejecutar un modelo gigante de 4B parámetros en una tarjeta de video de 8GB VRAM (como la NVIDIA GeForce RTX 4070 Laptop GPU), se aplican las siguientes reglas:

1. **Liberación de Memoria entre Fases (`keep_models_loaded=False`):**
   - El sistema descarga de VRAM el modelo de estructura dispersa antes de cargar el decodificador de forma (Shape), y descarga este último antes de cargar el decodificador de textura.
2. **Resolución de Inferencia Equilibrada (`1024_cascade` & 25 Pasos):**
   - Muestreo con sampler `euler` a 25 pasos, garantizando un equilibrio perfecto entre calidad AAA y tiempos de ejecución reducidos (~5 minutos).
3. **Resolución Dispersa Controlada (`32`):**
   - Mantiene la grilla de voxels en 32³ para limitar la memoria requerida durante la extracción de FlexiCubes.

---

## 4. Post-procesamiento y Retopología en Blender 4.4 (`scripts/blender_process_30k.py`)

Las mallas crudas producidas por IA contienen millones de polígonos desarticulados y ruidosos (4.2M+ triángulos). El script de Blender ejecuta las siguientes 5 etapas automatizadas:

1. **Voxel Remesh Hermético (Manifold Shell):** Reconstruye la malla completa con un tamaño de voxel dinámico (`max_dim / 180.0`), unificando todas las superficies sueltas en una sola carcasa cerrada.
2. **Decimación Controlada C++:** Reduce la densidad poligonal desde ~70,000 voxeles hasta exactamente **30,000 triángulos**.
3. **Suavizado de Superficie (Laplacian & Smooth Filters):** Elimina el aspecto pixelado/rugoso propio de las mallas Voxel, dejando superficies suaves y orgánicas.
4. **Optimización de Sombreado (Shade Smooth & Weighted Normals):** Aplica normales ponderadas (`weight=50`) y auto-smooth a 60° para un sombreado perfecto sin distorsión visual.
5. **Smart UV Unwrapping:** Genera coordenadas UV con márgenes de isla (`island_margin=0.004`) listas para la aplicación de texturas PBR.

---

## 5. Scripts Principales

* `scripts/test_trellis2_visualbruno.py`: Test unitario E2E Single-View.
* `scripts/test_trellis2_multiview_balanced.py`: Test E2E Multi-View optimizado para 8GB VRAM.
* `scripts/blender_process_30k.py`: Ejecutor headless de Blender 4.4 para retopología a 30k polígonos.
