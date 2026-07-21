# Auditoría del Workflow (Fase 3)

**Workflow auditado**: `04_hunyuan_multiview_unified.json`

## 1. Análisis de Nodos y Estructura

El workflow actual no expone la arquitectura interna de Hunyuan3D-2 de manera nativa en el grafo visual de ComfyUI. En su lugar, utiliza un nodo "Macro" que encapsula toda la lógica en el backend (Python).

### Tabla de Nodos Actuales

| Nodo | Clase | Inputs | Widgets | Output | Riesgo |
|---|---|---|---|---|---|
| 1, 2, 3, 14 | `LoadImage` | Ninguno | `image` | `IMAGE`, `MASK` | Bajo |
| 16 | `AssetFactory_LoadMultiviewDirectory` | Ninguno | `directory_path` | 4x `IMAGE` (Front, Left, Right, Back) | Bajo |
| 17 | `AssetFactory_RemoveMultiviewFakeBackground` | 4x `IMAGE` | `enabled` | 4x `IMAGE`, `STATUS` | Medio (limpieza básica) |
| 4 | `AssetFactory_HunyuanMacroPipeline` | 4x `IMAGE` | `category`, `budget`, `platform`, `style`, `seed` | `FINAL_GLB_PATH`, `PIPELINE_STATUS` | **Alto (Caja negra opaca)** |
| 5 | `AssetFactory_Preview3D` | `model_path` | Ninguno | UI 3D Viewer | Bajo |
| Varios | `PreviewImage` | `images` | Ninguno | Ninguno | Bajo |

## 2. Parámetros Ausentes (Ocultos en la Caja Negra)

El objetivo de ComfyUI es la reproducibilidad visual. El workflow auditado incumple esto porque oculta los siguientes parámetros críticos que deberían estar expuestos en el JSON:

- **Checkpoint Loader**: Ausente en el frontend.
- **VAE / Image Encoder**: Ausente en el frontend.
- **Conditioning Node**: Ausente en el frontend.
- **Latent Initialization**: Ausente.
- **Sampler / Scheduler**: Ausente (oculto en el backend python).
- **Steps, CFG, Guidance**: Ausente (oculto).
- **Dtype / Device**: Ausente.
- **Extracción de Malla**: `octree_resolution`, `num_chunks`, `mesh_algorithm`, `mesh_threshold` ocultos.
- **Post-procesamiento**: decimation, weld, smoothing, no son visibles.

## 3. Conclusión de la Auditoría

El workflow `04_hunyuan_multiview_unified.json` **no es un verdadero workflow de ComfyUI**, sino una interfaz gráfica simple que delega toda la inferencia y procesamiento a un script Python (`services/hunyuan_adapter.py` y `nodes_macro.py`). Esto impide experimentar, separar etapas (normal vs turbo) o depurar fallos en la generación de la malla, siendo esta la causa principal de la dificultad para corregir la geometría severamente defectuosa descrita en el prompt.
