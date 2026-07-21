# Inventario Real del Repositorio (Current State)

## 1. Estado de la Rama y Submódulos
- **Rama Actual**: `feat/hunyuan-multiview-character-pipeline` (3 commits ahead of origin).
- **Repositorios Anidados**: Se detectó que `ComfyUI_windows_portable/ComfyUI` está configurado como un modo `160000` (gitlink), pero **falta la entrada correspondiente en `.gitmodules`**. Esto produce el error `fatal: no submodule mapping found in .gitmodules`.
- **Archivos No Versionados**: Hay modificaciones locales (`modified content, untracked content`) atrapadas dentro del directorio de ComfyUI. Esto significa que **actualmente el repositorio no es reproducible desde un clon limpio**.

## 2. Estado del Workflow
- **Archivo**: `04_hunyuan_multiview_unified.json`
- **Ubicación Actual**: `ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/workflows/` (Atrapado en el submódulo roto).
- **Nodos Identificados en el JSON**:
  - `AssetFactory_HunyuanMacroPipeline` (Nodo 4)
  - `AssetFactory_Preview3D` (Nodo 5)
  - `AssetFactory_LoadMultiviewDirectory` (Nodo 16)
  - `AssetFactory_RemoveMultiviewFakeBackground` (Nodo 17)
  - Nodos nativos: `LoadImage`, `PreviewImage`

## 3. Conclusión de la Fase 1
El flujo de trabajo actual **no puede leerse ni ejecutarse desde un clon limpio** porque el código fuente de los custom nodes (`ComfyUI-LocalAssetFactory`) y los workflows (`workflows/`) residen dentro de una carpeta ignorada o mal enlazada como gitlink sin configuración de submódulo. Se requiere extraer este código al nivel raíz y generar el instalador en la Fase 4.
