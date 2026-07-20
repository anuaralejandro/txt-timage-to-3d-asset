# Text & Image to 3D Asset Generation Platform (`txt-timage-to-3d-asset`)

Sistema local de generación de assets 3D optimizados para videojuegos utilizando ComfyUI, Hunyuan3D-2, Ollama y Blender.

[![GitHub Repository](https://img.shields.io/badge/GitHub-txt--timage--to--3d--asset-blue?logo=github)](https://github.com/anuaralejandro/txt-timage-to-3d-asset.git)

---

## 🚀 Características Principales

- **100% Local-First**: Sin necesidad de servicios en la nube ni costos por API externa.
- **Arquitectura Basada en Nodos ComfyUI**: Pipeline modular mediante el paquete custom node `ComfyUI-LocalAssetFactory`.
- **Integración con Ollama**: Generación automática de especificaciones técnicas y prompts optimizados a partir de breves descripciones en español.
- **Hunyuan3D-2 Multi-View / Single-View**: Generación de mallas 3D de alta calidad a partir de conceptos 2D o imágenes en Pose-T (front, left, back, right).
- **Procesamiento Automatizado en Blender**: Optimización de presupuesto de polígonos (decimación), asignación de texturas, UV unwrapping y exportación de archivos `.glb`.
- **Previsualizador 3D Integrado**: Canvas WebGL interactivo directo dentro de la interfaz de ComfyUI.
- **Generación de Reporte Manifest**: Salida con metadatos completos `manifest.json`.

---

## 📁 Estructura del Repositorio

```
txt-timage-to-3d-asset/
├── .agents/
│   └── AGENTS.md                  # Reglas de proyecto y mapa de estado ("Graphify")
├── ComfyUI_windows_portable/
│   └── ComfyUI/
│       ├── comfy_extras/
│       │   └── nodes_hunyuan3d.py # Extensión core de Hunyuan3D-2
│       └── custom_nodes/
│           └── ComfyUI-LocalAssetFactory/
│               ├── blender/       # Scripts headless para Blender
│               ├── services/      # Ollama, ComfyBridge, HunyuanAdapter, BlenderRunner
│               ├── utilities/     # Conversión de tensores/imágenes y logging
│               ├── workflows/     # Workflows JSON exportables
│               ├── nodes.py       # Definición de los 7-8 Nodos Custom
│               ├── config.py      # Configuración de rutas y flags
│               └── schemas.py     # Esquemas Pydantic
├── DEBUGGING_GUIDE.md             # Guía paso a paso para depurar el sistema con ChatGPT
├── install_comfy.py               # Script de instalación y arranque
├── check_url.py                   # Script de verificación de endpoints
└── .gitignore                     # Filtros para excluir binarios y modelos pesados
```

---

## 🛠️ Instalación y Uso

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/anuaralejandro/txt-timage-to-3d-asset.git
   cd txt-timage-to-3d-asset
   ```

2. **Ejecutar Script de Instalación**:
   ```bash
   python install_comfy.py
   ```

3. **Arrancar ComfyUI**:
   Ejecutar `run_nvidia_gpu.bat` dentro de `ComfyUI_windows_portable/`.

4. **Cargar Workflows**:
   Abre la interfaz web de ComfyUI (`http://127.0.0.1:8188`) y carga el workflow desde `ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-LocalAssetFactory/workflows/full_text_to_asset_v2.json`.

---

## 🔍 Guía de Depuración (ChatGPT / AI Debugging)

Para entender en detalle la arquitectura de nodos, los esquemas de datos JSON y la forma de resolver errores comunes, consulta el archivo [DEBUGGING_GUIDE.md](file:///c:/Users/datam/Videos/ComftyUI-text-2-3d-asset-gen/DEBUGGING_GUIDE.md).
