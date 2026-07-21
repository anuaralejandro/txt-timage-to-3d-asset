# Reglas de Proyecto: Optimización de Contexto y Estado del Sistema

Este archivo contiene las directrices principales y el mapa de conocimiento ("Graphify") del sistema para optimizar el uso de tokens, mantener el contexto centrado y permitir debuggear el sistema desde cualquier IA / ChatGPT.

---

## 1. Mapa de Arquitectura y Grafo del Sistema (Knowledge Graph)

```
[Usuario / Client]
       │
       ▼
[Node 1: AssetBriefNode] ── (JSON Brief) ──► [Node 2: OllamaPromptArchitectNode]
                                                         │
                                               (Prompts & Spec JSON)
                                                         │
                        ┌────────────────────────────────┴────────────────────────────────┐
                        ▼                                                                 ▼
           [Node 3: LocalConceptImageNode]                                   [Node 4: LocalTextureGeneratorNode]
           (ComfyBridge ➔ Stable Diffusion)                                 (ComfyBridge ➔ Texture Prompts)
                        │                                                                 │
                 (Concept Image)                                                          │
                        │                                                                 │
           [Node X: Image Processing] ◄── [Node Y: LoadMultiviewDirectoryNode]            │
           (RemoveFakeBackgroundNode,             (Load Local Views)                      │
           RemoveMultiviewFakeBackgroundNode,                                             │
           ApplyManualMaskNode)                                                           │
                        │                                                                 │
                        ▼                                                                 │
           [Node 5: LocalHunyuanNode] ◄───────────────────────────────────────────────────┘
           (Hunyuan3D-2 Multi-View / 3D Gen)
                        │
                  (Raw GLB/OBJ)
                        │
                        ▼
           [Node 6: BlenderProcessorNode] ── (Headless Blender Execution)
                        │                  (Decimation, UVs, Materials, GLB Export)
                  (Processed GLB)
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
[Node 7: SaveManifestNode]    [Node 8: Preview3DNode]
(manifest.json Report)        (ComfyUI 3D Viewer Canvas)

[Node 9: HunyuanMacroPipelineNode] ── (Orquestador All-in-One de todo el proceso)
```

---

## 2. Componentes Principales del Sistema

1. **Custom Node Suite (`comfyui/ComfyUI-LocalAssetFactory/`)**:
   - `nodes.py`: Registra los 7/8 nodos interactivos de ComfyUI.
   - `schemas.py`: Modelos Pydantic (`AssetRequest`, `AssetSpecification`, `TexturePromptSet`).
   - `config.py`: Variables de entorno y configuración general de la pipeline.
   - `services/`:
     - `ollama_client.py`: Cliente HTTP local para generación de JSON specs con Ollama.
     - `comfy_bridge.py`: Conector API local con la instancia de ComfyUI.
     - `texture_generator.py`: Generador de mapas de textura.
     - `hunyuan_adapter.py`: Orquestador de inferencia Hunyuan3D-2 (multi-vista y single-image).
     - `blender_runner.py`: Ejecutor de subproceso headless de Blender.
     - `manifest_writer.py`: Generador del reporte final `manifest.json`.
     - `output_manager.py`: Administrador de estructura de carpetas de salida.
   - `blender/`: Scripts Python independientes para Blender (`process_mesh.py`, `render_glb_previews.py`).
   - `workflows/`: Workflows visuales JSON (ahora extraídos en la raíz `workflows/`).

2. **Core ComfyUI Extensible (`ComfyUI_windows_portable/ComfyUI/comfy_extras/nodes_hunyuan3d.py`)**:
   - Nodos nativos de Hunyuan3D-2 (`EmptyLatentHunyuan3Dv2`, `Hunyuan3Dv2ConditioningMultiView`, `VAEDecodeHunyuan3D`, `voxel_to_mesh`).

3. **Scripts de Raíz**:
   - `install_comfy.py`: Script de inicialización e instalación portátil.
   - `check_url.py`: Verificación de endpoints locales y servicios.

---

## 3. Reglas de Exploración de Directorios y Contexto

- **Evitar listar la raíz completa o directorios pesados**: NO ejecutes comandos como `tree` desde la raíz.
- **Directorios a IGNORAR categóricamente en Git & Contexto**:
  - `ComfyUI_windows_portable/python_embeded/` (Entorno Python)
  - `ComfyUI_windows_portable/ComfyUI/models/` (Modelos muy pesados binarios)
  - `ComfyUI_windows_portable/ComfyUI/output/` y `input/` (Assets de usuario)
  - `__pycache__`/ y `.git/`

---

## 4. Enfoque de Desarrollo y Debugging

- **Lectura de Código**: Para depurar o revisar nodos, inspecciona directamente `nodes.py`, `schemas.py` o los submódulos en `services/`.
- **Modificación del Core**: Al modificar archivos core de ComfyUI (ej: `comfy_extras/nodes_hunyuan3d.py`), realiza cambios puntuales y documentados sin romper el contrato base.
- **Workflows JSON**: Ubicados en `workflows/`. Evita editarlos a mano a menos que sea estrictamente necesario.

---

## 5. Optimización de Tokens con RTK (Rust Token Killer)

Para maximizar la eficiencia y reducir el consumo de tokens en comandos en línea y exploración del repositorio:
- **Usa RTK**: Para cualquier comando en línea (ej. `git diff`, `npm install`, comandos de construcción, etc.) que genere salidas verbosas.
- **Evita comandos inline ruidosos sin filtro**: Apóyate en RTK para podar los tokens antes de que la IA los procese. Ejecútalo como prefijo: `rtk <comando>` (ej: `rtk git status`).
- **Propósito**: RTK intercepta salidas, agrupa datos y elimina ruido (como comentarios repetitivos o espacios), evitando llenar la ventana de contexto.
- **Disponibilidad**: El repositorio de RTK ha sido descargado en este entorno y debe instalarse vía `cargo install --path .` (o usar la versión global si ya está configurado).
