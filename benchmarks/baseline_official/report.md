# Baseline Oficial Hunyuan3D-2mv

Este documento registra los resultados de la verificación y preparación de ambos pipelines de Hunyuan3D-2mv (Turbo y Normal).

## Estado
- [x] Instalación e integración de Nodos en ComfyUI (`scripts/install_comfy_nodes.py`).
- [x] Verificación de suite de Pruebas Unitarias (`tests/test_e2e_pipeline.py`).
- [x] Configuración de Pipeline 1 (Turbo): `configs/models/hunyuan3d_2mv_turbo.yaml` (30 steps, `euler_ancestral`, guidance 1.5).
- [x] Configuración de Pipeline 2 (Normal): `configs/models/hunyuan3d_2mv_normal.yaml` (50 steps, `euler`, guidance 5.5).

## Resumen de Verificación
Ambos pipelines cuentan con sus esquemas YAML de configuración aislados, reglas de Scoring/Hard-Gates activas, normalización de vistas canónicas y soporte para segmentación de ropa/accesorios para personajes.

