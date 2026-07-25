# Checklist de implementación — Segmentación y texturas 3D en ComfyUI

## 1. Preparación

- [ ] Confirmar rama base `feat/hunyuan-multiview-character-pipeline`.
- [ ] Crear rama de trabajo específica.
- [ ] Ejecutar o validar el workflow `04_hunyuan_multiview_unified.json`.
- [ ] Identificar el path exacto de salida de `AssetFactory_BlenderProcessor`.
- [ ] Inspeccionar registro actual de nodos.
- [ ] Inspeccionar Blender runner.
- [ ] Inspeccionar output manager.
- [ ] Inspeccionar manejo de requirements.
- [ ] Inspeccionar tests existentes.
- [ ] Documentar supuestos y convenciones del repositorio.

## 2. Compatibilidad

- [ ] No romper el workflow `04`.
- [ ] No cambiar firmas públicas existentes sin compatibilidad.
- [ ] Evitar imports pesados durante el startup de ComfyUI.
- [ ] Permitir iniciar ComfyUI sin modelos opcionales.
- [ ] Soportar Windows.
- [ ] Mantener compatibilidad razonable con Linux.
- [ ] No comitear checkpoints.
- [ ] No comitear outputs grandes.
- [ ] Añadir entradas relevantes a `.gitignore`.

## 3. Taxonomía

- [ ] Definir `unassigned = 0`.
- [ ] Definir `torso = 1`.
- [ ] Definir `neck = 2`.
- [ ] Definir `head = 3`.
- [ ] Definir `hair = 4`.
- [ ] Definir `arm_L = 5`.
- [ ] Definir `arm_R = 6`.
- [ ] Definir `leg_L = 7`.
- [ ] Definir `leg_R = 8`.
- [ ] Documentar que L/R es lateralidad anatómica.
- [ ] Añadir tests de lateralidad frontal.
- [ ] Añadir tests de lateralidad posterior.

## 4. Contratos y configuración

- [ ] Crear configuración low-VRAM.
- [ ] Crear esquema de manifest semántico.
- [ ] Crear dataclasses/Pydantic models.
- [ ] Versionar schemas.
- [ ] Validar rutas.
- [ ] Validar labels.
- [ ] Serializar settings.
- [ ] Registrar modelos usados.
- [ ] Registrar warnings.
- [ ] Registrar counts por label.
- [ ] Registrar confidencias.
- [ ] Añadir tests de round-trip JSON.

## 5. Memory lifecycle

- [ ] Implementar carga lazy.
- [ ] Implementar descarga explícita.
- [ ] Usar `torch.inference_mode()`.
- [ ] Usar FP16 donde sea seguro.
- [ ] Procesar batch 1.
- [ ] Procesar una vista a la vez.
- [ ] Mover modelo a CPU antes de destruir.
- [ ] Ejecutar `del`.
- [ ] Ejecutar `gc.collect()`.
- [ ] Ejecutar `torch.cuda.empty_cache()`.
- [ ] Resetear peak memory por fase.
- [ ] Registrar peak allocated.
- [ ] Registrar reserved memory.
- [ ] Fallar con mensajes accionables ante OOM.
- [ ] Añadir fallback o bypass por backend.

## 6. Proxy mesh

- [ ] Detectar triangle count.
- [ ] Decimar solo sobre el umbral.
- [ ] Default threshold 250k tris.
- [ ] Default proxy 150k tris.
- [ ] Preservar transformaciones.
- [ ] Triangular de forma consistente.
- [ ] Conservar GLB original.
- [ ] Guardar proxy separado.
- [ ] Crear BVH.
- [ ] Implementar proxy→high-poly.
- [ ] Validar cobertura de todas las caras.
- [ ] Registrar distancia de transferencia.
- [ ] Añadir fixture high-poly.
- [ ] Añadir tests de transferencia.

## 7. Blender semantic renderer

- [ ] Reutilizar Blender runner existente.
- [ ] Crear script headless.
- [ ] Cargar GLB limpio.
- [ ] Aplicar transforms de forma controlada.
- [ ] Crear cámara ortográfica.
- [ ] Encuadrar automáticamente.
- [ ] Render front.
- [ ] Render front-left.
- [ ] Render left.
- [ ] Render back-left.
- [ ] Render back.
- [ ] Render back-right.
- [ ] Render right.
- [ ] Render front-right.
- [ ] Guardar RGB.
- [ ] Guardar depth lineal.
- [ ] Guardar world normals.
- [ ] Guardar Face ID.
- [ ] Guardar camera intrinsics.
- [ ] Guardar camera extrinsics.
- [ ] Desactivar antialiasing de Face ID.
- [ ] Soportar IDs de meshes grandes.
- [ ] Validar píxel→face.
- [ ] Añadir test de oclusión.

## 8. Human parser

- [ ] Crear interfaz de backend.
- [ ] Implementar SCHP-LIP.
- [ ] Carga lazy.
- [ ] Configurar checkpoint path.
- [ ] Mapear hair.
- [ ] Mapear head/face.
- [ ] Mapear torso/clothing.
- [ ] Mapear left arm.
- [ ] Mapear right arm.
- [ ] Mapear left leg.
- [ ] Mapear right leg.
- [ ] Conservar confidence maps.
- [ ] Guardar parser labels por vista.
- [ ] Crear overlays debug.
- [ ] Añadir fallback sin parser.
- [ ] Testear mapping.

## 9. DWPose/OpenPose

- [ ] Crear interfaz de backend.
- [ ] Detectar hombros L/R.
- [ ] Detectar codos L/R.
- [ ] Detectar muñecas L/R.
- [ ] Detectar caderas L/R.
- [ ] Detectar rodillas L/R.
- [ ] Detectar tobillos L/R.
- [ ] Detectar pistas de cabeza/cuello.
- [ ] Guardar confidence.
- [ ] Adaptar front/back/side.
- [ ] Generar positive prompts.
- [ ] Generar negative prompts.
- [ ] Guardar `pose.json`.
- [ ] Añadir fallback geométrico.
- [ ] Testear no inversión L/R.

## 10. SAM2

- [ ] Crear interfaz de backend.
- [ ] Integrar SAM2.1 Hiera Small.
- [ ] Default FP16.
- [ ] Cargar una sola vez por fase.
- [ ] Procesar una región por vez si es necesario.
- [ ] Usar mask/box del parser.
- [ ] Usar puntos de pose.
- [ ] No usar SAM2 como clasificador.
- [ ] Guardar máscaras refinadas.
- [ ] Guardar confidence.
- [ ] Guardar overlays.
- [ ] Liberar embeddings.
- [ ] Testear bypass de SAM2.
- [ ] Testear missing checkpoint.

## 11. P3-SAM

- [ ] Crear backend opcional.
- [ ] Integrar Sonata.
- [ ] Integrar checkpoint P3-SAM.
- [ ] Configurar paths.
- [ ] Surface sample del mesh.
- [ ] Default 50k points.
- [ ] Seed reproducible.
- [ ] Mapear point masks a caras.
- [ ] Guardar regiones.
- [ ] Guardar IoU/confidence.
- [ ] No asignar anatomía directamente.
- [ ] Liberar P3-SAM/Sonata antes de 2D models.
- [ ] Añadir bypass.
- [ ] Añadir test con backend fake.
- [ ] Documentar prueba real pendiente si no hay pesos.

## 12. Fusión 2D→3D

- [ ] Crear `face_scores[F, 9]`.
- [ ] Crear observation counts.
- [ ] Leer Face ID.
- [ ] Ignorar background.
- [ ] Validar depth.
- [ ] Calcular normal-view weight.
- [ ] Aplicar confidence parser.
- [ ] Aplicar confidence SAM.
- [ ] Aplicar view reliability.
- [ ] Aplicar pose prior.
- [ ] Aplicar P3-SAM prior.
- [ ] Acumular por vista.
- [ ] Resolver argmax.
- [ ] Aplicar confidence threshold.
- [ ] Aplicar margin entre top-1/top-2.
- [ ] Guardar confidence por cara.
- [ ] Añadir tests sintéticos.

## 13. Postprocesamiento

- [ ] Construir face adjacency.
- [ ] Majority smoothing configurable.
- [ ] Encontrar connected components.
- [ ] Eliminar islas pequeñas.
- [ ] Rellenar huecos.
- [ ] Propagar `unassigned`.
- [ ] Mantener componente principal de arm_L.
- [ ] Mantener componente principal de arm_R.
- [ ] Mantener componente principal de leg_L.
- [ ] Mantener componente principal de leg_R.
- [ ] Aplicar restricciones L/R.
- [ ] Respetar regiones P3-SAM.
- [ ] Evitar sobre-suavizado.
- [ ] Guardar métricas antes/después.
- [ ] Añadir tests de islas y huecos.

## 14. Cuello

- [ ] Derivar línea de hombros.
- [ ] Derivar límite inferior de head.
- [ ] Crear banda candidata.
- [ ] Restringir al centro corporal.
- [ ] Excluir hair.
- [ ] Excluir arms.
- [ ] Excluir torso inferior.
- [ ] Aplicar conectividad.
- [ ] Implementar fallback por bounding box.
- [ ] Evitar clasificar pecho como cuello.
- [ ] Añadir test con pose.
- [ ] Añadir test sin pose.

## 15. Exportación

- [ ] Guardar `face_labels.npz`.
- [ ] Guardar `semantic_parts.json`.
- [ ] Guardar GLB coloreado.
- [ ] Guardar GLB semántico.
- [ ] Añadir materials por label para debug.
- [ ] Añadir metadata `extras`.
- [ ] Conservar UV existentes.
- [ ] Conservar transformaciones.
- [ ] Validar recarga Blender.
- [ ] Validar recarga trimesh.
- [ ] Validar face count.
- [ ] Documentar duplicación de vértices si ocurre.

## 16. Texturizado multivista

- [ ] Recibir front image.
- [ ] Recibir left image.
- [ ] Recibir right image.
- [ ] Recibir back image.
- [ ] Crear/reutilizar UV.
- [ ] Configurar bake 1024.
- [ ] Añadir opción 512.
- [ ] Añadir opción 2048.
- [ ] Proyectar por cámara.
- [ ] Validar visibilidad.
- [ ] Validar depth.
- [ ] Ponderar normal angle.
- [ ] Ponderar confidence.
- [ ] Ponderar distancia a borde.
- [ ] Bloquear contaminación entre labels.
- [ ] Bloquear arm_L↔arm_R.
- [ ] Bloquear leg_L↔leg_R.
- [ ] Bloquear hair→head.
- [ ] Bloquear torso→arms.
- [ ] Mezclar vistas.
- [ ] Hornear albedo.
- [ ] Generar coverage map.
- [ ] Generar confidence map.
- [ ] Dilatar UV 8–16 px.
- [ ] Detectar texels sin observación.
- [ ] Crear hook opcional de inpainting.
- [ ] Exportar GLB texturizado.
- [ ] Añadir prueba sintética de bleeding.

## 17. Custom nodes

- [ ] `AssetFactory_CreateSegmentationProxy`.
- [ ] `AssetFactory_P3SAMSegment`.
- [ ] `AssetFactory_RenderSemanticViews`.
- [ ] `AssetFactory_HumanParseViews`.
- [ ] `AssetFactory_PoseSemanticHints`.
- [ ] `AssetFactory_SAM2RefineParts`.
- [ ] `AssetFactory_FuseSemanticParts`.
- [ ] `AssetFactory_WriteSemanticGLB`.
- [ ] `AssetFactory_ProjectMultiviewTexture`.
- [ ] `AssetFactory_SegmentationPreview`.
- [ ] Registrar node mappings.
- [ ] Registrar display names.
- [ ] Añadir categorías coherentes.
- [ ] Validar inputs.
- [ ] Validar outputs.
- [ ] Añadir status.
- [ ] Añadir manejo de errores.
- [ ] Añadir docstrings.

## 18. Workflow `05`

- [ ] Crear `05_hunyuan_multiview_segment_and_texture_8gb.json`.
- [ ] Mantener workflow `04` intacto.
- [ ] Conectar salida BlenderProcessor.
- [ ] Conectar proxy.
- [ ] Conectar P3-SAM.
- [ ] Conectar renderer.
- [ ] Conectar parser.
- [ ] Conectar pose.
- [ ] Conectar SAM2.
- [ ] Conectar fusion.
- [ ] Conectar export.
- [ ] Conectar texture projection.
- [ ] Conectar previews.
- [ ] Añadir bypass P3-SAM.
- [ ] Añadir bypass SAM2.
- [ ] Añadir bypass texture.
- [ ] Añadir ruta solo segmentación.
- [ ] Organizar grupos/notas.
- [ ] Usar defaults de 8 GB.
- [ ] Abrir y validar JSON en ComfyUI.

## 19. Dependencias

- [ ] Crear extras/requirements de segmentación.
- [ ] No alterar PyTorch sin necesidad.
- [ ] Evitar conflicto NumPy.
- [ ] Evitar conflicto trimesh.
- [ ] Evitar conflicto Blender Python.
- [ ] Crear instalador Windows.
- [ ] Crear instalador Linux.
- [ ] Documentar checkpoints.
- [ ] Documentar directorios.
- [ ] Verificar licencias.
- [ ] Añadir checksum cuando sea viable.
- [ ] Añadir mensajes de missing dependency.
- [ ] Añadir mensajes de missing weights.

## 20. Tests

- [ ] Unit tests de labels.
- [ ] Unit tests de L/R.
- [ ] Unit tests de Face ID.
- [ ] Unit tests de camera transforms.
- [ ] Unit tests de fusion.
- [ ] Unit tests de adjacency.
- [ ] Unit tests de neck.
- [ ] Unit tests de proxy transfer.
- [ ] Unit tests de manifests.
- [ ] Integration test con fake backends.
- [ ] Integration test export/reload.
- [ ] Integration test high-poly proxy.
- [ ] Smoke test desde GLB existente.
- [ ] Regression test workflow `04`.
- [ ] Startup test sin extras.
- [ ] Low-VRAM execution test.
- [ ] Texture bleeding test.
- [ ] Ejecutar test suite.
- [ ] Registrar resultados reales.

## 21. Documentación

- [ ] Arquitectura.
- [ ] Instalación.
- [ ] Modelos/checkpoints.
- [ ] Licencias.
- [ ] Workflow.
- [ ] Configuración 8 GB.
- [ ] Ejemplo solo segmentación.
- [ ] Ejemplo segmentación + textura.
- [ ] Troubleshooting OOM.
- [ ] Troubleshooting Blender.
- [ ] Troubleshooting missing nodes.
- [ ] Debug de máscaras.
- [ ] Outputs generados.
- [ ] Limitaciones conocidas.

## 22. Validación final

- [ ] El workflow `04` sigue funcionando.
- [ ] El workflow `05` abre sin nodos desconocidos.
- [ ] El pipeline acepta un GLB ya existente.
- [ ] Se generan labels `0..8`.
- [ ] 100 % de caras tienen label válido.
- [ ] L/R es correcto en front.
- [ ] L/R es correcto en back.
- [ ] Hair y head están separados en fixture adecuado.
- [ ] Neck no invade el pecho de forma significativa.
- [ ] El preview coloreado se ve correctamente.
- [ ] El GLB recarga.
- [ ] El proxy transfiere labels.
- [ ] Texturas no cruzan partes en test sintético.
- [ ] Se genera coverage map.
- [ ] Se registra pico de VRAM.
- [ ] No hubo modelos grandes concurrentes.
- [ ] No hay checkpoints en git.
- [ ] No hay outputs grandes en git.
- [ ] Tests pasan o las limitaciones se documentan con precisión.

## 23. Entrega de Codex

- [ ] Resumen de implementación.
- [ ] Lista de archivos creados.
- [ ] Lista de archivos modificados.
- [ ] Comandos de instalación.
- [ ] Comandos de ejecución.
- [ ] Comandos de tests.
- [ ] Resultados reales de tests.
- [ ] Medición real de VRAM o limitación.
- [ ] Riesgos pendientes.
- [ ] Limitaciones reales.
- [ ] Próximos pasos indispensables.
