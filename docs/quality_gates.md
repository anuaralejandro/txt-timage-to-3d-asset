# Quality Gates y Métricas de Selección

Este documento define los umbrales (hard gates) que todo candidato geométrico (malla raw) debe superar para no ser descartado por el pipeline.

## 1. Hard Gates Geométricos
Si un candidato falla en cualquiera de estos criterios, es rechazado automáticamente (`rejected`), sin importar su puntuación en otras áreas.

*   **Silhouette IoU (Intersection over Union)**: $\ge 0.70$. La silueta renderizada del modelo 3D desde las cámaras canónicas debe coincidir al menos en un 70% con las imágenes de entrada limpias.
*   **Integridad Anatómica**:
    *   `has_head == 1`: El modelo debe tener una cabeza claramente definida.
    *   `arms_count == 2`: Se deben detectar exactamente dos brazos separados del torso (T-pose/A-pose clara).
    *   `legs_count == 2`: Se deben detectar exactamente dos piernas separadas.
*   **Topología Base**:
    *   `connected_components \le 5`: Idealmente 1 para el cuerpo, pero se tolera hasta 5 para permitir mallas desconectadas como ojos, cabello o accesorios menores antes de la retopología.
    *   `severe_cavities == 0`: No debe haber huecos catastróficos en el volumen (por ejemplo, el torso hueco).

## 2. Flujo de Selección
1.  Se generan múltiples candidatos variando parámetros (seeds, threshold, octree).
2.  Todos los candidatos pasan a estado `generated_unscored`.
3.  Se renderizan desde 7 vistas canónicas (`front, left, back, right, front_3q, back_3q, top`).
4.  Se evalúan mediante las métricas.
5.  Los que no superan los Hard Gates pasan a `rejected`.
6.  Si todos fallan, el job completo se marca como `failed_no_valid_candidate` (nunca se selecciona "el menos malo").
7.  Si varios pasan, se selecciona el que tenga mayor `Silhouette IoU` y menor `surface_noise`. Ese candidato pasa a estado `selected_high_mesh`.
