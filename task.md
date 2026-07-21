# Task backlog — Corrección integral del pipeline Hunyuan3D multivista

Convención:

```text
P0 = bloqueante
P1 = alta prioridad
P2 = producción
P3 = mejora
```

Estados:

```text
TODO
IN_PROGRESS
BLOCKED
DONE
```

---

# EPIC HY-00 — Recuperación y reproducibilidad

## HY-0001 — Detectar repositorios anidados

**Prioridad:** P0  
**Estado:** TODO

### Trabajo

- ejecutar `git ls-files --stage`;
- detectar entradas modo `160000`;
- inspeccionar `.git` dentro de `ComfyUI_windows_portable/ComfyUI`;
- documentar gitlinks y repositorios anidados.

### Aceptación

- existe `docs/current_state.md`;
- se identifica exactamente dónde vive el workflow real;
- no quedan rutas críticas invisibles para el repositorio padre.

---

