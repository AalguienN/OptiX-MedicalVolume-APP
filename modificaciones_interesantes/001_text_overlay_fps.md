# 001 - TextOverlay: Renderizado de FPS con bitmap font

## Fecha
2026-06-14

## Descripción
Se implementó un sistema de renderizado de texto directamente en OpenGL 3.3 core, sin usar ImGui ni el SDK de OptiX. Se usó un bitmap font 8x8 público (VGA ROM font, dominio público) para mostrar FPS en pantalla.

## Archivos creados
- `src/common/text_overlay.h` — Header reusable con la clase `TextOverlay`:
  - Bitmap font 8x8 con todos los caracteres ASCII imprimibles (32-126)
  - Textura atlas 128x64 generada desde el array de bytes
  - Shaders vertex/fragment embedidos como strings
  - Proyección ortho con Y invertida (0,0 = arriba-izquierda)
  - Métodos: `init()`, `destroy()`, `resize()`, `draw()`
  - Sin dependencias externas (solo GLFW/OpenGL 3.3)

## Archivos modificados
- `src/gl_cube_fps/gl_cube_fps.cpp`:
  - Se añadió `#include "text_overlay.h"`
  - Se inicializa `TextOverlay` tras el buffer de OpenGL
  - Se añadió suavizado EMA (alpha=0.05) para el FPS
  - Se reemplazó el código comentado de ImGui por `overlay.draw()`
  - Se llama a `overlay.destroy()` en la limpieza

## Uso desde otros subproyectos
Cualquier subproyecto que use GLFW + OpenGL 3.3 puede usarlo:

```cpp
#include "text_overlay.h"

TextOverlay overlay;
overlay.init(width, height);

// En el loop de render:
overlay.draw(10.0f, 10.0f, "Hola mundo", 1.0f, 1.0f, 1.0f, 1.0f);

// Al salir:
overlay.destroy();
```

No requiere cambios en CMakeLists.txt (todos los subproyectos ya incluyen `src/common/`).

## Licencia del font
El bitmap font 8x8 proviene de la VGA ROM de IBM (Marcel Sondaar), redistribuido como dominio público a través del proyecto font8x8 de Daniel Hepper.
