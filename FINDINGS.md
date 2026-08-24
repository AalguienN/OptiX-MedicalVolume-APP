# Findings: Reaching the "Application infrastructure" objective

Comparison between the plan defined in `~/Documents/TFM/tfm.tex` (Chapter *Development*, section
*Application infrastructure*, and the constraints from *Solution Design*) and the **current state of
this repository** (`optix_clone`).

---

## 1. What the document requires

The *Application infrastructure* subsection (`tfm.tex:836`) defines four deliverables:

1. **Creation of OpenGL context window**
2. **Creation of OptiX context, pipeline base, SBT** (Shader Binding Table)
3. **Implementation of CUDA-OpenGL interop** (shared PBO)
4. **Basic skeleton of the monolithic application for rendering**
   (volumes explicitly excluded at this stage)

### Constraints and design patterns that apply (from *Solution Design*)

| Source | Constraint |
|---|---|
| §High-level system architecture (`tfm.tex:638`) | System split into components: Volume Loading *(later)*, **OptiX Pipeline**, **CUDA-OpenGL Interoperability**, **OpenGL Display**, **User Interaction / Input** |
| §Design principles (`tfm.tex:665`) | 1) Common rendering pipeline shared by all strategies; 2) **Interchangeable volume representations behind a clear interface**; 3) GPU-oriented execution (everything on GPU, no per-frame host round-trips) |
| §CUDA-OpenGL Interoperability (`tfm.tex:656`) | PBO registered as CUDA graphics resource; OptiX writes directly into GPU memory visible to OpenGL; **no host copies** |
| §OpenGL Display (`tfm.tex:658`) | Display-only component: full-screen quad textured with the interop buffer, no render logic inside |
| §Shared Rendering Pipeline (`tfm.tex:707`) | Single raygen + single miss program common to all strategies; only intersection (+AS) is swapped per strategy |
| §Strategy-Specific OptiX Programs (`tfm.tex:747`) | Dense baseline = fixed-step marcher against a **single bounding-box GAS** — but this belongs to the *next* stage, not to infrastructure |
| §Non-functional requirements (`tfm.tex:514`) | ≥ 30 FPS, < 50 ms input latency → the loop must be continuous-launch (per frame), not one-shot |
| §Metrics Instrumentation (`tfm.tex:780`) | Later stage, but hooks should not require rewriting the render path |

Explicitly **out of scope** for infrastructure: DICOM loading, HU conversion, dense/sparse volume
representations, orbital camera math (next section), transfer functions.

---

## 2. Current state of `optix_clone` vs. requirements

The repo today is a set of **independent throwaway demos** (clones of OptiX SDK samples), each with
its own `main()`, duplicated macros and duplicated CMake logic:

| Component | Exists? | Where | Notes |
|---|---|---|---|
| OpenGL context window | Partial | `src/gl_cube_fps/gl_cube_fps.cpp:290` (also `gl_cube`) | GLFW window + GL context proven working, but embedded in demo code |
| OptiX context / pipeline / SBT | Partial | `src/base/base.cpp:72-179`, repeated in every demo | Context, module load (optix-ir file), program groups, pipeline, SBT all work — but copy-pasted in `base`, `test`, `gl_cube`, `gl_cube_fps`, `optixConsole` with no shared code |
| CUDA-GL interop (PBO) | Partial | `src/gl_cube_fps/gl_cube_fps.cpp:183-220` | PBO → `cudaGraphicsGLRegisterBuffer` (write-discard) → map → launch → unmap → `glTexSubImage2D` onto a texture. Correct pattern, not factored out |
| Monolithic app skeleton | **Missing** | — | No single application target; no component separation; no main loop owning window + pipeline + display together as designed modules |
| Shared utilities | Partial | `src/common/{helpers.h, vec_math.h, text_overlay.h}` | Reusable, but `OPTIX_CHECK`/`CUDA_CHECK` macros are redefined per-file (e.g. `base.cpp:27-63`) |
| Renderer interface (design principle 2) | **Missing** | — | Nothing anticipates swappable strategies; `Params` structs diverge slightly between demos |
| Build system | Weak | root `CMakeLists.txt` | Legacy `find_package(CUDA)`; C++11; identical ~30-line optix-ir custom command duplicated in every sub-CMakeLists; `project/CMakeLists.txt` is empty |

**Verdict:** all four infrastructure bullets exist only as *scattered proof-of-concept fragments*.
Nothing is wrong technically — `gl_cube_fps` already demonstrates the full per-frame loop
(window → map PBO → `optixLaunch` → unmap → draw quad) — but nothing matches the architecture
the thesis commits to: there is no monolithic application, and the five architectural components
of Figure *OverallSystemArchitecture* are not separable.

---

## 3. Changes required to reach the objective

### 3.1 New application skeleton (single executable)

Create e.g. `src/app/` implementing the components of §ArchitectureOverview as classes, wired in
one `main()`:

```
src/app/
├── app_main.cpp          # main(): owns components + render loop
├── gl_window.h/.cpp      # [Component] GLFW window + GL context + vsync; exposes framebuffer size
├── optix_context.h/.cpp  # [OptiX Pipeline] cudaFree(0), optixInit(), device context, stream
├── pipeline_base.h/.cpp  # [OptiX Pipeline] module (.optixir) load, program groups,
│                         #   pipeline create + stack sizes, SBT build (raygen/miss records)
├── interop_buffer.h/.cpp # [Interop] PBO creation, cudaGraphicsGLRegisterBuffer(WriteDiscard),
│                         #   map()/unmap() returning uchar4* device pointer
├── display.h/.cpp        # [Display] full-screen quad VAO + texture, blit from PBO (display-only)
├── camera.h              # placeholder struct (eye/u/v/w or view-proj); real orbit comes later
│                         #   (User Interaction Implementation stage)
└── shared_device.h       # Params struct + SbtRecord<T> shared with device code
src/app/device/
└── device_programs.cu    # __raygen__rg + __miss__ms ONLY (no volume logic yet);
                          #   raygen writes a placeholder pattern through params.image
```

Key decisions matching the document:

- **Monolithic**: one executable, components as internal modules — exactly "basic skeleton of the
  monolithic application".
- **Render loop** (`app_main.cpp`): poll events → if camera changed (always, for now):
  `map interop buffer` → `optixLaunch(pipeline, stream, d_params, …, width, height, 1)` →
  `unmap` → `display.draw()` → swap buffers. Continuous launch satisfies the 30 FPS NFR and
  mirrors the loop already validated in `gl_cube_fps`.
- **No host copies**: image data never touches CPU RAM (interop via registered PBO only).
- **Interface seam for strategies** (principle 2): `pipeline_base` takes program-group names /
  module handles as configuration, so the dense-baseline intersection program of the next stage
  plugs in without touching window/display/input code.
- **SBT**: raygen + miss records now; hitgroup slot left empty until volumes arrive (a
  bounding-box GAS will also come with the dense baseline, per `tfm.tex:747`).

### 3.2 De-duplicate shared code into `src/common`

- Move `SbtRecord<T>`, `OPTIX_CHECK`, `OPTIX_CHECK_LOG`, `CUDA_CHECK`, `GL_CHECK` into
  `src/common/` headers (currently duplicated in all five demos).
- One canonical `Params` layout in `shared_device.h` included by both host and `.cu`.

### 3.3 Build system

- Promote `find_package(glfw3 OpenGL)` to the root `CMakeLists.txt`; keep the existing
  `--optix-ir` nvcc custom command but factor it into a CMake function/macro (e.g.
  `add_optix_ir_target(<name>.cu)`) so it stops being copy-pasted per directory.
- Add `add_subdirectory(app)`; keep the demos under `src/` untouched as reference material
  (they document the learning path and remain useful for debugging driver/toolkit issues).
- Fill or delete the empty `project/CMakeLists.txt` to avoid confusion about where the real
  app lives.

### 3.4 Acceptance criteria for this stage

1. A single binary opens an GLFW window and stays interactive until closed.
2. Every frame is rendered by OptiX into the mapped PBO and shown via the display quad —
   zero `cudaMemcpy*(…, cudaMemcpyHostToDevice)` of pixel data in the hot path.
3. OptiX context/module/program groups/pipeline/SBT construction lives in one place
   (`pipeline_base`), parameterizable enough to add the baseline intersection program later
   without edits outside the strategy layer.
4. No volume/DICOM/camera-orbit code yet — those belong to the following sections
   (`tfm.tex:842`, `tfm.tex:845`).

### Reusable assets already in-repo

- `src/gl_cube_fps/` — closest reference for the full loop (PBO interop + per-frame launch).
- `src/base/` — minimal clean context/pipeline/SBT flow to lift into `pipeline_base`.
- `src/common/vec_math.h`, `helpers.h`, `text_overlay.h` — keep as-is.
