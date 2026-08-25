# Application Infrastructure — Implementation Notes

Source material for the *Development → Application infrastructure* section of the thesis
(`tfm.tex:836`). Covers the work performed on commit range that introduced `src/app/`.

## Objective

Implement the four deliverables defined in the plan:

1. Creation of an OpenGL context window.
2. Creation of the OptiX context, base pipeline and Shader Binding Table (SBT).
3. Implementation of the CUDA-OpenGL interoperability through a shared Pixel Buffer Object (PBO).
4. Basic skeleton of the monolithic rendering application (volumes explicitly excluded at this stage),

while honoring the architecture of Section *High-level system architecture* (`tfm.tex:638`) and the
three design principles of Section *Design principles* (`tfm.tex:665`).

## Starting point

The repository previously consisted of five independent demo programs (`base`, `gl_cube`,
`gl_cube_fps`, `optixConsole`, `test`), each a standalone `main()` derived from OptiX SDK samples.
They proved individual techniques (context creation, GAS building, PBO interop, per-frame launch)
but duplicated identical code five times: OptiX/CUDA/GL error macros, SBT record definitions,
module loading, and the CMake custom command that compiles `.cu` files to OptiX IR. None of them
matched the target architecture: there was no single application and no separation between the
components committed to in the design chapter.

`gl_cube` was selected as the functional basis because it already contained the complete vertical
slice: GLFW window, triangle-mesh GAS, raygen/miss/closest-hit programs, PBO interop and a
fullscreen-quad display, refreshed every frame.

## Implementation

The application now lives in `src/app/` as one executable (`optix_app`) whose modules correspond
one-to-one with the components of the architectural overview figure:

| Module | Architectural component | Responsibility |
|---|---|---|
| `gl_window.{h,cpp}` | OpenGL Display (window part) | GLFW initialization, GL 3.3 core context, double-buffered window with vsync; framebuffer-size queries |
| `display.{h,cpp}` | OpenGL Display | Presentation only: owns the RGBA8 texture, the fullscreen-quad shader program and VAO; uploads the PBO into the texture (`glTexSubImage2D` with `GL_PIXEL_UNPACK_BUFFER` bound) and draws the quad. Contains no rendering logic |
| `optix_context.{h,cpp}` | OptiX Pipeline (foundation) | CUDA runtime activation (`cudaFree(0)`), `optixInit()`, `OptixDeviceContext` creation, command stream |
| `pipeline_base.{h,cpp}` | OptiX Pipeline | Loads the compiled `.optixir` module, creates the ray-generation, miss and hit-group program groups, links the pipeline, computes and sets the device stack size, and builds the SBT (raygen/miss/hitgroup records). Program entry-point names are constructor parameters, leaving a seam for strategy-specific intersection programs |
| `interop_buffer.{h,cpp}` | CUDA-OpenGL Interoperability | Owns the PBO, registers it as a CUDA graphics resource with `cudaGraphicsMapFlagsWriteDiscard`, and exposes map/unmap returning the device pointer OptiX writes into |
| `placeholder_scene.{h,cpp}` | (temporary stand-in) | Builds the cube mesh and its compacted bounding-volume GAS; will be replaced by strategy-provided acceleration structures in the dense-baseline stage |
| `camera.h` | User Interaction / Input (placeholder) | Orbital camera state (theta/phi/radius/fov/aspect) producing eye point and UVW basis; driven by time until the interaction stage implements input handling |
| `shared_device.h` | shared host/device contract | `Params`, `RayGenData`, `MissData`, `HitGroupData` and the aligned `SbtRecord<T>` template used by both host code and device programs |
| `device/device_programs.cu` | OptiX device programs | Ray generation (primary ray from camera UVW basis), miss (background gradient), closest hit (face normal shading); compiled offline to OptiX IR |
| `app_main.cpp` | composition root | Constructs all components, runs the frame loop, handles framebuffer resize |

Supporting changes outside `src/app/`:

- `src/common/check_macros.h`: single definition of `CUDA_CHECK`, `OPTIX_CHECK`, `OPTIX_CHECK_LOG`
  plus non-throwing `*_NOEXCEPT` variants used during cleanup (destructors must not throw).
- `src/common/gl_check.h`: `GL_CHECK` and `GL_CHECK_NOEXCEPT`.
- `cmake/OptiXIR.cmake`: reusable CMake function `add_optix_ir_target()` encapsulating the nvcc
  `--optix-ir` invocation, replacing the per-directory copy-pasted custom command.
- Root `CMakeLists.txt`: discovers CUDA, GLFW and OpenGL once; `src/CMakeLists.txt` builds only
  `app`. The five demos were removed from the tree.

## Frame loop and data flow

Each iteration of `app_main.cpp` executes:

1. Poll events; query framebuffer size and, if the window was resized, recreate the interop buffer
   and display texture and update camera aspect ratio.
2. Advance the placeholder orbit camera and compute eye + UVW basis.
3. Map the PBO through CUDA (`cudaGraphicsMapResources` +
   `cudaGraphicsResourceGetMappedPointer`), obtaining a device `uchar4*`.
4. Upload launch parameters asynchronously and update the raygen SBT record (camera basis) on the
   stream.
5. `optixLaunch(pipeline, stream, params, sizeof(Params), sbt, width, height, 1)` — one launch per
   pixel buffer, continuous while the window is open.
6. Unmap the PBO; the display component pulls it into the texture and draws the fullscreen quad;
   buffers are swapped.

No rendered pixel ever traverses the host: the image produced by OptiX lives exclusively in GPU
memory shared with OpenGL through the registered PBO, satisfying the interop requirement
(`tfm.tex:656`) and the GPU-oriented execution principle.

## Compliance with design constraints

- **Common rendering pipeline** (principle 1): raygen, miss, transfer of colors and compositing
  order are fixed once in `device_programs.cu` and `pipeline_base`; strategies will only replace
  the intersection stage and traversable.
- **Interchangeable representations** (principle 2): `PipelineBase::init()` accepts entry-point
  names, and `Params.handle` carries the traversable; the dense-baseline GAS and intersection
  program plug in without touching window, display or loop code. The cube scene is quarantined in
  `PlaceholderScene` precisely so it can be swapped out.
- **GPU-oriented execution** (principle 3): per-frame launches with zero host copies; the only
  host→device traffic is the small `Params` struct and the raygen SBT record.
- **Real-time readiness** (non-functional requirements): continuous relaunch per frame at vsync
  rate is the same structure needed for the 30 FPS / <50 ms latency targets; instrumentation hooks
  can be added around the single `optixLaunch` call site.

## Verification

- Full rebuild from scratch through the existing `mkbuild.sh` (CMake Release, `sm_120`) completes
  without errors or compiler warnings.
- The binary opens a window and renders the shaded rotating cube indefinitely; killed externally
  after several seconds of continuous rendering with no CUDA, OptiX or GL errors reported.

## Explicitly out of scope at this stage

DICOM loading and preprocessing, Hounsfield conversion, volume representations, orbital-camera
input handling, and metrics instrumentation belong to the subsequent sections of the development
plan (`tfm.tex:842` onwards) and were deliberately left out.
