# Windows Compatibility — Build Portability

Documents the portability work that makes the project build on Windows (MSVC + CUDA on
Visual Studio 2022) without altering the existing Linux workflow (`mkbuild.sh`, OptiX SDK
under `$HOME/nvidia/Optix91`, `sm_120`).

## Objective

The development machine runs Linux; the validation machine runs Windows with an
RTX 4060 Laptop GPU (compute capability 8.9 → `sm_89`). The application had to compile
on both from the same source tree, with a one-command build script on each platform and
no behavioural change to the code.

## Environment

| Component | Linux (development) | Windows (validation) |
|---|---|---|
| GPU / target arch | RTX (Blackwell), `sm_120` | RTX 4060 Laptop, `sm_89` (auto-detected) |
| Compiler | g++ / GCC | MSVC 14.38 (Visual Studio 2022 Community) + Ninja |
| CUDA toolkit | per existing setup | 13.2 (`nvcc` on `PATH`) |
| CMake | per existing setup | 3.31.3 |
| OptiX | SDK at `$HOME/nvidia/Optix91` | 9.1.0 at `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |
| Driver | per existing setup | 596.36 (ships `nvoptix.dll`) |

No third-party package needed to be installed on Windows: the OptiX SDK 9.1.0
distribution bundles everything the project requires besides the C++ toolchain —
the GLFW 3.4 headers, `glfw3dll.lib`, `glfw3.dll` and a CMake package config
(`lib/cmake/glfw3`).

## Key platform findings

**OptiX 9 does not require a link-time import library on Windows.**
The SDK ships no `optix.lib`; the host API is resolved at runtime. `optix_stubs.h`
declares every host function as an inline stub forwarding through a global
`OptixFunctionTable`; the single definition lives in `optix_context.cpp`
(`optix_function_table_definition.h`), and `optixInit()` — already called during
`OptixContext` construction — loads the driver's `nvoptix.dll` (exporting only
`optixQueryFunctionTable`) and fills the table. The same mechanism is used on Linux
against `libnvoptix.so.1`; it is the reason the Linux build links `dl`. Consequently
no OptiX library had to be added to the Windows link line.

**The Windows SDK's OpenGL surface is legacy-only.**
Two separate gaps had to be bridged:

1. Recent Windows SDK installations (10.0.19041 / 22621 / 26100 present here) ship
   only the legacy `GL\GL.h` (GL 1.1 types, constants and prototypes); the modern
   `GL\glext.h` may be absent entirely. `glfw3.h` includes `<GL/gl.h>` and, when
   `GLFW_INCLUDE_GLEXT` is defined, `<GL/glext.h>` — the second include must therefore
   be satisfiable, and the modern core prototypes/constants the code uses
   (`GL_CLAMP_TO_EDGE`, `GL_RGBA8`, `GL_STREAM_DRAW`, `GL_VERTEX_SHADER`, VAO and
   buffer-object entry points, ...) must be declared.
2. The `opengl32.lib` import library covers only the legacy entry points:
   `__imp_glGenVertexArrays`, `__imp_glCreateProgram`, `__imp_glBufferData`, ... are
   not present in it (verified against every SDK version installed), so linking the 23
   modern core functions fails with `LNK2019` even though `opengl32.dll` exports
   them. This is the same reason GLEW/GLAD load functions through
   `wglGetProcAddress` on Windows.

## Implementation

### `winmkbuild.bat` (new)

Windows counterpart of `mkbuild.sh`, mirroring its contract (clean configure + build,
Release, `sm` architecture explicit):

1. Locate the project root (directory containing the script).
2. Detect the CUDA compute capability with
   `nvidia-smi --query-gpu=compute_cap --format=csv`; the column-header line is
   filtered with a numeric regex and the dot stripped (`8.9` → `sm_89`). Falls back
   to `sm_89` if the query fails. (`mkbuild.sh` fixes `sm_120`; the two machines have
   different GPUs, so the value is a parameter in both scripts.)
3. Locate Visual Studio through `vswhere` and enter the 64-bit developer environment
   (`vcvars64.bat`), failing with a clear message if the C++ toolset is absent.
4. Remove and recreate `build/`, configure with the Ninja generator
   (`-DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_ARCH=<detected>`),
   build with `cmake --build build --config Release` (equivalent of `make -j$(nproc)`;
   Ninja is parallel by default).

### CMake changes

Root `CMakeLists.txt`:

- `OPTIX_ROOT_DIR` defaults to the Windows SDK path under `if(WIN32)` and to
  `$ENV{HOME}/nvidia/Optix91` otherwise; it remains a `CACHE PATH` variable overridable
  from the command line on both platforms.
- On Windows, if `glfw3_DIR` is not already set and the SDK ships its GLFW package
  config, that path is used as a hint for `find_package(glfw3)`; on Linux the system
  package is found exactly as before.

`src/app/CMakeLists.txt`:

- Windows-only compile definitions `GL_GLEXT_PROTOTYPES`, `GLFW_INCLUDE_GLEXT`,
  `WIN32_LEAN_AND_MEAN`, `NOMINMAX` (the first two make `glfw3.h` pull in the modern
  prototypes through the shim; `NOMINMAX` protects `std::min`/`std::max` from the
  Windows macros).
- On Windows the repository's `include/` directory is added to the target include path
  (it holds the GL shim and a byte-identical copy of the SDK headers).
- `gl_shim_loader.cpp` is added to the sources only on Windows.
- `dlopen`/`m` handling: `dl` and `m` are now linked only under `if(UNIX)` (the
  unconditional `dl` broke the Windows link).
- A `POST_BUILD` step copies the SDK's prebuilt `glfw3.dll` next to `optix_app.exe`.

### GL shim: `include/GL/glext.h` (new)

Sits in the include path only on Windows and satisfies `glfw3.h`'s
`#include <GL/glext.h>`:

- `#include <GL/gl.h>` first (the legacy header, for the base types and the legacy
  entry points), then declares the constants the code uses that the legacy header
  lacks and prototypes the modern core functions with the same `WINGDIAPI` /
  `APIENTRY` decoration `GL\GL.h` uses, so overlapping declarations stay consistent.
- The 16 entry points present in `opengl32.lib` (`glBindTexture`, `glDrawArrays`,
  `glGetError`, `glVertexAttribPointer`, `glUniform3f`, ...) stay ordinary function
  declarations and link normally.
- The 23 entry points missing from the import library (`glGenVertexArrays`,
  `glCreateShader`, `glBindBuffer`, `glUseProgram`, ...) are declared as extern
  function-pointer globals together with a `glShimLoadModern()` entry point.

### Runtime loader: `src/app/gl_shim_loader.cpp` (new)

Defines the 23 function-pointer globals and, in `glShimLoadModern()`, resolves each
with `wglGetProcAddress()` (declared by the SDK's `wingdi.h`), exits the process with a
diagnostic if any lookup fails. It is called from `GlWindow::init()` immediately
after `glfwMakeContextCurrent()` (guarded by `#ifdef _WIN32`), when the ICD can serve
the entry points.

Call sites need no changes: C++ calls a function pointer through its name
(`glGenVertexArrays(1, &vao)`), so the rest of the code is unaware of the
indirection.

## Runtime bug fixes (surfaced on the Windows machine)

Running the first full end-to-end session (DICOM load, OptiX tracing, window
close) exposed two pre-existing bugs that the Windows/MSVC build made visible.
Neither is Windows-specific; the fixes are plain source changes.

### `src/common/metrics.cpp` — self-shadowing `CUDA_CHECK` (MSVC `C4700`)

`advanceResultQueue()` contained

```cpp
cudaError_t err = cudaEventQuery(slot.end);
if (err == cudaErrorNotReady)
    break;
CUDA_CHECK(err);
```

The check macros declare their own local of the same kind
(`do { cudaError_t err = call; ... }`), so `CUDA_CHECK(err)` expands to
`cudaError_t err = err;` — the initializer reads its own not-yet-initialized
storage (undefined behaviour). MSVC diagnosed exactly this with
`C4700: variable 'err' used uninitialized`; at run time the stack garbage in
that slot was a non-zero, out-of-range value, so `cudaGetErrorString` reported
"unrecognized error code" and the check threw as soon as a finished frame's
event query returned anything other than `cudaErrorNotReady`. The Linux toolchain
never tripped this (the slot happened to hold zero / the warning was not
emitted), which is why the bug stayed latent.

Fix: the result of `cudaEventQuery` is kept in a distinct variable
(`queryErr`) and checked explicitly, throwing with the real
`cudaGetErrorString` text if the query genuinely fails. The compiler warning is
gone and no other call site uses the shadowing pattern.

### `src/app/interop_buffer.{h,cpp}` — unregistering a still-mapped resource

`destroy()` called `cudaGraphicsUnregisterResource` directly. If a frame
exception (such as the one above) left the PBO mapped, the destructor ran with
the resource still mapped and the cleanup call failed with
"resource already mapped".

Fix: a `mapped_` state (set in `map()`, cleared in `unmap()`); `destroy()` now
unmaps on the default stream before unregistering, so teardown is safe even
mid-frame after an exception.

## Linux compatibility audit

- `mkbuild.sh` and `cmake/OptiXIR.cmake` are unmodified.
- Every CMake addition is gated by `if(WIN32)`; the Linux default of `OPTIX_ROOT_DIR`
  and the Linux link set (`dl` + `m`) are unchanged.
- `gl_shim_loader.cpp` is compiled and `include/GL/glext.h` is placed on the include
  path only on Windows; on Linux `GLFW_INCLUDE_GLEXT` is not defined, so `glfw3.h`
  never requests the shim and it cannot shadow the system GLVND `GL/glext.h`.
- The Windows-specific source change visible to both compilers is the 3-line
  `#ifdef _WIN32` block in `gl_window.cpp`, removed by the Linux preprocessor.
- The two runtime bug fixes (`metrics.cpp`, `interop_buffer.{h,cpp}`) are
  platform-neutral source changes and therefore also apply to the Linux build;
  they only change observable behaviour in failure cases (a spurious throw with
  an "unrecognized error code" and a failed cleanup unregister), not the
  normal rendering path.

## Files changed

| File | Change |
|---|---|
| `winmkbuild.bat` | New: one-command Windows build script |
| `CMakeLists.txt` | Platform-aware `OPTIX_ROOT_DIR` default; SDK-bundled GLFW discovery on Windows |
| `src/app/CMakeLists.txt` | Windows definitions/shim dir/loader source/dll copy; `dl m` confined to UNIX |
| `include/GL/glext.h` | New: GL constants + prototypes; 23 entry points as function pointers |
| `src/app/gl_shim_loader.cpp` | New: `wglGetProcAddress`-based loader for the missing entry points |
| `src/app/gl_window.cpp` | Calls `glShimLoadModern()` after the context becomes current (`_WIN32` only) |
| `src/common/metrics.cpp` | Self-shadowing `CUDA_CHECK(err)` replaced by an explicit check (MSVC `C4700`; spurious "unrecognized error code" throw) |
| `src/app/interop_buffer.{h,cpp}` | Tracks `mapped_`; unmaps before unregistering in `destroy()` (safe teardown after a mid-frame exception) |

## Verification

- Clean run of `winmkbuild.bat` from an empty `build/`: configure and all 14 build
  steps succeed — OptiX IR compilation of `device_programs.cu`, 12 object files and
  the final link — with exit code 0.
- Artifacts: `build\bin\optix_app.exe`, `build\bin\glfw3.dll` (auto-copied),
  `build\src\app\device_programs.optixir`, `build\compile_commands.json`.
- Only compiler diagnostics are pre-existing MSVC `C4996` deprecation warnings
  (`getenv`, `fopen`) in untouched code; no C++ or OptiX API semantics were altered.
- OptiX runtime resolution was confirmed on the validation machine: driver 596.36
  provides `nvoptix.dll` exporting `optixQueryFunctionTable`, matching the SDK 9.1.0
  ABI used by `optixInit()`.
- Full end-to-end run on the validation machine after the two runtime fixes:
  `optix_app.exe` loads a 512×512×216 CT series (216 MB volume upload), runs the
  manual-mode render loop (OptiX tracing on `sm_89`) and shuts down cleanly on
  window close — no CUDA cleanup errors and no spurious exceptions.
