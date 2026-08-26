# Dual-Mode Rendering Pipeline — Manual vs. OptiX Trace

Documents the implementation of the dual-mode rendering pipeline for the thesis
comparison between manual ray-AABB traversal (dense baseline) and OptiX RT-core
accelerated traversal (sparse strategies). Introduces a CLI `--mode manual|optix`
flag to select between the two code paths at runtime.

## Objective

Implement two distinct traversal strategies for volume ray-casting so they can
be compared under identical conditions:

1. **Manual mode** (`--mode manual`): The raygen program performs its own
   ray-AABB slab test and iterates through the volume with a for-loop. No
   `optixTrace` call; the GPU's RT cores are idle. This is the dense baseline
   control condition.

2. **OptiX mode** (`--mode optix`): The raygen program calls `optixTrace`,
   handing traversal to the RT-core BVH hardware. A custom intersection program
   (`__intersection__is`) performs the slab test and reports the entry/exit
   t-values. A closest-hit program (`__closesthit__ch_vol`) receives those
   t-values and performs the same volume march. This path exercises the RT-core
   and is where sparsity strategies become load-bearing.

Both modes produce identical visual output for the same dataset and transfer
function. The difference is purely in *how* the ray-AABB intersection is found.

## OptiX 9.1 API differences from 7.x

Several API changes in OptiX 9.1 affected the implementation:

### Ray origin/direction query functions renamed

OptiX 7.x provided `optixGetRayOrigin()` and `optixGetRayDirection()`. These
no longer exist in OptiX 9.1. The replacements are:

- `optixGetWorldRayOrigin()` — available in IS, AH, CH, MS (and RG after `optixTrace`)
- `optixGetWorldRayDirection()` — same availability

This affects `device_programs.cu` where the intersection and closest-hit
programs query the current ray.

### Payload functions restricted to non-raygen programs

In OptiX 9.1, `optixGetPayload_0/1/2()` are **not** callable from raygen
programs. They are only available in IS, AH, CH, and MS. The raygen program
reads payload back through the `optixTrace` output parameters (`p0, p1, p2`)
which are written in-place by the CH/MS programs via `optixSetPayload_*`.

This was discovered at runtime: the module compiled but produced
`OPTIX_ERROR_INVALID_FUNCTION_USE` with the error "Illegal call to
optixGetPayload in function __raygen__rg_optix with semantic type RAYGEN".

### No standalone intersection program group

OptiX 7.x had `OPTIX_PROGRAM_GROUP_KIND_INTERSECTION` as a separate program
group kind. OptiX 9.1 does not — the intersection program is embedded in the
hitgroup program group via `OptixProgramGroupHitgroup::moduleIS` and
`::entryFunctionNameIS`.

This means `createProgramGroups()` creates only three program groups (raygen,
miss, hitgroup) instead of four. The IS module is set directly on the hitgroup
descriptor:

```cpp
hitgroup_desc.hitgroup.moduleIS     = module_;
hitgroup_desc.hitgroup.entryFunctionNameIS = intersectionEntry;
```

### AABB-based GAS uses `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`

The old `OPTIX_BUILD_INPUT_TYPE_AABB` with `aabbArray` struct does not exist
in OptiX 9.1. The correct equivalent is:

- Build input type: `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`
- Struct: `buildInput.customPrimitiveArray` with `.aabbBuffers` (a
  `const CUdeviceptr*`), `.numPrimitives`, etc.
- AABB struct: `OptixAabb` (same layout: `{minX, minY, minZ, maxX, maxY, maxZ}`)

### `usesPrimitiveTypeFlags` set to 0

Setting `usesPrimitiveTypeFlags = 0` in the pipeline compile options enables
**both** `OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM` and
`OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE` (per the OptiX 9.1 documentation). This
is necessary because the single OptiX IR file contains programs for both modes
(manual uses triangle-compatible shaders; optix mode uses custom intersection).

## Architecture

### Program graph

```
Manual mode (--mode manual):
  __raygen__rg ──────────────────────── (no optixTrace, manual AABB loop)
  __miss__ms
  __closesthit__ch                     (stub, not invoked)

OptiX mode (--mode optix):
  __raygen__rg_optix ──► optixTrace ──┐
                                       ├──► __intersection__is   (AABB slab test)
                                       ├──► __closesthit__ch_vol (volume march)
                                       └──► __miss__ms           (background)
```

### Shared code

Both modes share:

- `intersectAABB()` — slab test against the volume bounding box
- `volumeMarch(float3 origin, float3 direction, float tmin, float tmax)` —
  front-to-back compositing loop with early ray termination (accumA >= 0.99)
- `setPayload(float3 p)` — packs float3 into three uint32 payload slots

The `volumeMarch` helper takes origin and direction as explicit parameters
(rather than calling `optixGetWorldRayOrigin/Direction` internally) because
the manual raygen has no active OptiX ray context — those functions are only
valid during/after `optixTrace`.

### VolumeScene vs PlaceholderScene

- **PlaceholderScene** (manual mode): Builds a triangle-mesh GAS (a simple cube
  or plane). The raygen program never calls `optixTrace`, so the GAS is only
  used to satisfy `optixLaunch`'s requirement for a traversable handle.

- **VolumeScene** (optix mode): Builds an AABB-based GAS using
  `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`. The single AABB encompasses the
  entire volume bounding box. The intersection program is invoked during RT-core
  traversal and performs the slab test, reporting entry/exit t-values as OptiX
  attributes.

### Attribute passing between IS and CH

The intersection program reports two 32-bit attributes via
`optixReportIntersection`:

```cuda
optixReportIntersection(tmin, 0,
    __float_as_uint(tmin),   // attribute 0: entry t
    __float_as_uint(tmax));  // attribute 1: exit t
```

The closest-hit program reads them back:

```cuda
float tmin = __uint_as_float(optixGetAttribute_0());
float tmax = __uint_as_float(optixGetAttribute_1());
```

This requires `numAttributeValues = 2` in the pipeline compile options.

## Files changed

### `src/app/pipeline_base.h`

- Added `enum class TraceMode { MANUAL, OPTIX }`
- Extended `init()` signature with `const char* intersectionEntry` and
  `TraceMode mode` parameters
- Added `mode()` accessor

### `src/app/pipeline_base.cpp`

- `createModule()`: Sets `usesPrimitiveTypeFlags = 0` (enables both CUSTOM and
  TRIANGLE) instead of mode-specific flags
- `createProgramGroups()`: When mode is OPTIX and an intersection entry is
  provided, sets `moduleIS` and `entryFunctionNameIS` on the hitgroup descriptor
  (OptiX 9.1 embeds IS in the hitgroup PG — no standalone intersection PG)
- `createPipeline()`: Links only three program groups (raygen, miss, hitgroup)

### `src/app/device/device_programs.cu`

- Extracted `volumeMarch()` as a shared `__device__` helper taking explicit
  `origin, direction, tmin, tmax` parameters (not calling
  `optixGetWorldRayOrigin/Direction` internally)
- Added `__raygen__rg_optix` — computes camera ray, calls `optixTrace`, reads
  payload from the `optixTrace` output parameters `p0/p1/p2` (not via
  `optixGetPayload_*`, which is illegal in raygen in OptiX 9.1)
- Added `__intersection__is` — calls `optixGetWorldRayOrigin/Direction`,
  performs `intersectAABB`, reports entry/exit t-values via
  `optixReportIntersection`
- Added `__closesthit__ch_vol` — reads attributes 0/1, calls
  `volumeMarch(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), tmin, tmax)`,
  sets payload
- Removed `getPayload()` helper (was illegal to call from raygen in OptiX 9.1)
- Added `getPayload()` was removed entirely since it is no longer used

### `src/app/volume_scene.cpp`

- Fixed for OptiX 9.1 API: `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES` with
  `customPrimitiveArray` struct (was written for older OptiX with
  `OPTIX_BUILD_INPUT_TYPE_AABB` / `aabbArray`)
- `aabbBuffers` is `const CUdeviceptr*` (pointer to array of device pointers),
  not raw `CUdeviceptr`

### `src/app/volume_scene.h`

- Added `init(OptixDeviceContext, CUstream, float3 bmin, float3 bmax)` that
  takes explicit bounding box corners

### `src/app/app_main.cpp`

- Added `#include "volume_scene.h"`
- Added `--mode manual|optix` CLI argument parsing (manual is default)
- Branching on mode:
  - OPTIX: Creates `VolumeScene`, selects `__raygen__rg_optix` raygen entry,
    creates pipeline with `__closesthit__ch_vol` and `__intersection__is`
  - MANUAL: Creates `PlaceholderScene`, selects `__raygen__rg` raygen entry,
    creates pipeline with `__closesthit__ch` (stub) and no intersection program

### `src/app/CMakeLists.txt`

- Added `volume_scene.cpp` to the `optix_app` executable sources

## Bugs encountered and fixed

### 1. `optixGetWorldRayOrigin` undefined in raygen

**Symptom:** `OPTIX_ERROR_INVALID_FUNCTION_USE` during `optixModuleCreate` with
error "Illegal call to optixGetWorldRayOrigin in function __raygen__rg".

**Cause:** The `volumeMarch()` helper called `optixGetWorldRayOrigin()` and
`optixGetWorldRayDirection()` internally. In the manual raygen (`__raygen__rg`),
no `optixTrace` has been called, so there is no active ray context — these
functions are only valid during/after traversal.

**Fix:** Changed `volumeMarch` signature to accept explicit `origin` and
`direction` parameters. The manual raygen passes its locally-computed camera
ray; the closest-hit program passes `optixGetWorldRayOrigin/Direction()`.

### 2. `optixGetPayload` illegal in raygen

**Symptom:** `OPTIX_ERROR_INVALID_FUNCTION_USE` during `optixModuleCreate` with
error "Illegal call to optixGetPayload in function __raygen__rg_optix with
semantic type RAYGEN".

**Cause:** In OptiX 9.1, `optixGetPayload_0/1/2()` are only available in IS,
AH, CH, and MS — not in raygen programs. The raygen receives payload through
the `optixTrace` output parameters which are written in-place.

**Fix:** After `optixTrace(params.handle, ..., p0, p1, p2)`, the raygen reads
the result directly from `p0, p1, p2`:

```cuda
float3 color = make_float3(
    __uint_as_float(p0),
    __uint_as_float(p1),
    __uint_as_float(p2));
```

### 3. `OPTIX_PROGRAM_GROUP_KIND_INTERSECTION` does not exist

**Symptom:** Compilation error: `'OPTIX_PROGRAM_GROUP_KIND_INTERSECTION' was
not declared in this scope`.

**Cause:** OptiX 9.1 does not have a standalone intersection program group kind.
The intersection program is embedded in the hitgroup program group.

**Fix:** Removed the separate intersection PG creation. Instead, set
`hitgroup_desc.hitgroup.moduleIS` and `::entryFunctionNameIS` directly when
creating the hitgroup program group. Removed the `intersectionPG_` member from
`PipelineBase`.

### 4. `usesPrimitiveTypeFlags` incompatible with single OptiX IR

**Symptom:** `OPTIX_ERROR_INVALID_FUNCTION_USE` during module creation even
after fixing the above. The module contains programs for both modes, but was
compiled with only one primitive type flag.

**Cause:** Setting `OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE` for manual mode or
`OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM` for optix mode made the module reject the
other mode's programs.

**Fix:** Set `usesPrimitiveTypeFlags = 0`, which the OptiX 9.1 documentation
states "corresponds to enabling OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM and
OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE" — both flags simultaneously.

## Usage

```bash
# Manual mode (dense baseline, no RT-core traversal)
./build/bin/optix_app <path-to-dicom> --mode manual

# OptiX mode (AABB GAS, intersection program, optixTrace)
./build/bin/optix_app <path-to-dicom> --mode optix

# Default is manual if --mode is omitted
./build/bin/optix_app <path-to-dicom>
```

## Status

- Manual mode (`--mode manual`): Builds and runs. Verified at runtime.
- OptiX mode (`--mode optix`): Builds successfully. Module creation and pipeline
  linking pass. Runtime testing pending (requires display with DICOM data and
  validation that the intersection program correctly reports t-values).
