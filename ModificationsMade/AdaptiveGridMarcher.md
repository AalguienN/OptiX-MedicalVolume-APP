# Adaptive-Step Grid Marcher Strategy

Documents the implementation of the adaptive-step volume-rendering strategy for
the thesis (Section "Adaptive-step grid marcher" of `tfm.tex`). The dense voxel
grid is left unmodified; a separate, same-resolution **Chebyshev (L∞) distance
map** to the nearest rendering-relevant voxel drives an adaptive march that
leaps over empty regions in single samples. Introduced as a new traversal mode
via the `--mode adaptive` CLI flag, alongside the existing
`manual|optix|bricked` modes.

## Objective

Exploit spatial sparsity in CT/MR volumes without touching the dense volume
representation. The baseline samples every half-voxel step regardless of
whether the surrounding volume contributes anything to the image. By
precomputing, per voxel, the distance to the nearest *rendering-relevant*
(relevant) voxel, the marcher can advance by that distance (minus one) in a
single step, skipping entire empty regions while still sampling material at the
dense baseline's density. Designed to be measured against the dense baseline
and the bricked strategy.

## Design

### Representation

- The dense voxel grid stored in the 3D volume texture within the single-AABB
  GAS is left **unmodified**.
- A separate 3D texture of the same resolution stores, at each voxel, the
  **Chebyshev (L-infinity) distance** to the nearest relevant voxel, in voxel
  units. Relevant voxels store 0.
- A voxel is *rendering-relevant* when its transfer-function opacity is ≥ ε
  (`T(s) ≥ ε`), matching the bricked strategy's occupancy definition so the two
  empty-space notions agree (`αᵢ = T(sᵢ) < ε` → irrelevant, per
  "Rendering-Irrelevant Voxels").
- The distance map is a **1 byte / voxel** (`unsigned char`) array, clamped to
  ≤ 255. For the CT dataset (512×512×90) that is **22.5 MB**, the intended
  memory overhead of the approach.

### Distance-map construction (host side)

The `AdaptiveGridMarcher` computes occupancy from the volume + transfer
function (LUT index `(s - scalarMin)/(scalarMax - scalarMin) * 2047`, round-to-
nearest-even, matching the baseline), then runs a **corrected two-pass raster
Chamfer sweep** over the 3D grid:

- Phases initialised to 255 ("infinity"); occupied voxels to 0.
- *Forward sweep* (x,y,z ascending): relax against the 13 already-computed
  lower-orthant neighbours (`dz=-1` any dx,dy; `dz=0,dy=-1` any dx;
  `dz=0,dy=0,dx=-1`) using `v = min(v, neighbour + 1)`.
- *Backward sweep* (x,y,z descending): relax against the 13 upper-orthant
  mirror neighbours.
- Because the update `d ← min(d, d_neighbour + 1)` uses the Chebyshev (max-norm)
  metric, propagating `+1` in the correct orthants converges to the exact
  L∞ distance in two passes. Verified against brute-force over 50 random
  volumes during development.

The result is uploaded once as a 3D texture:
- `cudaMalloc3DArray` with `cudaCreateChannelDesc<unsigned char>`.
- Pitched `cudaMemcpy3D` from the host array.
- `cudaFilterModePoint` (nearest) — the device must read the *exact* stored
  distance so the leap stays strictly inside the guaranteed-empty region
  (linear filtering across an occupied boundary could shrink the value).
- `cudaAddressModeClamp`, `normalizedCoords = 1`, and — critically —
  `cudaReadModeNormalizedFloat` (see the bug below).

### Traversal

The adaptive mode uses a **manual raygen program**
(`__raygen__rg_adaptive`) with a **software AABB entry/exit test** — the same
single-AABB pattern as the manual dense baseline, and **no `optixTrace`** (no
RT-core traversal). The GAS handle is present but never traversed.

`adaptiveMarch()` mirrors the dense baseline `volumeMarch()` step-for-step for
image parity, with adaptive step sizing:

1. Base `stepSize` = half the smallest voxel spacing (the baseline's step);
   `maxSteps` capped at 4096.
2. At each step, sample the volume and apply the transfer function, compositing
   front-to-back exactly like the baseline.
3. **Conditional distance-map read**: only when the sample's TF opacity is
   `< ε` (the voxel cannot be relevant) is the distance map consulted. This
   keeps the cost inside opaque material identical to the dense baseline.
4. **The leap**: if the distance `D ≥ 2`, advance by
   `t += (D - 1) * voxelWorldStep` and count a leap; otherwise advance by the
   baseline `stepSize`.

#### Axis-aware step (`voxelWorldStep`)

The distance value `D` is a voxel count valid independently in every axis. To
keep every axis strictly inside the guaranteed-empty L∞ ball when advancing
along an arbitrary ray, the leap length is computed per-axis:

```
invDir    = 1 / |direction|                     (per axis)
voxelWorldStep = min_j( spacing_j * invDir_j )
t += (D - 1) * voxelWorldStep
```

i.e. the world length of advancing through `(D-1)` voxels is dominated by the
axis with the smallest per-axis crossing distance. This replaced an earlier,
unsatisfactory `(D-1) * params.minSpacing` heuristic (which only applied the
smallest voxel spacing, not the per-axis projection), and removed the now-unused
`Params.minSpacing` field.

#### Gated distance read

Reading the distance map only when the sampled TF alpha is `< ε` (i.e. `D >= 2`
implies transparency) yields an image identical to reading every step, while
dropping the texture fetch inside opaque material.

### Diagnostic counters

Per-pixel counters (volume samples, distance-map reads, leaps) are accumulated
into `Params.dbgCounters` (3 × `unsigned int`, `atomicAdd`, gated on a non-null
pointer) and printed per frame by the host. The `--frames N` option makes the
app exit early so the summary can be read, which was essential for the
measurement-driven diagnosis below.

## Files changed

### New: `src/app/adaptive_grid_marcher.h` / `.cpp`

Class `AdaptiveGridMarcher` implementing host-side construction and upload:

- `build(volume, tf, scalarMin, scalarMax, epsilon)` — computes occupancy from
  the volume + TF, runs the corrected 2-pass Chebyshev distance transform,
  prints the occupancy %, max distance, and D≥2 / D≥10 distribution, then
  `uploadToDevice()`.
- `uploadToDevice()` — creates the `unsigned char` 3D array + texture, uploads
  with a pitched `cudaMemcpy3D`, `cudaFilterModePoint` +
  `cudaReadModeNormalizedFloat` + clamp + normalized coords.
- `distanceMap()`, `deviceDistanceTex()` accessors; destructor frees the array
  and texture object.

### New: `src/app/device/adaptive_programs.cu`

- `adaptiveMarch()` — fixed-base-step march with conditional distance-map read
  and `(D-1) * voxelWorldStep` leap; identical TF lookup + compositing to the
  baseline for image parity; accumulates `nSamples`/`nDistReads`/`nLeaps`.
- `__raygen__rg_adaptive` — manual software AABB hit test, then `adaptiveMarch`.
- `#include`d from `device_programs.cu` so everything still lands in a single
  OptiX IR module (one `OptixModule`).

### `src/app/shared_device.h`

- Extended `Params` with `distanceTex` (texture object), `epsilon`, and
  `dbgCounters` for the adaptive strategy, under a comment marking them as
  adaptive-only fields.

### `src/app/pipeline_base.h`

- `enum class TraceMode` gained `ADAPTIVE`.

### `src/app/app_main.cpp`

- New CLI options: `--mode adaptive` (new mode) and `--frames <N>` (exit after
  N frames so the counter summary prints).
- Adaptive branch: builds a `VolumeScene` (single AABB GAS), an
  `AdaptiveGridMarcher`, wires `__raygen__rg_adaptive`, fills the adaptive
  `Params` fields, allocates `dbgCounters`, and prints the per-frame counter
  summary.
- Updated the usage text.

### `src/app/CMakeLists.txt`

- Added `adaptive_grid_marcher.cpp` to the `optix_app` sources.

### `cmake/OptiXIR.cmake`

- Fixed the OptiX IR custom command to rebuild when **any** sibling `.cu`/`.h`/
  `.hpp` in the device directory changes (previously it only `DEPENDS` on
  `device_programs.cu`, so edits to `adaptive_programs.cu` did not trigger a
  rebuild of the IR).

## Bug fix: `unsigned char` texture read returning garbage

### Symptom

When first measured, the adaptive mode rendered at the *same* volume-sample
count as the dense baseline plus extra distance reads, yet reported **0 leaps**
across hundreds of frames — so it was *slower* than the baseline despite the
host map being rich in leappable space (61% of voxels had D≥2, max D=109).

### Root cause

The distance texture was created with `cudaCreateChannelDesc<unsigned char>()`
but read on the device as `tex3D<float>(...)` with
`cudaReadModeElementType`. With element-type read mode this **returns garbage
(~0) values** for an unsigned-char channel (reproduced in isolation: the array
data was correct, but the texture read returned denormal ~4e-45 values). The
device therefore always saw `D = 0`, so `iD < 2` forever and the leap branch
never fired — the marcher degenerated to the dense baseline plus one distance
read per transparent sample, exactly matching the observed 0-leap, slower
behaviour.

### Fix

- `adaptive_grid_marcher.cpp:187` — set `cudaReadModeNormalizedFloat` on the
  texture descriptor. A `uchar` channel read in normalized mode returns the
  byte as [0,1] correctly (verified: value 21 → 21/255).
- `adaptive_programs.cu:110` — rescale on the device:
  `D = 255.0f * tex3D<float>(...)`.

This keeps the 1-byte/voxel memory benefit of the `unsigned char` map while
yielding an exact distance on read.

## Verification

CT lung dataset (512×512×90, HU [-1024, 1665], occupancy ~35.8%, max D=109),
200 frames, `--epsilon 0.01`, RTX 5070:

| mode    | ms/frame | vol-samples/frame | dist-reads/frame | leaps/frame |
|---------|----------|-------------------|------------------|-------------|
| manual  | 6.63     | 13,329,360        | —                | —           |
| bricked | 5.40     | —                 | —                | —           |
| adaptive| **2.93** | **2,648,468**     | 20,273,428       | 11,415,535  |

- Volume samples dropped **5×** (13.33M → 2.65M/frame) once leaps fired.
- Adaptive is **~2.3× faster than the dense baseline** and **~1.8× faster** than
  bricked on this dataset.
- Distance reads (20.3M/frame) are a real overhead but are far outweighed by the
  sample savings.
- Image parity with the dense baseline is preserved by construction: a sampled
  distance `D` guarantees every voxel within Chebyshev radius `D` is empty, and
  the `(D-1)` leap stays strictly inside that guaranteed-empty region, so no
  relevant voxel is ever skipped; material is sampled at the same density.

Note: the diagnostic counters use `unsigned int` and overflow for raw totals
over 400+ frames (~5.33B samples); always measure with ≤ 200 `--frames` for
reliable raw counter comparison (per-frame numbers stay consistent).

## Usage

```bash
# Adaptive-step mode
./build/bin/optix_app <path-to-dicom> --mode adaptive

# Adjust the relevance threshold that defines occupied voxels
./build/bin/optix_app <path-to-dicom> --mode adaptive --epsilon 0.02

# Exit after 200 frames to read the per-frame counter summary
./build/bin/optix_app <path-to-dicom> --mode adaptive --frames 200
```

## Status

- Adaptive mode (`--mode adaptive`, `--epsilon`, `--frames`, `dbgCounters`)
  builds and runs on all modes.
- Root cause of the original "slower than baseline, 0 leaps" behaviour
  diagnosed and fixed (uchar texture read mode); adaptive now measurably faster
  than the dense baseline and bricked on the CT dataset.
- `dbgCounters` / `--frames` kept in for thesis metrics; may be stripped later
  if not needed.
