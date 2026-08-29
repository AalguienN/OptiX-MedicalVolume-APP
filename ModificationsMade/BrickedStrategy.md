# Bricked / Tiled Volume Strategy

Documents the implementation of the bricked volume strategy for the thesis
(Section "Bricked / tiled volume" of `tfm.tex`). The volume is subdivided into
fixed-size bricks of `n³` voxels; each brick carries metadata used to skip
entire bricks that contain no rendering-relevant voxels during ray traversal.
Introduced as a third traversal mode via the `--mode bricked` CLI flag, on top
of the existing `manual|optix` dual-mode baseline.

## Objective

Exploit spatial sparsity in CT/MR volumes by subdividing the dense grid into
fixed-size bricks and skipping empty ones, so the renderer does not spend time
sampling voxels that contribute nothing to the final image under the transfer
function. Brick size `n` is a free parameter so the trade-off between metadata
overhead, traversal overhead, skip granularity, and unnecessary in-brick
marching can be measured against the dense baseline.

## Design

### Representation

Per the thesis, each brick of `n³` voxels carries a `BrickMeta`:

```
minScalar  float   minimum scalar value in the brick
maxScalar  float   maximum scalar value in the brick
minOpacity float   minimum opacity after transfer-function application
maxOpacity float   maximum opacity after transfer-function application
relevant   uint    = 1 if maxOpacity >= epsilon (contains rendering-relevant voxels)
```

- A voxel is *rendering-relevant* when its transfer-function opacity is ≥ ε
  (`αᵢ = T(sᵢ) < ε` → irrelevant, as defined in "Rendering-Irrelevant Voxels").
- Brick coordinates are `ceil(dim/n)` per axis; the last brick per axis may be
  partial. Metadata lives in a z-major linear array of
  `brickCount.x * brickCount.y * brickCount.z` entries.
- The voxel data itself stays in the dense 3D texture (the same one the
  baseline uses): a non-empty brick is marched as a sub-region of it, exactly
  like the baseline — "computed as a smaller dense representation the same way
  as the baseline".

### Traversal

The bricked mode uses the same **single-AABB GAS + `optixTrace`** pattern as the
OptiX dense baseline (`__raygen__rg_optix` + `__intersection__is`); the
accelerator only returns the volume entry/exit. Empty-space skipping happens in
software inside the closed-hit program:

1. `__closesthit__ch_brick` receives the volume entry/exit `t`-values from the
   intersection program's attributes.
2. `brickMarch()` steps with the *same* fixed step size
   (`min(spacing) * 0.5`, the baseline's step) and, at each step, computes the
   brick containing the sample position.
3. If that brick's metadata marks it empty (`relevant == 0`), the entire brick
   is skipped in one step by advancing `t` to the brick's exit intersection
   along the ray; otherwise sampling/compositing proceeds identically to the
   baseline.

Because sampling within non-empty spans happens at the baseline's step grid
(diverging only after a skip crosses an empty brick, where no voxel contributes
anyway), the rendered image is visually identical to the dense baseline.

Pixel layout/indexing:

- world → brick index: `local = (p - volumeOrigin) / brickSize; brick = floor(local)`, clamped.
- brick → world bounds: `bmin = volumeOrigin + brick * brickSize; bmax = bmin + brickSize`.
- linear index (z-major, matches the host): `b.x + brickCount.x * (b.y + brickCount.y * b.z)`.

## Files changed

### New: `src/app/bricked_volume.h` / `.cpp`

Host-side construction of the representation:

- `build(volume, tf, brickSize, scalarMin, scalarMax, epsilon)` — computes the
  brick grid, scans each brick's voxels computing min/max scalar and
  min/max transfer-function opacity (re-using the baseline's LUT index
  formula `(s - scalarMin)/(scalarMax - scalarMin) * 2047` with round-to-
  nearest-even), sets `relevant = maxOpacity >= epsilon`, then uploads the
  `BrickMeta` array to GPU memory once.
- Exposes `brickDims()`, `brickCount()`, `brickSize()`, `deviceMeta()`,
  `numBricks()`, `numRelevantBricks()`, `metadataBytes()`.

### `src/app/shared_device.h`

- Added `struct BrickMeta` (host/device contract for the metadata).
- Extended `Params` with `brickMeta` (device pointer), `brickDims`,
  `brickCount`, `brickSize`. Added `#include <cuda_runtime.h>` so the header
  is self-contained (`cudaTextureObject_t`, `cudaArray`).

### `src/app/device/device_programs.cu`

- Added brick helpers: `brickFromPos()`, `brickLinearIndex()`,
  `brickWorldBounds()`.
- Added `brickMarch()` — fixed-step march with empty-brick skipping (step
  size, TF lookup, and front-to-back compositing identical to `volumeMarch`).
- Added `__closesthit__ch_brick` — reads entry/exit attributes, calls
  `brickMarch()`, packs the payload. Reuses `__raygen__rg_optix` and
  `__intersection__is` unchanged.

### `src/app/pipeline_base.h` / `.cpp`

- `enum class TraceMode` gained `BRICKED`.
- The intersection program is now embedded in the hitgroup for *any* non-manual
  mode (`mode != TraceMode::MANUAL && intersectionEntry`), previously it was
  only embedded for `OPTIX`.

### `src/app/app_main.cpp`

- New CLI options:
  - `--mode bricked` (third mode),
  - `--brick-size <n>` (voxels per brick edge, default 16),
  - `--epsilon <f>` (opacity relevance threshold, default 0.01).
- Bricked branch: builds a `VolumeScene` (single AABB GAS), a `BrickedVolume`,
  wires `__closesthit__ch_brick`, fills the brick fields of `Params`, and
  prints the brick grid + relevant-brick ratio + metadata size.
- Refactor: the duplicated per-frame render loop (resize → map PBO → upload
  `Params` → update raygen record → `optixLaunch` → unmap → present) and the
  metrics summary were extracted into shared helpers
  (`runRenderLoop()`, `printMetricsSummary()`); per-mode setup now lives in one
  place instead of in two large duplicated branches. Manual-mode debug stats
  (nonzero counts, center/origin voxel, middle slice) are preserved.

### `src/app/CMakeLists.txt`

- Added `bricked_volume.cpp` to the `optix_app` sources.

## Verification

CT lung dataset (512×512×90, HU [-1024, 1665]):

| mode          | brick size | bricks | relevant | metadata | FPS   |
|---------------|-----------|--------|----------|----------|-------|
| manual        | —         | —      | —        | —        | 144.8 |
| optix         | —         | —      | —        | —        | 140.7 |
| bricked       | 8         | 49152  | 21826    | 960 KiB  | 209.7 |
| bricked       | 16        | 6144   | 3318     | 120 KiB  | 201.3 |
| bricked       | 32        | 768    | 532      | 15 KiB   | 177.2 |

Fidelity vs the optix dense baseline (window capture, same camera): PSNR
60.6–61.1 dB on the CT dataset and 72.0 dB on an MR dataset — visually
identical output. As expected, the fraction of relevant bricks rises with brick
size (coarser skip granularity) while metadata overhead falls.

## Usage

```bash
# Bricked mode, 16³ voxel bricks
./build/bin/optix_app <path-to-dicom> --mode bricked --brick-size 16

# Smaller bricks = finer empty-space skipping (and more metadata)
./build/bin/optix_app <path-to-dicom> --mode bricked --brick-size 8

# Adjust the relevance threshold used to classify bricks as empty
./build/bin/optix_app <path-to-dicom> --mode bricked --brick-size 16 --epsilon 0.02

# With metrics logging for the performance comparison
./build/bin/optix_app <path-to-dicom> --mode bricked --metrics out.csv
```

## Status

- Bricked mode (`--mode bricked`, `--brick-size`, `--epsilon`): builds and runs.
- Verified on a CT (sparse) dataset and an MR (dense control) dataset; runs
  26–49% faster than the baseline on the sparse CT while matching its output
  (≥ 60 dB PSNR).
- Note: for the CT, `--epsilon` defaults to 0.01; lower values classify more
  bricks as relevant (less skipping), higher values skip more aggressively.