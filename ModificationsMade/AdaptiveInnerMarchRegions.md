# Adaptive Inner March for the Region Strategies

Documents the extension of the adaptive-step march to the **sparse/hierarchical
strategies**: the octree / bricked-regions / octree-regions modes now use the
same Chebyshev distance-map stepping *inside* each traversed region (brick or
leaf) that the standalone `--mode adaptive` strategy uses across the whole
volume. The outer empty-space skipping of each strategy (RT-core BVH over
per-region AABBs, software octree subtree descent) is unchanged; the adaptive
leap compresses the empty space that remains *within* a region.

Gated behind the new `--adaptive-march <on|off>` CLI flag (default **off**, so
the basic fixed-step traversal remains the default). With the flag on, the
intra-region traversal is adaptive; with it off (default) the behaviour is
byte-identical to the previous fixed-step region marchers.

## Objective

A region (brick or octree leaf) is classified *rendering-relevant* if it
contains **at least one** relevant voxel — but it can still be mostly empty
inside (e.g. a 16³ brick around a blood vessel, or a 64³ leaf holding a small
structure). The RT cores / octree only skip *whole* empty regions, so the fixed
inner march currently samples through the intra-region empty space voxel by
voxel. Since the adaptive-step marcher's leap guarantee only depends on the
dense volume texture + the Chebyshev distance map, the same policy can be
reused inside a region span to skip that empty interior cheaply.

## Design

### Shared intra-region march

Previously each strategy carried its own copy of the fixed-step march
(`regionMarchSeed` in `bricked_regions_programs.cu`, `marchLeafRegion` in
`octree_programs.cu`), both verbatim copies of the dense baseline `volumeMarch`.
These are removed and replaced by a single gated function in
`adaptive_programs.cu`:

```
regionMarch(origin, dir, tmin, tmax, stepSize, &accumR, &accumG, &accumB, &accumA)
  ├── params.useAdaptive == 1  → spanMarchAdaptive(...)   (Chebyshev leap policy)
  └── params.useAdaptive == 0  → spanMarchFixed(...)      (dense-baseline step)
```

Both variants are *seedable*: they march a span and continue the already
accumulated RGBA, so multiple regions on one ray composite correctly (the same
contract the old `regionMarchSeed` had). The raygen loops of bricked-regions and
octree-regions and the closest-hit octree traversal simply call `regionMarch`.

Because `adaptive_programs.cu` is `#include`d from `device_programs.cu` before
the strategy files, the shared helpers are visible to all of them from the
single OptiX IR module.

### Refactor of the adaptive code

`adaptiveMarch()` was split into small reusable pieces (now also shared by the
region march):

- `voxelWorldStepFor(dir)` — the per-ray world time of one "safe voxel step".
- `sampleTexCoord(pos)` — world → clamped normalized texture coordinate.
- `compositeSample(texCoord, stepSize, accum…)` — volume sample, TF lookup and
  front-to-back compositing; returns the raw TF value.
- `adaptiveAdvance(t, texCoord, tfVal, stepSize, voxelWorldStep, …)` — the
  conditional distance-map read + `(D-1)` leap / fixed-step decision.
- `spanMarchFixed` / `spanMarchAdaptive` — seedable span marchers (fixed step and
  adaptive step respectively).
- `regionMarch` — the gated dispatcher above.
- `adaptiveMarch` — rewritten as `spanMarchAdaptive` over the full volume entry/
  exit span; the standalone `--mode adaptive` output is unchanged.

Correctness notes that carry over from the standalone adaptive strategy:

- The leap guarantee is preserved inside a region: a sampled distance `D ≥ 2`
  means every voxel within Chebyshev radius `D` is empty, so `(D-1)` safe voxel
  steps never cross a relevant voxel. Overshooting the region exit is harmless —
  the region raygen loops discard the final `t` and re-trace from the BVH/AABB
  exit.
- The distance map is read **only when the sample's TF opacity is `< ε`**, so
  the cost inside opaque material stays identical to the fixed march (`D=0`
  boundary cases fall back to `stepSize`).
- `Params.useAdaptive` gates the branch; it is zero unless the flag is on, so
  the default path is exactly the old fixed-step march.

## Files changed

### `src/app/shared_device.h`

- Added `unsigned int useAdaptive` to `Params` (gate for the shared region
  march). `distanceTex` / `epsilon` now serve both the standalone adaptive mode
  and the region strategies.

### `src/app/device/adaptive_programs.cu`

- Extracted the step policy into `voxelWorldStepFor`, `sampleTexCoord`,
  `compositeSample`, `adaptiveAdvance`.
- Added the seedable span marchers `spanMarchFixed` / `spanMarchAdaptive` and the
  gated dispatcher `regionMarch`.
- `adaptiveMarch()` / `__raygen__rg_adaptive` rewritten on top of
  `spanMarchAdaptive` (unchanged rendered output).

### `src/app/device/bricked_regions_programs.cu`

- Removed the duplicated fixed-step `regionMarchSeed`; the raygen loop now calls
  the shared `regionMarch`.

### `src/app/device/octree_programs.cu`

- Removed `marchLeafRegion`; `octreeTraverse()` now calls the shared
  `regionMarch` for each relevant leaf.

### `src/app/device/octree_region_programs.cu`

- Call site switched to `regionMarch` (inherits the gate with no extra device
  code).

### `src/app/app_main.cpp`

- New CLI flag `--adaptive-march <on|off>` (**default off**).
- `enableAdaptiveStep()` lambda builds the `AdaptiveGridMarcher` and wires
  `params.distanceTex` / `params.epsilon` / `params.useAdaptive` for the octree,
  bricked-regions and octree-regions modes when the flag is on.
- Usage text extended; the affected modes print `Adaptive inner march: on|off`.

## Verification

CT dataset (512×512×63, HU [-1024, 1351]), `--epsilon 0.01`, 10 frames, default
view. Belt-and-braces comparison of `--adaptive-march off` vs `on`:

### Default region sizes (brick 16³, octree/octree-regions leaf 8³)

| mode | off ms | on ms | off samples/frame | on samples/frame | dist-reads/frame | leaps/frame |
|------|--------|-------|-------------------|------------------|------------------|-------------|
| bricked-regions | 7.00 | **6.65** | 200,542,476 | 153,937,923 | 28,020,963 | 12,270,361 |
| octree | 10.58 | 11.13 | 148,409,577 | 142,471,573 | 15,914,996 | 3,450,371 |
| octree-regions | 8.35 | 8.82 | 148,330,524 | 142,412,867 | 15,924,147 | 3,419,865 |

With small 8³ leaves the regions are already so tight that there is little
intra-region empty space to skip, and the extra distance-read cost is not repaid
(octree / octree-regions get slightly slower). bricked-regions (16³ bricks) gains
~23% fewer samples and a small speedup.

### Larger regions expose the win (`--leaf-size 64`)

| mode | off ms | on ms | off samples/frame | on samples/frame | dist-reads/frame | leaps/frame |
|------|--------|-------|-------------------|------------------|------------------|-------------|
| octree | 7.17 | **4.99** | 287,040,773 | 155,382,118 | 29,730,845 | 14,553,639 |
| octree-regions | 7.38 | **5.65** | 286,850,128 | 155,529,047 | 30,005,840 | 14,570,242 |

Large regions are mostly empty inside, so the leap policy cuts samples ~46% and
render time ~30%, confirming the adaptive inner march helps most when regions are
big enough to contain substantial empty space.

### Visual fidelity

| comparison | mean |ΔR| | max |Δbyte| | bytes within 2 LSB |
|---|---|---|---|---|
| bricked-regions off vs on | 0.031 | 17 | 99.9% |
| octree off vs on | 0.019 | 15 | 100.0% |
| octree-regions off vs on | 0.019 | 15 | 100.0% |

The tiny residual difference is inherent to the adaptive semantics: samples with
TF opacity `< ε` (but above the `a > 0.001` compositing cutoff) are leapt over,
so a negligible amount of low-opacity contribution is skipped — identical to the
pre-existing standalone `--mode adaptive` behaviour. Visually imperceptible.

## Usage

```bash
# Default: fixed-step (basic) intra-region traversal
./build/bin/optix_app <path-to-dicom> --mode octree

# Enable the adaptive inner march for the region strategies
./build/bin/optix_app <path-to-dicom> --mode bricked-regions --adaptive-march on
./build/bin/optix_app <path-to-dicom> --mode octree --leaf-size 64 --adaptive-march on

# A/B comparison (metrics + snapshot)
./build/bin/optix_app <path-to-dicom> --mode bricked-regions --adaptive-march off --frames 200 --metrics brick_off.csv
./build/bin/optix_app <path-to-dicom> --mode bricked-regions --adaptive-march on  --frames 200 --metrics brick_on.csv
```

## Status

- The adaptive-step march is now the shared intra-region traversal for the
  octree / bricked-regions / octree-regions strategies, selectable via
  `--adaptive-march` (default **off**, preserving the previous fixed-step
  behaviour byte-for-byte).
- The standalone `--mode adaptive` is unchanged.
- Measured benefit is strongly dependent on region leaf size: small 8³ regions
  gain little or lose slightly; large regions show ~30% faster rendering with
  essentially no visual difference. This size-dependence is a useful result for
  the thesis comparison.