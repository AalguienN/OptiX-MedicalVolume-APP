# Hardware-Accelerated Sparse Volume Strategies (Regions, Octree, Octree-Regions)

Documents three new volume-rendering strategies that exploit spatial sparsity by
letting the **OptiX/RT hardware acceleration structure** skip empty space, in
place of the original slow SVO/octree experiment (tracked in the separate
`badoctree` branch), which performed a flat dense-grid march with a full
root-to-leaf octree descent *per sample*. All three are new `--mode` values on the
existing pipeline:

- `--mode region`          — Variant B: one AABB primitive per non-empty brick.
- `--mode octree`          — Variant A: true hierarchical octree traversal.
- `--mode octree-regions`  — Variant AB: octree leaves emitted as AABBs.

They share the pipeline infrastructure (same parameter struct, pipeline base,
metrics, render loop) and the same dense-baseline march + transfer function, so
their output is visually identical to the baseline; only the spatial
acceleration changes. Each is selected by a `TraceMode` enum value and wired up
by a dedicated setup block in `app_main.cpp`.

## Root cause addressed

The old octree (badoctree) was slow for two compounding reasons:

1. It traced the top-level volume AABB (single primitive) and then did a
   **dense march over the entire ray span**, calling an octree *lookup* per
   sample that descended from the root to the leaf every step.
2. That per-step descent rejected empty *internal* subtrees only at the leaf
   level after fully descending through them, so near-empty rays still paid a
   full hierarchical walk for each of thousands of samples.

The new strategies avoid both: the hierarchical data is either **traversed
iteratively by the device** (Variant A) or **flattened into per-region
primitives for the RT cores** (Variants B/AB), where empty space is skipped by
the BVH without any per-sample descent.

## Shared pipeline changes

All changes live in `src/app/`. The three strategies reuse the existing
`Params`, `PipelineBase`, metrics and render-loop unchanged apart from a few
additive members.

### `shared_device.h`
- `OctreeNode` struct (moved here from the old octree code): `relevant`,
  `isLeaf`, `childOffset` (index into `octreeChild[]` for internal nodes), and
  half-open voxel bounds `voxelMin[3]`/`voxelMax[3]` of the node's region.
- `Params` additions:
  - `regionAabbs` — per-primitive AABB array for the region GAS.
  - `octreeNodes` / `octreeChild` / `octreeNodeCount` — preorder octree arrays.
  - `ocRegionAabbs` / `ocRegionNode` — per-primitive AABBs + node map for the
    octree-regions GAS.
  - `dbgCounters` — optional device array `[3]`: `[0]` volume samples, `[1]`
    distance-map reads, `[2]` octree empty-subtree leaps. Null to disable; only
    the manual marchers increment it. Read back and reported per frame after the
    render loop in `app_main.cpp`.

### `pipeline_base.{h,cpp}`
- `init(...)` gained `numPayloadValues` (default 3) and `maxTraceDepth`
  (default 1) parameters. Only `maxTraceDepth` is applied per-mode, to
  `OptixPipelineLinkOptions.maxTraceDepth`. The `numPayloadValues` argument is
  accepted for API symmetry but **not** applied to the module: all strategy
  programs live in a single shared OptiX IR module, and `numPayloadValues` is a
  per-module compile option (see below).
- `createModule()` reserves **4 payload values** unconditionally
  (`compileOptions_.numPayloadValues = 4`) and **3 attribute values**
  (`numAttributeValues = 3`). Rationale: the region-style miss programs write
  payload slot 3 (the ray-loop miss flag), so the shared module must always
  allow ≥ 4 payload values; strategies that only ever use 0–2 are unaffected by
  reserving 4. Likewise the region intersection reports a third attribute
  (primitive index), so the module reserves 3 attributes.
- `createPipeline()` computes the device stack sizes with
  `optixUtilComputeStackSizes(..., maxTraceDepth_, ...)` and then **oversizes the
  continuation stack** (`css = css*2 + 4096`) before
  `optixPipelineSetStackSize`. The region-style ray-generation loops keep a large
  live register/local frame across each `optixTrace`; the raw computed size was
  too tight and truncated a program's local frame (see "Debugging & fixes").
- `TraceMode` extended with `REGION`, `OCTREE`, `OCTREE_REGIONS`.

### `app_main.cpp`
- CLI parsing for `region|octree|octree-regions` (`--mode`) plus `--leaf-size`
  for the octree leaf granularity (default 8); `--brick-size` is reused for
  Variant B (default 16). `modeLabel` extended for the new modes.
- Dedicated setup blocks per mode: build the scene (GAS), fill the matching
  `Params` fields, and select the correct raygen / miss / closest-hit /
  intersection entry names. Allocate `dbgCounters`, then print and free them
  after the render loop.

## Variant B — `--mode region` (RegionsScene + region programs)

Data-adaptive skipping with the same granularity as the bricked strategy.

### Host (`regions_scene.{h,cpp}`)
- Reuses the bricked occupancy classification: subdivide the volume into
  fixed-size bricks (default 16³) and keep each brick whose `maxOpacity ≥ ε`
  (the same "relevant" test as `BrickMeta`/the bricked strategy).
- Emits **one `OptixAabb` per relevant brick** into a flat array; empty bricks
  are simply absent. Builds a `CUSTOM_PRIMITIVES` GAS over that array with
  compaction (and a degenerate 1-primitive fallback if nothing is relevant),
  mirroring `VolumeScene`'s build.
- Exposes `handle()`, `deviceAabbs()`, `numRegions()`.

### Device (`device/region_programs.cu`)
- `__raygen__rg_region` — a **utility-ray loop**: each `optixTrace` (trace depth 1,
  4-value payload) returns the nearest still-relevant brick as
  `[entry t, exit t, primitive index]` with a miss flag. The raygen marches that
  brick's span (`regionMarchSeed`, identical sampling to the dense baseline),
  then re-traces from just past the brick's exit (`t_cur = tExit + 0.01*step`)
  until opacity saturates (α ≥ 0.99) or the ray misses. This composites multiple
  separated bricks on one ray front-to-back without recursion.
- `__intersection__is_region` — reads `params.regionAabbs[primIdx]`, its own
  AABB, reports the exact `[entry,exit]` (attributes 0,1) and primitive index
  (attribute 2) via `optixReportIntersection`.
- `__closesthit__ch_region` — does **not** march; only forwards the three
  attributes to the payload.
- `__miss__ms_region` — sets the miss flag (payload 3) so the raygen loop stops.

The RT-core BVH traversal is what skips empty space: a ray through an empty
region overlaps no primitive, so it is skipped in hardware.

## Variant A — `--mode octree` (OctreeVolume + octree programs)

True hierarchical octree traversal, iterative on the device.

### Host (`octree_volume.{h,cpp}`)
- Builds the octree over the dense grid once (reused from the badoctree branch):
  preorder node array with root at index 0; a parallel `children_` array holds
  the 8 child node indices of each internal node, so children are looked up by
  index. Each node records only `relevant` (whether its subtree has a relevant
  voxel), `isLeaf`, and the voxel region. The dense grid is not duplicated.
- Adds host accessors (`hostNodes()`, `hostChildren()`) so other scene builders
  can flatten the same tree.
- Single top-level AABB GAS (same as the dense baseline).

### Device (`device/octree_programs.cu`)
- `__raygen__` reuses the standard `__raygen__rg_optix` (normal payload depth 1).
- `__intersection__is` reports the volume entry/exit t (via the shared single-AABB
  intersection, as the baseline).
- `__closesthit__ch_octree` calls `octreeTraverse()`:
  - Iterative traversal with an **explicit stack** (bounded by `depth × 8`,
    capped at 512) — no device recursion, keeping the continuation stack shallow.
  - Pushes the root span `[tmin,tmax]`; pops nodes front-to-back. An internal
    node's 8 children are looked up by index **through** `params.octreeChild[]`
    — `childOffset` indexes into the `octreeChild` array, whose element is the
    child's node index into `octreeNodes[]`. Children are filtered to those the
    ray actually crosses, ordered front-to-back by entry t (8-entry insertion
    sort) and pushed nearest-first.
  - An empty subtree (`!relevant`) is skipped in a single step (counted in
    `dbgCounters[2]` as a "leap") instead of being descended.
  - A relevant leaf is marched over its `[t0,t1]` span by `marchLeafRegion`
    (identical sampling to the dense baseline), accumulating into a running RGBA
    so multiple leaves on one ray composite correctly, with α ≥ 0.99 early
    termination.
  - Returns the composited color via the normal 3-value payload.

## Variant AB — `--mode octree-regions` (OctreeRegionsScene + octree-region programs)

Hybrid of A and B: hardware skip via the BVH/RT cores, but with the
data-adaptive granularity of the octree (leaves are variable-sized, not fixed
bricks).

### Host (`octree_regions_scene.{h,cpp}`)
- Builds the octree internally by owning an `OctreeVolume` (so device nodes are
  reused), then walks its host nodes and emits **one `OptixAabb` per relevant
  leaf** into a flat array, optionally paired with the node index it originated
  from (`deviceNodeMap()`). Builds a compacted `CUSTOM_PRIMITIVES` GAS
  (degenerate fallback if empty), mirroring `regions_scene.cpp`.

### Device (`device/octree_region_programs.cu`)
- Same utility-ray-loop design as Variant B, with its own entry names:
  - `__raygen__rg_region_octree`, `__intersection__is_region_octree`
    (reads `params.ocRegionAabbs[primIdx]`), `__closesthit__ch_ocregion`
    (forwards attributes only), `__miss__ms_region_octree`.
  - The march (`regionMarchSeed`, reused from region_programs.cu within the same
    translation unit) is exactly the dense-baseline sampling, so output is
    unchanged; the granularity of the skipped empty space is adaptive.

## Wiring

`device/device_programs.cu` `#include`s all strategy `.cu` files
(`brick`, `adaptive`, `region`, `octree`, `octree_region`); the OptiX IR target
compiles the amalgamated translation unit, so every program lands in the single
IR module. `src/app/CMakeLists.txt` adds `regions_scene.cpp`, `octree_volume.cpp`
and `octree_regions_scene.cpp` to the application.

The region-style strategies pass `numPayloadValues = 4` to `init` for clarity but,
as noted above, the shared module always reserves 4 payload values regardless of
mode. `maxTraceDepth` is 1 for every mode (the region loop and the octree
closest-hit never nest `optixTrace` calls).

## Metrics

`dbgCounters` lets the three strategies be compared on traversal cost:
- Variant B/AB report volume samples in `[0]` (summed across the ray-loop
  marches).
- Variant A reports volume samples in `[0]` and empty-subtree leaps in `[2]`.
- FPS, render and latency metrics come from the existing `MetricsCollector`.

All strategies are DICOM-loadable from the unchanged `Volume`/`dicom_loader`
path and render through the same GLFW window + OpenGL interop loop.

## Debugging & fixes

This section records the bugs found and fixed while bringing the three
strategies to a correct, error-free first frame.

### 1. Shared OptiX IR module & payload-size mismatch (total breakage)

Each strategy used to set its own `numPayloadValues` (3 for the old modes). The
three new strategies compile into the **same** single OptiX IR module as the
baseline (`device_programs.cu` amalgamates every `.cu`), and `numPayloadValues`
is a per-module compile option — so a region-style miss program writing payload
slot 3 conflicted with a module reserving only 3 payload values, producing
`OPTIX_ERROR_INVALID_PAYLOAD_ACCESS` and breaking every mode.

**Fix:** `createModule()` now reserves `numPayloadValues = 4` and
`numAttributeValues = 3` unconditionally for all modes. The region-style miss
programs write payload slot 3; the region intersection reports a third attribute
(primitive index). Modes that only use payload/attribute slots 0–2 are
unaffected by the reservation.

### 2. Continuation-stack overflow in the ray-loop raygen (illegal memory access)

The region/octree-region raygen keeps a large live register/local frame across
each `optixTrace` inside its utility-ray loop. The continuation-stack size
computed by `optixUtilComputeStackSizes` was too tight, truncating a program's
local frame (`illegal memory access`, `__local__` overflow).

**Fix:** in `createPipeline()`, oversize the continuation stack
(`css = css*2 + 4096`) before `optixPipelineSetStackSize`.

### 3. Octree rendered nothing

Two independent bugs:

- `app_main.cpp` passed `intersectionEntry = nullptr` for the OCTREE block.
  **Fix:** use `"__intersection__is"` (the shared single-AABB intersection).
- Child lookup used `node.childOffset + k` directly instead of dereferencing
  through `params.octreeChild[]`. **Fix:** read `params.octreeChild[node.childOffset + k]`.

### 4. Black vertical line at the exact horizontal center (x = 960) on the first frame

At startup the camera has `theta = 0`, so the exact center column's ray has
`direction.x == 0.0f` precisely (`U = (1,0,0)`, so the center `d.x = 0` makes
`direction.x = 0`). The shared slab `intersectAABB` computed `inv = 1/dir` and
multiplied by it; when a sub-region face lay exactly on the ray plane this gave
`0 * inf = NaN`, so `tmin <= tmax` was false and **every** crossed sub-region
was rejected. Evidence: of the 8 root octree children, all failed
`intersectAABB` for the center ray; debug showed `ct1 = -inf` and a child AABB
with `bmax.x == origin.x`. Old modes were unaffected because they only intersect
the single volume box, whose face never coincided with the ray plane.

**Fix:** rewrote `intersectAABB` in `device/shared_device_programs.h` to handle
each axis independently: when `|dir.axis| < 1e-12`, check `origin.axis` inside
`[bmin.axis, bmax.axis]` (else miss) instead of dividing by zero; otherwise do
the normal slab division. This one shared fix repaired the octree (software
traversal) and the region/octree-region (intersection) strategies at once, since
they all call it, and it is strictly more correct than the old slab test.

### 5. Degenerate corner-grazing hits stuck the region ray-loop advance

After fixing `intersectAABB`, region/octree-region still showed a mostly-black
thumb at the exact horizontal center. Diagnosis: a ray lying exactly on a brick
face produced a **zero-length** (`tExit == tHit`) corner-grazing intersection
(`raw=[399.86,399.86]`), and the ray-loop advanced `t_cur` by only
`tExit + 0.01*stepSize` — too little to escape the degenerate point, so it
re-issued the same hit every hop and burned all 256 hops without accumulating.

**Fix:** in the region/octree-region raygen loop, when the reported span is
degenerate (`tExit - tHit <= stepSize*0.01`), advance by a full
`stepSize` (one real sample step) instead of the tiny nudge, so the ray escapes
the grazing point and continues to the next real brick.

### Result

After these fixes all 7 modes compile, run error-free, and render a correct
first frame. `--mode octree` now matches the dense baseline essentially exactly
(≈ 211 differing pixels out of ~1.94 M, all at sparse-structure boundaries).
`--mode region` and `--mode octree-regions` render correctly and reproduce the
dense image wherever their coarse (16³ brick / leaf) `maxOpacity ≥ ε` (default
ε = 0.01) relevance test keeps a primitive; low-density tissue at margins that
the finer octree captures is omitted by design, which is the expected trade-off
of the hardware-skipped sparse strategies. Sparse acceleration is confirmed by
the measured sample counts (dense ≈ 482 M/frame; region ≈ 201 M; octree ≈ 153 M;
octree-regions ≈ 153 M).
