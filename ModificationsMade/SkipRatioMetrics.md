# Skip Ratio Metrics across Traversal Strategies

Source material for the *Development* section of the thesis (measurement of empty-space
skipping effectiveness per strategy). Extends the `MetricsCollector` instrumentation
(`MetricsInstrumentation.md`) with a per-frame **skip ratio** for every rendering strategy:
the fraction of traversal steps that are not actual voxel samples, split into two
complementary interpretations.

## Objective

The thesis compares traversal strategies by how much empty space each avoids sampling.
A single raw counter is not enough: a hardware/software strategy must be measured against
the *dense baseline* to quantify what it skipped. The skip ratio is defined per frame as

```
skip_ratio = 1 − vol_samples / total_steps
```

with two denominators, each logged independently:

- **`total_steps_interval`** (`[3]`): the fixed-step samples the strategy would have taken
  over the *marched interval* — the span it actually traversed (full volume for the dense
  baseline and standalone adaptive; only the interior of found regions for the region
  strategies).
- **`total_steps_bounds`** (`[4]`): the fixed-step samples over the full *volume-AABB span*
  for the ray, counted once per ray before any empty-space skip.

These two give a consistent comparison: the dense baseline and the standalone adaptive
march produce `[3] == [4]` (their interval *is* the volume bounds), so both ratios agree at
~0 for the baseline; the region/hierarchical strategies show a small `interval` ratio
(little empty space inside the regions they actually enter) but a large `bounds` ratio (the
whole regions they skipped are excluded from `[4]`).

## Design

### Device counters (`shared_device.h`)

One device buffer of five `unsigned int` counters, indexed as

| index | counter | semantics |
|---|---|---|
| `[0]` | vol_samples | voxels actually sampled and composited |
| `[1]` | dist_reads | distance-map texture reads (adaptive leap policy) |
| `[2]` | leaps | empty regions / empty subtrees / empty tiles skipped in one step |
| `[3]` | total_steps_interval | would-be fixed steps over the marched interval span |
| `[4]` | total_steps_bounds   | would-be fixed steps over the full volume-AABB span |

Counters are **reset per frame** by `metrics.beginFrame()` (`cudaMemsetAsync` on the render
stream before the launch) and **snapshotted per frame** by `metrics.endFrame()`
(`cudaMemcpyAsync` into a ring-pool slot on the same stream, ordered before the slot's end
event). The snapshot is therefore known valid once the end event completes in
`advanceResultQueue()`, keeping the loop free of per-frame host/GPU sync (same design
principle as the timing measurement).

### Per-strategy accumulation

- **Dense baseline (manual / optix)**: `volumeMarch()` in
  `shared_device_programs.h` adds `[3] = [4] = maxSteps` once per ray; `[0]` counts the
  loop iterations (`[1] == [2] == 0`). Both ratios equal the baseline's early-termination
  savings only.
- **Standalone adaptive**: `spanMarchAdaptive()` adds `[0]`, `[1]`, `[2]`, `[3]`;
  `__raygen__rg_adaptive` adds `[4] = maxSteps` once per ray.
- **bricked-regions / octree-regions**: each BVH-hit of a non-empty brick/region is counted
  in `[2]` (`++nRegionSkips`); `[4]` is added once per ray over the AABB span; the
  intra-region `regionMarch` supplies `[0]` and `[3]`.
- **octree**: each empty subtree skipped (both at pop-time and, after a counter bug fix, at
  internal-node child-cull time) increments `[2]`; `[4]` once per ray in the closest-hit;
  `regionMarch` supplies `[0]` and `[3]`.
- **nanovdb**: each empty tile leap increments `[2]`; `[0]` samples; `[3] == [4] = maxSteps`
  once per ray.

### Host side (`metrics.h` / `metrics.cpp`)

- `enablePerFrameCounters(uint* dCounters, bool interval, bool bounds)` — wires the device
  buffer and which of the two ratios to log; called after the buffer is allocated and the
  metric's CSV path is known.
- `FrameSlot` gained `unsigned int counts[kNumCounters]` for the per-frame snapshot.
- `beginFrame()` / `endFrame()` perform the async reset / snapshot as above.
- `finalizeFrame(renderMs, counts)` writes the two column groups; the CSV header is now
  emitted lazily on the first row (it must reflect the flag-enabled columns, which are only
  known after `enablePerFrameCounters`).
- `skipRatio(samples, totalSteps)` clamps the denominator to the larger of the two and caps
  the result to `[0,1]` for degenerate counters.

### CLI (`app_main.cpp`)

- `--dbg-counter-interval` — append `vol_samples,dist_reads,leaps,total_steps_interval,skip_ratio_interval`.
- `--dbg-counter-bounds` — append `vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds`.
- Both default **off** (zero logging overhead unless asked for); they can be combined.
- After the render loop the device counters (per-frame, last frame) are read back and printed
  with both skip ratios on stdout.

## Bugs found and fixed during validation

### 1. `spanMarchFixed` had lost its sampling loop (region strategies rendered black)

**Symptom:** bricked-regions / octree / octree-regions reported `vol_samples = 0` (so
`skip_ratio = 1`, meaningless) *and* rendered black. `[3]` was large but `[0]` was zero —
the span march was counting steps but never sampling.

**Root cause:** during the `AdaptiveInnerMarchRegions.md` refactor
(`regionMarch` dispatch), the fixed-step `spanMarchFixed` body was reduced to just the
counter block; the `for` loop that samples and composites was dropped. Everything routed
through it (all three region strategies) sampled nothing.

**Fix:** restored the fixed-step loop in `spanMarchFixed` (`compositeSample` + `++nSamples`
per step, matching the dense-baseline semantics).

**Result:** bricked-regions now reports 142,081,642 samples/frame, and its output matches the
dense baseline on non-black pixels (mean per-pixel RGB diff ≈ 0.5).

### 2. Octree empty-subtree skips were undercounted

**Symptom:** octree reported `leaps = 0` despite `bounds` skip ratio ≈ 0.82.

**Root cause:** internal octree nodes cull their irrelevant children during gathering without
counting them; only stack entries popped later and found irrelevant incremented the leap
counter — which never happened, since internal nodes never push irrelevant children.

**Fix:** increment `nLeaps` at the child-culling site in `octreeTraverse()`.

**Result:** octree now reports 19,329,567 leaps/frame.

### 3. Summary line misreported per-frame counters as "over N frames"

**Symptom:** the stdout summary divided the device-counter values by the frame count
("... over 5 frames (per frame: X/5)"), but the counters are reset per frame, so the readback
is already a *single frame's* totals. The CSV rows being identical across frames (static
scene) was misread as a snapshot bug; the values were always correct.

**Fix:** the summary now prints the readback directly as per-frame values without the
misleading division.

## Verification

CT dataset (`manifest-1787681973389_DICOM/series_1`), default view, 3 frames, both
`--dbg-counter-interval` and `--dbg-counter-bounds` on:

| mode | vol_samples | dist_reads | leaps | total_steps_interval | total_steps_bounds | skip_ratio interval | skip_ratio bounds |
|---|---|---|---|---|---|---|---|
| manual | 378,401,581 | 0 | 0 | 585,587,620 | 585,587,620 | 0.354 | 0.354 |
| optix | 378,401,581 | 0 | 0 | 585,587,620 | 585,587,620 | 0.354 | 0.354 |
| adaptive | 110,022,651 | 20,273,428 | 11,415,535 | 585,587,620 | 585,587,620 | 0.812 | 0.812 |
| bricked-regions | 142,081,642 | 0 | 5,162,304 | 148,264,644 | 585,587,620 | 0.042 | 0.757 |
| octree | 106,430,019 | 0 | 19,329,567 | 118,320,222 | 585,587,620 | 0.100 | 0.818 |
| octree-regions | 106,399,934 | 0 | 5,628,150 | 118,268,430 | 585,587,620 | 0.100 | 0.818 |
| nanovdb | 118,890,252 | 0 | 18,111,646 | 585,587,620 | 585,587,620 | 0.797 | 0.797 |

Interpretation: the dense baseline's ~0.35 ratio reflects early-ray-termination savings only;
standalone adaptive and nanovdb skip ~80% by empty-space leaping; the region/hierarchical
strategies skip little *inside* the regions they enter (interval ≈ 0.04–0.10) but avoid
75–82% of the volume bounds entirely via their region/subtree culling — exactly the 
complementary view the two ratios were designed to expose.

## CSV output

```
frame,render_ms,fps,latency_ms,vol_samples,dist_reads,leaps,total_steps_interval,skip_ratio_interval,vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds
1,6.838,146.231,-1.000,142081642,0,5162304,148264644,0.041702,142081642,0,5162304,585587620,0.757369
2,5.740,159.004,-1.000,142081642,0,5162304,148264644,0.041702,142081642,0,5162304,585587620,0.757369
```

## Files changed

- `src/app/shared_device.h` — counter layout extended to five indices (comment + `Params`
  buffer pointer unchanged).
- `src/app/device/shared_device_programs.h` — `volumeMarch()` adds `[3] == [4]`.
- `src/app/device/adaptive_programs.cu` — `spanMarchFixed`/`spanMarchAdaptive` add `[3]`;
  `__raygen__rg_adaptive` adds `[4]`; **restored `spanMarchFixed` sampling loop**.
- `src/app/device/bricked_regions_programs.cu` / `octree_region_programs.cu` — `nRegionSkips`
  → `[2]`, `[4]` once per ray.
- `src/app/device/octree_programs.cu` — `[4]` in closest-hit; **leap counting at child-cull
  site**.
- `src/app/device/nanovdb_programs.cu` — `[0]/[2]` and `[3] == [4]`.
- `src/app/app_main.cpp` — `--dbg-counter-interval` / `--dbg-counter-bounds`, buffer
  allocation + readback, stdout summary fix, usage text.
- `src/common/metrics.h` / `metrics.cpp` — `enablePerFrameCounters`, per-frame snapshots,
  lazy CSV header, `skipRatio`.

## Status

- Per-frame skip-ratio CSV logging working for all modes, gated behind the two flags.
- Three latent bugs (dead `spanMarchFixed`, octree leap undercount, summary mislabel)
  found during validation and fixed; region strategies render and count correctly.
- Changes not yet committed.