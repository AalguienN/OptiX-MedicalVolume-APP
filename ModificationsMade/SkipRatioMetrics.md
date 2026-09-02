# Skip Ratio Metric across Traversal Strategies

Source material for the *Development* section of the thesis (measurement of empty-space
skipping effectiveness per strategy). Extends the `MetricsCollector` instrumentation
(`MetricsInstrumentation.md`) with a single per-frame **skip ratio** for every rendering
strategy.

## Objective

The thesis compares traversal strategies by how much empty space each avoids sampling.
A raw counter is not enough: a strategy must be measured against the *dense baseline* to
quantify what it skipped. The metric is defined per frame as

```
skip_ratio = 1 − vol_samples / total_steps
```

where **`total_steps`** (`[3]`) is the fixed-step count an unskipped, early-termination-free
dense march would take over the **full volume-AABB span of the ray**, counted once per ray
*before* any empty-space skipping. Using the full volume bounds as the single denominator
makes the ratio directly comparable across all strategies:

- `vol_samples ≈ total_steps`  → the strategy sampled almost everything the baseline would
  (little or no skipping);
- `vol_samples ≪ total_steps`  → the strategy avoided a large fraction of the volume.

## Design

### Device counters (`shared_device.h`)

One device buffer of four `unsigned int` counters, indexed as

| index | counter | semantics |
|---|---|---|
| `[0]` | vol_samples | voxels actually sampled and composited |
| `[1]` | dist_reads | distance-map texture reads (adaptive leap policy) |
| `[2]` | leaps | empty regions / empty subtrees / empty tiles skipped in one step |
| `[3]` | total_steps_bounds | would-be fixed steps over the full volume-AABB span (counted once per ray) |

Counters are **reset per frame** by `metrics.beginFrame()` (`cudaMemsetAsync` on the render
stream before the launch) and **snapshotted per frame** by `metrics.endFrame()`
(`cudaMemcpyAsync` into a ring-pool slot on the same stream, ordered before the slot's end
event). The snapshot is therefore known valid once the end event completes in
`advanceResultQueue()`, keeping the loop free of per-frame host/GPU sync (same design
principle as the timing measurement).

### Per-strategy accumulation

- **Dense baseline (manual / optix)**: `volumeMarch()` in `shared_device_programs.h` adds
  `[3] = maxSteps` once per ray over the full volume span; `[0]` counts the loop iterations
  (`[1] == [2] == 0`). The ratio reflects the baseline's early-termination savings only.
- **Standalone adaptive**: `spanMarchAdaptive()` adds `[0]`, `[1]`, `[2]`;
  `__raygen__rg_adaptive` adds `[3]` once per ray over the AABB span.
- **bricked-regions / octree-regions**: each BVH-hit of a non-empty brick/region is counted
  in `[2]` (`++nRegionSkips`); `[3]` is added once per ray over the AABB span; the
  intra-region `regionMarch` supplies `[0]` (and `[1]`, `[2]` with `--adaptive-march on`).
- **octree**: each empty subtree skipped (both at pop-time and at internal-node child-cull
  time) increments `[2]`; `[3]` once per ray in the closest-hit; `regionMarch` supplies
  `[0]`.
- **nanovdb**: each empty tile leap increments `[2]`; `[0]` samples; `[3] = maxSteps` once
  per ray.

### Host side (`metrics.h` / `metrics.cpp`)

- `enablePerFrameCounters(uint* dCounters, bool logBounds)` — wires the device buffer and
  whether to log the skip-ratio columns; called after the buffer is allocated and the
  metric's CSV path is known.
- `FrameSlot` gained `unsigned int counts[kNumCounters]` for the per-frame snapshot.
- `beginFrame()` / `endFrame()` perform the async reset / snapshot as above.
- `finalizeFrame(renderMs, counts)` writes the column group; the CSV header is now emitted
  lazily on the first row (it must reflect the flag-enabled columns, which are only known
  after `enablePerFrameCounters`).
- `skipRatio(samples, totalSteps)` clamps the denominator to the larger of the two and caps
  the result to `[0,1]` for degenerate counters.

### CLI (`app_main.cpp`)

- `--dbg-counter-bounds` — append `vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds`.
- Defaults **off** (zero logging overhead unless asked for).
- After the render loop the device counters (per-frame, last frame) are read back and
  printed with the skip ratio on stdout.

An earlier two-ratio design (`skip_ratio_interval` over the actually-marched span alongside
`skip_ratio_bounds`) was **dropped**: for the strategies that march the full volume span
(dense baseline, standalone adaptive, nanovdb) the two ratios coincided, and for the region
strategies the interval ratio was near-degenerate (little empty space remains inside a
marching span). The single bounds-based ratio is both comparable and information-bearing, so
the extra counter, CLI flag and CSV columns were removed.

## Bugs found and fixed during validation

### 1. `spanMarchFixed` had lost its sampling loop (region strategies rendered black)

**Symptom:** bricked-regions / octree / octree-regions reported `vol_samples = 0` (so
`skip_ratio = 1`, meaningless) *and* rendered black.

**Root cause:** during the `AdaptiveInnerMarchRegions.md` refactor (`regionMarch` dispatch),
the fixed-step `spanMarchFixed` body was reduced to just its diagnostics; the `for` loop that
samples and composites was dropped. Everything routed through it sampled nothing.

**Fix:** restored the fixed-step loop in `spanMarchFixed`.

**Result:** bricked-regions now reports 142,081,642 samples/frame, and its output matches the
dense baseline on non-black pixels (mean per-pixel RGB diff ≈ 0.5).

### 2. Octree empty-subtree skips were undercounted

**Symptom:** octree reported `leaps = 0` despite a bounds skip ratio ≈ 0.82.

**Root cause:** internal octree nodes cull their irrelevant children during gathering without
counting them; only stack entries popped later and found irrelevant incremented the leap
counter, which never happened since internal nodes never push irrelevant children.

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

CT dataset (`manifest-1787681973389_DICOM/series_1`), default view, 3 frames,
`--dbg-counter-bounds`:

| mode | vol_samples | dist_reads | leaps | total_steps_bounds | skip_ratio bounds |
|---|---|---|---|---|---|
| manual | 378,401,581 | 0 | 0 | 585,587,620 | 0.354 |
| optix | 378,401,581 | 0 | 0 | 585,587,620 | 0.354 |
| adaptive | 110,022,651 | 20,273,428 | 11,415,535 | 585,587,620 | 0.812 |
| bricked-regions | 142,081,642 | 0 | 5,162,304 | 585,587,620 | 0.757 |
| octree | 106,430,019 | 0 | 19,329,567 | 585,587,620 | 0.818 |
| octree-regions | 106,399,934 | 0 | 5,628,150 | 585,587,620 | 0.818 |
| nanovdb | 118,890,252 | 0 | 18,111,646 | 585,587,620 | 0.797 |

Interpretation: the dense baseline's ~0.35 ratio reflects early-ray-termination savings only;
standalone adaptive and nanovdb skip ~80% by empty-space leaping; the region/hierarchical
strategies avoid 76–82% of the volume bounds entirely via their region/subtree culling.
Because the denominator is the same full-volume span for every strategy, these numbers are
directly comparable — the headline cross-strategy result of the skip-ratio instrumentation.

## CSV output

```
frame,render_ms,fps,latency_ms,vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds
1,6.838,146.231,-1.000,142081642,0,5162304,585587620,0.757369
2,5.740,159.004,-1.000,142081642,0,5162304,585587620,0.757369
```

## Files changed

- `src/app/shared_device.h` — four-counter layout + comment (bounds at `[3]`).
- `src/app/device/shared_device_programs.h` — `volumeMarch()` adds `[3]` once per ray.
- `src/app/device/adaptive_programs.cu` — `[0]/[1]/[2]` in the span marchers, `[3]` once per
  ray in the raygen; **restored `spanMarchFixed` sampling loop**.
- `src/app/device/bricked_regions_programs.cu` / `octree_region_programs.cu` — `nRegionSkips`
  → `[2]`, `[3]` once per ray.
- `src/app/device/octree_programs.cu` — `[3]` in closest-hit; **leap counting at child-cull
  site**.
- `src/app/device/nanovdb_programs.cu` — `[0]/[2]` and `[3]` once per ray.
- `src/app/app_main.cpp` — `--dbg-counter-bounds`, 4-counter allocation + readback, bounds-only
  summary, usage text.
- `src/common/metrics.h` / `metrics.cpp` — `enablePerFrameCounters(dCounters, logBounds)`,
  per-frame snapshots, lazy CSV header, `skipRatio`.

## Status

- Per-frame skip-ratio CSV logging working for all modes, gated behind `--dbg-counter-bounds`.
- Three latent bugs (dead `spanMarchFixed`, octree leap undercount, summary mislabel) found
  during validation and fixed; region strategies render and count correctly.
- Changes not yet committed.