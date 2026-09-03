# Automated Test Harness for Cross-Strategy / Cross-Dataset Evaluation

Source material for the *Methodology / Evaluation* section of the thesis. Provides an automated
way to run every rendering strategy across every TCIA dataset with a broad sweep of parameter
dimensions, and to record, per dataset, its **size on disk** and its **rendering-relevant
sparsity**.

Builds on the single-strategy instrumentation (see `MetricsInstrumentation.md` and
`SkipRatioMetrics.md`) by automating the matrix of runs those PDFs describe being run by hand.

## Objective

- Run all rendering strategies across *all* TCIA datasets with varied parameters covering the
  main parameter dimensions, saving per-run metrics CSVs and PPM snapshots.
- Report two dataset-level properties that contextualise the per-strategy timings:
  - **Size on disk** of the DICOM series; and
  - **Rendering-relevant sparsity** — the fraction of the volume whose transfer-function
    opacity is below the `epsilon` threshold (i.e. what the sparse strategies can skip).

## Script: `run_tests.sh`

### Dataset discovery (dynamic)

Datasets are not hard-coded. The script scans `$HOME/Documents/TCIA` (override with the
`TCIA_ROOT` env var) for any `<manifest>*/series_*` directory and runs the whole matrix on
whatever it finds. Adding a new manifest or series requires no script change.

### Output location (brother folder)

All outputs go to a folder **next to the project root** (i.e. a sibling of `optix_clone/`),
`../optix_test_results/`, never inside the repo. Layout per dataset:

```
optix_test_results/
└── <manifest>__<series>/
    ├── dataset_info.csv          # disk size (dataset level, static)
    ├── sparsity.csv              # measured rendering-relevant sparsity per strategy
    ├── <config>.csv              # per-frame metrics (frame,render_ms,fps,latency_ms,...)
    ├── <config>.ppm              # last-frame PPM snapshot
    └── <config>.log              # stdout+stderr of that run
```

### Parameter coverage (34 configurations)

Fixed configurations (default epsilon `0.01`):

| configuration | mode + args |
|---|---|
| `manual` | manual |
| `optix` | optix |
| `bricked-regions_brick8` | bricked-regions `--brick-size 8` |
| `bricked-regions_brick32` | bricked-regions `--brick-size 32` |
| `octree_leaf4` | octree `--leaf-size 4` |
| `octree_leaf8_adapt` | octree `--leaf-size 8 --adaptive-march on` |
| `octree-regions_leaf4` | octree-regions `--leaf-size 4` |
| `octree-regions_leaf8_adapt` | octree-regions `--leaf-size 8 --adaptive-march on` |
| `nanovdb_trilinear` | nanovdb `--nanovdb-sampler trilinear` |

Epsilon sweep — `{0.01, 0.05, 0.1, 0.2, 0.5}` for each of the five strategies that consume it:

| mode | mode + sweep args |
|---|---|
| adaptive | adaptive `--epsilon $eps` |
| bricked-regions | bricked-regions `--brick-size 16 --epsilon $eps` |
| octree | octree `--leaf-size 8 --epsilon $eps` |
| octree-regions | octree-regions `--leaf-size 8 --epsilon $eps` |
| nanovdb | nanovdb `--nanovdb-sampler nearest --epsilon $eps` |

Redundant configurations were folded into the sweep: plain `adaptive`, `bricked-regions_brick16`,
`octree_leaf8`, `octree-regions_leaf8`, `nanovdb_nearest` and the old `*_eps01` variants are all
covered by the `0.01` sweep entries (since `0.01` is the binary default). With all 47 datasets
this yields 34 × 47 = **1,598 runs**.

### CLI flags

- `./run_tests.sh` — run everything.
- `--dry-run` — print commands without executing.
- `--resume` — skip any configuration whose CSV already exists and has content (failed runs get
  a `FAILED` marker file that also counts as existing, so a later `--resume` skips them).
- `--frames N` — frames per run (default 200).

## Sparsity report (C++ change)

### Definition

A voxel is *rendering-relevant* iff its transfer-function opacity `>= epsilon`, using exactly the
same classification every sparse strategy applies (see `adaptive_grid_marcher.cpp`,
`bricked_regions_scene.cpp`, `octree_volume.cpp`, `nanovdb_volume.cpp`):

```
tf_t  = (scalar − scalarMin) * (1/(scalarMax − scalarMin)) * 2047
idx   = clamp(rint(tf_t), 0, 2047)
relevant = (tf.lut[idx].w >= epsilon)
sparsity = 100 * (1 − relevantCount / totalVoxels)
```

### Implementation (`src/app/app_main.cpp`)

Inserted after the scalar range is established (right before the per-strategy setup). Computed
host-side once per invocation using the in-memory `volume.data`, the transfer function, and the
run's `brickEpsilon`. Because it is mode-independent, **every** strategy/epsilon run prints the
line, e.g.:

```
Volume stats: rendering-relevant (eps=0.1): 2474314/47972352 relevant, 94.8422% sparse
```

The script greps this line from each run's `.log` and records it (with that run's epsilon) into
`sparsity.csv`.

## Size-on-disk report (bash)

Pure filesystem stat — the total bytes of the `.dcm` files in the series directory:

```bash
disk_bytes=$(find "$series_path" -name '*.dcm' -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
```

Stored in `dataset_info.csv` alongside a human-readable form (`numfmt` → e.g. `92MiB`). Computed
once per dataset, independent of strategy/epsilon.

## Verification

Single dataset (`manifest-1787673532591_DICOM/series_1`, 183 slices), `--frames 5`, all 34
configurations, exit 0.

`dataset_info.csv`:

```
dataset,disk_bytes,disk_human
manifest-m_DICOM__series_1,96374036,92MiB
```

`sparsity.csv` (abridged — sparsity is a dataset+epsilon property, so all strategies at the same
epsilon agree):

```
strategy,epsilon,sparsity_pct
manual,0.01,59.805
adaptive_eps005,0.05,89.7467
adaptive_eps010,0.1,94.8422
adaptive_eps020,0.2,96.8217
adaptive_eps050,0.5,98.7755
```

The sparsity grows monotonically with epsilon, as expected (a higher opacity threshold classifies
more voxels as non-relevant). The direct binary run printed the same `94.8422` at `eps=0.1`
independently, confirming the capture is faithful.

## Files changed

- `run_tests.sh` — new automated test harness (dataset discovery, strategy/epsilon sweep,
  per-run CSV/PPM/log, disk-size `dataset_info.csv`, sparsity capture, `--dry-run`/`--resume`/
  `--frames` flags).
- `src/app/app_main.cpp` — added the rendering-relevant sparsity% print for all modes.

## Status

- 34-configuration matrix runs cleanly to completion (verified on a single dataset; full 1,598-run
  sweep not yet executed).
- Sparsity and disk-size capture verified against direct binary output.
- Changes not yet committed.
