# Metrics Instrumentation Implementation

Source material for the *Development → Metrics Instrumentation Implementation* section of the thesis
(`tfm.tex:1102`). Implements the performance-metric collection defined in *Evaluation Criteria*
(`tfm.tex:596`) and designed in *Metrics Instrumentation* (`tfm.tex:787`), as a **shared tool used by
all rendering strategies** so the instrumentation does not need to be rewritten per algorithm.

## Objective

Collect the performance metrics the thesis uses to compare rendering strategies against a common,
consistent basis:

- **Frame rate (FPS)** computed over a fixed observation window of `n` frames: `FPS = n / Σ tᵢ`.
- **Latency** between an input event and the frame that reflects it: `L = t_display − t_input`.
- **GPU memory usage**, queried once after construction and reported separately from the per-frame
  series (it does not vary during a session for the static volumes considered in this work).
- **Per-frame logging to disk** for offline aggregation and for the SSIM/PSNR visual-fidelity
  comparison described in the thesis.

All of this must be collected **without altering the rendering path**, so that instrumentation
overhead does not confound the comparison between strategies (`tfm.tex:787`).

## Design decisions

### Shared, strategy-agnostic host-side utility

A single `MetricsCollector` class lives in `src/common/` (already the shared include path for the
application). It is written once against the CUDA stream on which render launches occur and is
completely independent of the traversal strategy: any implementation (manual dense baseline, OptiX
trace, and future sparse strategies) that brackets its launch with `beginFrame()`/`endFrame()` and
reports input events through `onInputEvent()` gets the full metric set automatically.

Since all current and future strategies share the same render loop in `app_main.cpp`, the
instrumentation is wired in exactly once, in that shared loop, and applies to every mode.

### Non-blocking per-frame GPU timing (event ring pool)

Per-frame render time is obtained with CUDA events bracketing each OptiX launch
(`cudaEventRecord` start before, end after). To avoid a per-frame host/GPU synchronization that
would stall the render loop and distort the very measurement being taken (and to honour the
no-per-frame-host-round-trip requirement, `tfm.tex:812`), timing uses a **ring pool of 32 start/end
event pairs**.

- Each frame records a start event just before the launch and an end event just after it.
- Completed frames are read back **asynchronously** with `cudaEventQuery`, one or more frames later,
  and finalized with `cudaEventElapsedTime`.
- `flush()`, called after the render loop (`cudaStreamSynchronize` + drain), ensures the final
  in-flight frames are not lost when the session ends.

This yields per-frame GPU kernel time without pacing the loop on the host.

### Latency pairing

Input callbacks call `metrics.onInputEvent()`, which stores a `steady_clock` timestamp. The next
frame that is finalized after that input becomes the "resulting frame"; the latency is the elapsed
time from the input timestamp to that frame's presentation. Latency is logged per measured event.

### Memory usage

`cudaMemGetInfo` is used to compute device memory in use, recorded once per configuration after the
pipeline, volume, and parameters have been constructed and uploaded.

### Logging

When enabled, the per-frame series is appended to a CSV file (`frame, render_ms, fps, latency_ms`)
with a header written on first run, and `fflush`-ed after every row so data reaches disk
continuously during the session — not only at shutdown. A human-readable summary is also printed to
stdout at exit.

## Files changed

### `src/common/metrics.h` (new)

`class MetricsCollector`:

- `init(cudaStream_t, const char* logPath = nullptr, unsigned int windowFrames = 120)`
- `beginFrame()` / `endFrame()` — bracket the OptiX launch.
- `onInputEvent()` — timestamp an input event for latency.
- `flush()` — synchronize and drain pending frames after the loop.
- `recordGpuMemory()` / `static totalGpuMemory()`
- Accessors: `latestRenderMs`, `latestFps`, `latestLatencyMs`, `meanRenderMs`, `maxRenderMs`,
  `minRenderMs`, `meanFps`, `numFramesRecorded`, `gpuMemoryBytes`, `hasLatency`.
- `writeReport(const char*)` — optional human-readable report.

### `src/common/metrics.cpp` (new)

Implementation of the above, including the event ring pool, the FPS-window computation
(`FPS = n / Σtᵢ` over the last `n` finalized frames), latency pairing, memory query, and CSV logging.

### `src/app/app_main.cpp`

- Added `#include "metrics.h"`.
- Created `MetricsCollector metrics;` once after the OptiX context is initialized and initialized it
  with the render stream, an optional CSV path, and the observation window.
- Wired `metrics.beginFrame()` / `metrics.endFrame()` around the `optixLaunch` in the shared render
  loop — this single integration point covers **both** the manual and OptiX modes.
- Input callbacks (mouse button, cursor drag, scroll) now call `metrics.onInputEvent()`.
- `metrics.recordGpuMemory()` after construction, before the render loop.
- `metrics.flush()` plus a per-mode summary print after the loop.
- Added CLI options:
  - `--metrics <path>` — append the per-frame CSV series to `<path>`.
  - `--window <n>` — FPS observation window in frames (default 120).

### `src/app/CMakeLists.txt`

- Added `../common/metrics.cpp` to the `optix_app` executable sources.

## Usage

```bash
# Manual dense baseline, logging per-frame series to CSV with the default 120-frame window
./build/bin/optix_app <path-to-dicom> --mode manual --metrics results_manual.csv

# OptiX trace strategy with a custom 60-frame observation window
./build/bin/optix_app <path-to-dicom> --mode optix --metrics results_optix.csv --window 60
```

At the end of each run a summary is printed to stdout, e.g.:

```
Metrics summary (optix mode):
  frames   : 723
  render ms: mean=12.401 min=11.901 max=48.330
  fps      : mean=80.643 latest=80.817
  gpu mem  : 1536 bytes
  latency  : latest=4.213 ms
```

## CSV output

```
frame,render_ms,fps,latency_ms
1,12.703,78.723,-1.000
2,12.441,79.598,-1.000
...
```

The `latency_ms` column holds the measured value for frames that follow an input event and `-1.000`
otherwise.

## Design principles compliance

- **Common rendering pipeline shared by all strategies** (`tfm.tex:671`): instrumentation lives in
  the one shared render loop; no strategy duplicates it.
- **Instrumentation without altering the render path** (`tfm.tex:787`): timing is read back
  asynchronously so the loop is not paced by host/GPU synchronization.
- **GPU-oriented execution** (`tfm.tex:665`): no host copies of image data; only lightweight timing
  and logging metadata touches the host.
- **Per-frame series logged for variance/worst-case reporting** (`tfm.tex:619`): per-frame
  render time, FPS, and latency are all stored/logged, enabling not only averages but also variance
  and worst-case behavior to be reported.

## Status

- Both `--mode manual` and `--mode optix` instrumented through the shared loop.
- Build verified (`make` clean).
- Runtime validation against live DICOM data pending (requires a display session with a real
  dataset); the metric plumbing itself compiles and is independent of the dataset.
