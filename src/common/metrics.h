#pragma once

// Shared metrics instrumentation for all volume-rendering strategies.
//
// This host-side utility collects the performance metrics defined in the
// thesis Evaluation Criteria (tfm.tex, sec:EvaluationCriteria) without
// altering the rendering path, so that instrumentation overhead does not
// confound comparisons between strategies. It is strategy-agnostic: any
// rendering implementation that brackets its OptiX launch with
// beginFrame()/endFrame() and reports input events through onInputEvent()
// can use it unchanged.
//
// Collected metrics:
//   - Per-frame GPU render time  (CUDA events bracketing each OptiX launch)
//   - FPS over a fixed observation window of N frames
//   - Latency (input event -> frame displayed)
//   - GPU memory usage (queried on demand, e.g. once after construction)
//
// Timing uses a ring pool of CUDA events so that a frame's elapsed time is
// read back asynchronously a few frames later, without forcing a per-frame
// host/GPU synchronization that would stall the render loop and distort the
// very measurement being taken.
//
// All per-frame series are appended to a CSV log file for offline
// aggregation and for the SSIM/PSNR visual-fidelity comparison described in
// the thesis.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

class MetricsCollector
{
public:
    MetricsCollector()                    = default;
    ~MetricsCollector();

    MetricsCollector(const MetricsCollector&)            = delete;
    MetricsCollector& operator=(const MetricsCollector&) = delete;

    // Initialize timing state. stream is the CUDA stream on which the
    // render launches are recorded. logPath, if non-empty, is where the
    // per-frame CSV series are appended (in append mode). windowFrames is
    // the fixed observation window n used for FPS computation.
    void init(cudaStream_t stream, const char* logPath = nullptr, unsigned int windowFrames = 120);

    // Enable per-frame traversal-counter logging (the dbgCounters device
    // array of 4 unsigned ints used by the renderers, see shared_device.h).
    // dCounters is reset to zero on the GPU at each frame start and
    // snapshotted to host memory right before that frame's end event, so the
    // readback is ordered on the stream without a host/GPU sync. When enabled
    // (logBounds == true), the skip-ratio column pair is appended to the CSV:
    //   vol_samples..total_steps_bounds,skip_ratio_bounds
    void enablePerFrameCounters(unsigned int* dCounters, bool logBounds);

    // Record a start timestamp immediately before the caller's OptiX
    // launch on the configured stream.
    void beginFrame();

    // Record an end timestamp immediately after the caller's OptiX launch.
    // The elapsed render time for the frame is read back asynchronously
    // (non-blocking); it becomes visible through latestRenderMs()/latestFps()
    // once the GPU has completed it.
    void endFrame();

    // Must be called immediately after an input event is registered (e.g.
    // inside a GLFW callback). The recorded timestamp is paired with the
    // frame that is displayed after the input took effect, producing the
    // latency for that interaction.
    void onInputEvent();

    // Synchronize with the GPU and finalize any frames whose timestamps have
    // not yet been read back. Call after the render loop to ensure the
    // summary reflects every rendered frame.
    void flush();

    // Query current GPU memory and record it (reported separately from the
    // per-frame series, since it does not vary during a session for the
    // static volumes considered in this work).
    void recordGpuMemory();

    // Convenience: total GPU VRAM in bytes (cudaMemGetInfo).
    static size_t totalGpuMemory();

    // Accessors
    double        latestRenderMs() const   { return latestRenderMs_; }
    double        latestFps() const        { return latestFps_; }
    double        latestLatencyMs() const  { return latestLatencyMs_; }
    double        meanRenderMs() const;
    double        meanFps() const;
    double        maxRenderMs() const;
    double        minRenderMs() const;
    unsigned int  numFramesRecorded() const { return static_cast<unsigned int>(renderTimesMs_.size()); }
    size_t        gpuMemoryBytes() const   { return gpuMemoryBytes_; }
    bool          hasLatency() const       { return hasLatency_; }

    // Write the full recorded series plus summary statistics to a human
    // readable report. Returns false on failure.
    bool writeReport(const char* path) const;

private:
    static constexpr unsigned int kEventPoolSize = 32;
    static constexpr unsigned int kNumCounters   = 4;

    struct FrameSlot
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t end   = nullptr;
        bool        used  = false;
        unsigned int counts[kNumCounters];   // filled when counters enabled
    };

    void advanceResultQueue();
    void finalizeFrame(double renderMs, const unsigned int* counts);
    void writeCsvHeader();
    static double skipRatio(unsigned int samples, unsigned int totalSteps);

    bool                           initialized_   = false;
    cudaStream_t                   stream_        = nullptr;

    FrameSlot                      pool_[kEventPoolSize] = {};
    unsigned int                   writeIndex_    = 0;
    unsigned int                   readIndex_     = 0;

    unsigned int                   windowFrames_  = 120;
    double                         latestRenderMs_ = 0.0;
    double                         latestFps_     = 0.0;
    double                         latestLatencyMs_ = 0.0;
    bool                           hasLatency_    = false;
    size_t                         gpuMemoryBytes_ = 0;

    std::chrono::steady_clock::time_point pendingInputTs_{};
    bool                           inputPending_  = false;

    std::vector<double>            renderTimesMs_;
    std::vector<double>            fpsHistory_;
    std::vector<double>            latencyMs_;

    std::FILE*                     logFile_       = nullptr;
    std::string                    logPath_;
    bool                           headerWritten_ = true;   // false => emit header on first row

    unsigned int*                  dCounters_     = nullptr; // device counters (4 x uint)
    bool                           logBounds_     = false;   // skip_ratio_bounds columns
};
