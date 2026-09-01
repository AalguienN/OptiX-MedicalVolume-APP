#include "metrics.h"
#include "check_macros.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace {

double fpsFromWindow(const std::vector<double>& renderTimesMs, unsigned int windowFrames)
{
    if (renderTimesMs.empty())
        return 0.0;

    const size_t n = static_cast<size_t>(windowFrames);
    const size_t count = std::min(n, renderTimesMs.size());
    const size_t offset = renderTimesMs.size() - count;

    double totalSeconds = 0.0;
    for (size_t i = offset; i < renderTimesMs.size(); ++i)
        totalSeconds += renderTimesMs[i] / 1000.0;

    // FPS = n / sum(t_i)  (thesis, sec:EvaluationCriteria)
    return (totalSeconds > 0.0) ? static_cast<double>(count) / totalSeconds : 0.0;
}

}  // namespace

MetricsCollector::~MetricsCollector()
{
    for (unsigned int i = 0; i < kEventPoolSize; ++i)
    {
        if (pool_[i].start)
            cudaEventDestroy(pool_[i].start);
        if (pool_[i].end)
            cudaEventDestroy(pool_[i].end);
    }
    if (logFile_)
        std::fclose(logFile_);
}

void MetricsCollector::init(cudaStream_t stream, const char* logPath, unsigned int windowFrames)
{
    stream_        = stream;
    windowFrames_  = windowFrames > 0 ? windowFrames : 1;
    initialized_   = true;
    latestFps_     = 0.0;
    latestRenderMs_ = 0.0;
    latestLatencyMs_ = 0.0;
    hasLatency_    = false;
    writeIndex_    = 0;
    readIndex_     = 0;

    for (unsigned int i = 0; i < kEventPoolSize; ++i)
    {
        CUDA_CHECK(cudaEventCreate(&pool_[i].start));
        CUDA_CHECK(cudaEventCreate(&pool_[i].end));
    }

    if (logPath && logPath[0] != '\0')
    {
        logPath_ = logPath;
        logFile_ = std::fopen(logPath, "a");
        if (logFile_)
        {
            // Write a header on the first row if the file is empty (i.e. this
            // is the first run); deferred so that enablePerFrameCounters() can
            // add its columns before the header is emitted.
            std::fseek(logFile_, 0, SEEK_END);
            headerWritten_ = (std::ftell(logFile_) != 0);
        }
    }
}

void MetricsCollector::enablePerFrameCounters(unsigned int* dCounters, bool interval, bool bounds)
{
    dCounters_   = dCounters;
    logInterval_ = interval;
    logBounds_   = bounds;
}

void MetricsCollector::beginFrame()
{
    if (!initialized_)
        return;

    // Allocate the next ring slot and timestamp the start of this frame's
    // launch. If the ring is full, retire the oldest unread frame.
    FrameSlot& slot = pool_[writeIndex_ % kEventPoolSize];
    if (slot.used)
        readIndex_ = (readIndex_ + 1) % kEventPoolSize;

    // Reset this frame's diagnostics on the GPU before the launch runs.
    // Ordered on the same stream, so it never races the previous frame's
    // counter snapshot (memcpy) issued in endFrame().
    if (dCounters_)
        CUDA_CHECK(cudaMemsetAsync(dCounters_, 0, sizeof(unsigned int) * kNumCounters, stream_));

    CUDA_CHECK(cudaEventRecord(slot.start, stream_));
    slot.used = true;
    writeIndex_ = (writeIndex_ + 1) % kEventPoolSize;
}

void MetricsCollector::endFrame()
{
    if (!initialized_)
        return;

    // Timestamp the end of this frame's launch. The most recently recorded
    // start corresponds to the slot just before the current write position.
    FrameSlot& slot = pool_[(writeIndex_ + kEventPoolSize - 1) % kEventPoolSize];

    // Snapshot this frame's diagnostics into the slot before recording the
    // end event: on the same stream and before slot.end, so the copy is known
    // complete once cudaEventQuery(slot.end) succeeds in advanceResultQueue().
    if (dCounters_ && slot.used)
        CUDA_CHECK(cudaMemcpyAsync(slot.counts, dCounters_,
                                   sizeof(unsigned int) * kNumCounters,
                                   cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaEventRecord(slot.end, stream_));

    advanceResultQueue();
}

void MetricsCollector::advanceResultQueue()
{
    // Read back the oldest completed frames asynchronously (non-blocking).
    unsigned int guard = 0;
    while (readIndex_ != writeIndex_ && pool_[readIndex_].used && guard++ < kEventPoolSize)
    {
        FrameSlot& slot = pool_[readIndex_];
        cudaError_t err = cudaEventQuery(slot.end);
        if (err == cudaErrorNotReady)
            break;  // GPU hasn't finished it yet; read it on a later frame
        CUDA_CHECK(err);

        if (!slot.end || !slot.start)
            break;

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, slot.start, slot.end));
        slot.used = false;
        readIndex_ = (readIndex_ + 1) % kEventPoolSize;

        finalizeFrame(static_cast<double>(elapsedMs),
                      (dCounters_ && (logInterval_ || logBounds_)) ? slot.counts : nullptr);
    }
}

void MetricsCollector::finalizeFrame(double renderMs, const unsigned int* counts)
{
    renderTimesMs_.push_back(renderMs);
    latestRenderMs_ = renderMs;
    latestFps_ = fpsFromWindow(renderTimesMs_, windowFrames_);
    fpsHistory_.push_back(latestFps_);

    using namespace std::chrono;
    const auto now = steady_clock::now();

    double latencyMs = -1.0;
    if (inputPending_)
    {
        latencyMs = duration_cast<microseconds>(now - pendingInputTs_).count() / 1000.0;
        latestLatencyMs_ = latencyMs;
        hasLatency_ = true;
        latencyMs_.push_back(latencyMs);
        inputPending_ = false;
    }

    if (logFile_)
    {
        if (!headerWritten_)
        {
            writeCsvHeader();
            headerWritten_ = true;
        }

        std::fprintf(logFile_, "%u,%.3f,%.3f,%.3f",
                     numFramesRecorded(), renderMs, latestFps_, latencyMs);
        if (counts)
        {
            const unsigned int samples = counts[0];
            if (logInterval_)
            {
                std::fprintf(logFile_, ",%u,%u,%u,%u,%.6f",
                             counts[0], counts[1], counts[2], counts[3],
                             skipRatio(samples, counts[3]));
            }
            if (logBounds_)
            {
                std::fprintf(logFile_, ",%u,%u,%u,%u,%.6f",
                             counts[0], counts[1], counts[2], counts[4],
                             skipRatio(samples, counts[4]));
            }
        }
        std::fprintf(logFile_, "\n");
        std::fflush(logFile_);
    }
}

void MetricsCollector::writeCsvHeader()
{
    if (!logFile_)
        return;
    std::fprintf(logFile_, "frame,render_ms,fps,latency_ms");
    if (logInterval_)
        std::fprintf(logFile_, ",vol_samples,dist_reads,leaps,total_steps_interval,skip_ratio_interval");
    if (logBounds_)
        std::fprintf(logFile_, ",vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds");
    std::fprintf(logFile_, "\n");
}

double MetricsCollector::skipRatio(unsigned int samples, unsigned int totalSteps)
{
    // skip_ratio = 1 - samples/total_steps, clamp the denominator to avoid
    // division by zero and cap the result to [0,1] for degenerate counters.
    const unsigned int denom = samples > totalSteps ? samples : totalSteps;
    if (denom == 0)
        return 0.0;
    double ratio = 1.0 - static_cast<double>(samples) / static_cast<double>(denom);
    return ratio < 0.0 ? 0.0 : (ratio > 1.0 ? 1.0 : ratio);
}

void MetricsCollector::onInputEvent()
{
    if (!initialized_)
        return;
    using namespace std::chrono;
    pendingInputTs_ = steady_clock::now();
    inputPending_   = true;
}

void MetricsCollector::flush()
{
    if (!initialized_)
        return;

    // Wait for all pending work on the stream, then read back every frame
    // that has not yet been finalized.
    cudaError_t syncErr = cudaStreamSynchronize(stream_);
    CUDA_CHECK(syncErr);

    for (unsigned int i = 0; i < kEventPoolSize; ++i)
    {
        FrameSlot& slot = pool_[readIndex_ % kEventPoolSize];
        if (!slot.used)
            break;

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, slot.start, slot.end));
        slot.used = false;
        readIndex_ = (readIndex_ + 1) % kEventPoolSize;

        finalizeFrame(static_cast<double>(elapsedMs),
                      (dCounters_ && (logInterval_ || logBounds_)) ? slot.counts : nullptr);
    }
}

void MetricsCollector::recordGpuMemory()
{
    if (!initialized_)
        return;
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    gpuMemoryBytes_ = totalBytes - freeBytes;
}

size_t MetricsCollector::totalGpuMemory()
{
    size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess)
        return 0;
    return totalBytes;
}

double MetricsCollector::meanRenderMs() const
{
    if (renderTimesMs_.empty())
        return 0.0;
    double sum = std::accumulate(renderTimesMs_.begin(), renderTimesMs_.end(), 0.0);
    return sum / static_cast<double>(renderTimesMs_.size());
}

double MetricsCollector::meanFps() const
{
    if (fpsHistory_.empty())
        return 0.0;
    double sum = std::accumulate(fpsHistory_.begin(), fpsHistory_.end(), 0.0);
    return sum / static_cast<double>(fpsHistory_.size());
}

double MetricsCollector::maxRenderMs() const
{
    if (renderTimesMs_.empty())
        return 0.0;
    return *std::max_element(renderTimesMs_.begin(), renderTimesMs_.end());
}

double MetricsCollector::minRenderMs() const
{
    if (renderTimesMs_.empty())
        return 0.0;
    return *std::min_element(renderTimesMs_.begin(), renderTimesMs_.end());
}

bool MetricsCollector::writeReport(const char* path) const
{
    if (!path || path[0] == '\0')
        return false;

    std::FILE* f = std::fopen(path, "w");
    if (!f)
        return false;

    std::fprintf(f, "MetricsCollector report\n");
    std::fprintf(f, "=======================\n");
    std::fprintf(f, "Frames recorded     : %u\n", numFramesRecorded());
    std::fprintf(f, "Observation window  : %u\n", windowFrames_);
    std::fprintf(f, "GPU memory used     : %zu bytes\n", gpuMemoryBytes_);
    std::fprintf(f, "\nRender time (ms):\n");
    std::fprintf(f, "  mean   : %.3f\n", meanRenderMs());
    std::fprintf(f, "  min    : %.3f\n", minRenderMs());
    std::fprintf(f, "  max    : %.3f\n", maxRenderMs());
    std::fprintf(f, "\nFPS:\n");
    std::fprintf(f, "  latest : %.3f\n", latestFps_);
    std::fprintf(f, "  mean   : %.3f\n", meanFps());
    if (hasLatency_)
    {
        std::fprintf(f, "\nLatency (ms):\n");
        std::fprintf(f, "  latest : %.3f\n", latestLatencyMs_);
        if (!latencyMs_.empty())
        {
            double sum = std::accumulate(latencyMs_.begin(), latencyMs_.end(), 0.0);
            std::fprintf(f, "  mean   : %.3f\n", sum / static_cast<double>(latencyMs_.size()));
            std::fprintf(f, "  max    : %.3f\n",
                         *std::max_element(latencyMs_.begin(), latencyMs_.end()));
        }
    }
    std::fprintf(f, "\nPer-frame series written to: %s\n",
                 logPath_.empty() ? "(none)" : logPath_.c_str());
    std::fclose(f);
    return true;
}
