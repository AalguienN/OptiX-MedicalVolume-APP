#include "adaptive_grid_marcher.h"
#include "camera.h"
#include "check_macros.h"
#include "dicom_loader.h"
#include "display.h"
#include "gl_check.h"
#include "gl_window.h"
#include "interop_buffer.h"
#include "metrics.h"
#include "optix_context.h"
#include "pipeline_base.h"
#include "placeholder_scene.h"
#include "bricked_regions_scene.h"
#include "octree_volume.h"
#include "octree_regions_scene.h"
#include "nanovdb_volume.h"
#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"
#include "volume_scene.h"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

// Holds the per-window state so input callbacks can reach both the camera
// (to mutate the view) and the shared metrics collector (to timestamp the
// input event for latency measurement) without resorting to globals.
struct WindowUserData
{
    Camera*          camera  = nullptr;
    MetricsCollector* metrics = nullptr;
};

WindowUserData* userData(GLFWwindow* win)
{
    return static_cast<WindowUserData*>(glfwGetWindowUserPointer(win));
}

void glfwMouseButtonCallback(GLFWwindow *win, int button, int action, int /*mods*/) {
    auto* ud = userData(win);
    if (!ud) return;
    if (ud->metrics) ud->metrics->onInputEvent();
    if (ud->camera)  ud->camera->onMouseButton(button, action);
}

void glfwCursorPosCallback(GLFWwindow *win, double xpos, double ypos) {
    auto* ud = userData(win);
    if (!ud) return;
    if (ud->metrics && ud->camera && ud->camera->leftMouseDragging) ud->metrics->onInputEvent();
    if (ud->camera) ud->camera->onCursorPos(xpos, ypos);
}

void glfwScrollCallback(GLFWwindow *win, double /*xoffset*/, double yoffset) {
    auto* ud = userData(win);
    if (!ud) return;
    if (ud->metrics) ud->metrics->onInputEvent();
    if (ud->camera)  ud->camera->onScroll(yoffset);
}

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <path-to-dicom-series> [--mode manual|optix|adaptive]\n"
              << "\n"
              << "Options:\n"
              << "  --mode manual      Dense baseline: manual ray-AABB traversal in raygen (default)\n"
              << "  --mode optix       OptiX trace: AABB GAS + intersection program + optixTrace\n"
              << "  --mode adaptive    Adaptive-step grid marcher: Chebyshev distance map drives\n"
              << "                     the ray-march step size (software, no RT-core traversal)\n"
               << "  --mode bricked-regions  Bricked-regions: bricked volume but one GAS AABB per\n"
               << "                     non-empty brick; RT cores skip empty space in hardware\n"
              << "  --mode octree      SVO / Octree: hierarchical octree empty-subtree skipping\n"
              << "                     (software traversal, single top-level AABB)\n"
              << "  --mode octree-regions  Octree regions: octree leaves emitted as GAS AABBs so\n"
               << "                     RT cores skip empty space with data-adaptive granularity\n"
               << "  --mode nanovdb     NanoVDB: sparse GPU volume tree (NanoVDB) with built-in\n"
               << "                     empty-space skipping, sampled via the device accessor\n"
<< "  --brick-size <n>   Voxels per brick edge (n^3) for --mode bricked-regions\n"
               << "                     (default 16)\n"
               << "  --epsilon <f>      Opacity threshold classifying a voxel as rendering-\n"
               << "                     relevant for --mode bricked-regions / adaptive / nanovdb /\n"
               << "                     octree / octree-regions (default 0.01)\n"
               << "  --adaptive-march <on|off>  Use the adaptive-step (Chebyshev distance map)\n"
               << "                     inner march for the octree / bricked-regions /\n"
               << "                     octree-regions strategies (default off)\n"
                << "  --metrics <path>   Append per-frame performance series (CSV) to <path>\n"
                << "  --window <n>       FPS observation window in frames (default 120)\n"
<< "  --frames <n>       Exit after n frames (benchmarking; default: until closed)\n"
                 << "  --dbg-counter-bounds    Log per-frame skip_ratio_bounds columns to CSV\n"
                 << "                     (fraction of the volume-bounds avoided; default off)\n"
                 << "  --nanovdb-sampler <nearest|trilinear>  NanoVDB interpolation (default nearest)\n"
                << "  --snapshot <path>  Write the last rendered frame to <path> as a PPM image\n"
                << "  --summary <path>   Append one aggregate metrics row to <path> (CSV) on exit\n"
                << "  --dataset <label>  Dataset label for the --summary row (default: derived)\n"
                << "  --strategy <label> Strategy/config label for the --summary row (default: mode)\n"
                << "  --disk-bytes <n>   Series size on disk (bytes) for the --summary row\n"
                 << "  --build-time       Time the load-time construction of the strategy and\n"
                 << "                     record it as build_time_ms in the aggregate summary row\n";
}

// Shared per-frame render loop. The Params struct is wired once before this
// call; only the image pointer and its dimensions change across frames, so
// they are updated here and uploaded with a single async host->device copy.
void runRenderLoop(GlWindow& window, Display& display, InteropBuffer& interopBuffer,
                   OptixContext& optixContext, PipelineBase& pipeline,
                   MetricsCollector& metrics, Camera& camera,
                   Params& params, CUdeviceptr d_params, unsigned int maxFrames,
                   const char* snapshotPath)
{
    unsigned int frame = 0;
    while (!window.shouldClose() && (maxFrames == 0 || frame < maxFrames))
    {
        ++frame;
        glfwPollEvents();

        int fb_w, fb_h;
        window.framebufferSize(fb_w, fb_h);
        if (params.image_width != static_cast<unsigned int>(fb_w) ||
            params.image_height != static_cast<unsigned int>(fb_h))
        {
            unsigned int width  = static_cast<unsigned int>(fb_w);
            unsigned int height = static_cast<unsigned int>(fb_h);
            interopBuffer.resize(width, height);
            display.resize(width, height);
            camera.aspect = static_cast<float>(width) / static_cast<float>(height);
            params.image_width  = width;
            params.image_height = height;
            glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        }

        float3 cam_U, cam_V, cam_W;
        camera.uvw(cam_U, cam_V, cam_W);

        uchar4 *d_pixels = interopBuffer.map(optixContext.stream());
        params.image = d_pixels;

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &params,
                                   sizeof(Params), cudaMemcpyHostToDevice,
                                   optixContext.stream()));

        RayGenData rgData;
        rgData.cam_eye   = camera.eye();
        rgData.camera_u  = cam_U;
        rgData.camera_v  = cam_V;
        rgData.camera_w  = cam_W;
        pipeline.updateRayGenRecord(optixContext.stream(), rgData);

        metrics.beginFrame();
        OPTIX_CHECK(optixLaunch(pipeline.pipeline(), optixContext.stream(),
                                d_params, sizeof(Params), pipeline.sbt(),
                                params.image_width, params.image_height, 1));
        metrics.endFrame();

        interopBuffer.unmap(optixContext.stream());

        display.present(interopBuffer.pbo());

        window.swapBuffers();
    }

    metrics.flush();

    // Optional last-frame dump (PPM P6) for offline visual-fidelity checks
    // against the dense baseline (Section "Visual Fidelity" of the thesis).
    if (snapshotPath && snapshotPath[0])
    {
        uchar4* d_pixels = interopBuffer.map(optixContext.stream());
        std::vector<uchar4> host(params.image_width * params.image_height);
        CUDA_CHECK(cudaMemcpy(host.data(), d_pixels,
                              host.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
        interopBuffer.unmap(optixContext.stream());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::ofstream out(snapshotPath, std::ios::binary);
        if (out)
        {
            out << "P6\n" << params.image_width << " " << params.image_height << "\n255\n";
            for (const uchar4& p : host)
            {
                out.put(static_cast<char>(p.x));
                out.put(static_cast<char>(p.y));
                out.put(static_cast<char>(p.z));
            }
            std::cout << "Snapshot written to " << snapshotPath << "\n";
        }
        else
        {
            std::cerr << "Failed to open snapshot file " << snapshotPath << "\n";
        }
    }
}

void printMetricsSummary(const MetricsCollector& metrics, const char* modeLabel)
{
    std::cout << "\nMetrics summary (" << modeLabel << " mode):\n"
              << "  frames   : " << metrics.numFramesRecorded() << "\n"
              << "  render ms: mean=" << metrics.meanRenderMs()
              << " min=" << metrics.minRenderMs()
              << " max=" << metrics.maxRenderMs() << "\n"
              << "  fps      : mean=" << metrics.meanFps()
              << " latest=" << metrics.latestFps() << "\n"
              << "  gpu mem  : " << metrics.gpuMemoryBytes() << " bytes\n";
    if (metrics.hasLatency())
        std::cout << "  latency  : latest=" << metrics.latestLatencyMs() << " ms\n";
}

// Appends one aggregate row (a single execution) to the shared summary CSV.
// The CSV has one header line, written only when the file is empty/new. This is
// called at the end of a successful run with every value already in memory, so
// the harness does not have to parse the stdout log.
static void writeSummaryRow(const char* path,
                            const char* dataset, const char* strategy,
                            long long diskBytes, const char* mode,
                            const std::string& seriesPath,
                            const Volume& volume, const std::string& modality,
                            size_t relCount, size_t relTotal, double relSparse,
                            float scalarMin, float scalarMax, float epsilon,
                            double buildMs,
                            const MetricsCollector& metrics,
                            const unsigned int dbg[4], double skipRatio,
                            const std::string& snapshotPath)
{
    if (!path || path[0] == '\0')
        return;

    // build_time_ms (load-time construction time, --build-time) is appended as
    // an extra column only when it was actually measured, so the schema of
    // pre-existing results.csv files remains unchanged.
    const char kHeaderBase[] =
        "dataset,strategy,mode,dicom_series_path,num_slices,modality,slice_dims,"
        "vol_dims,spacing,origin,scalar_min,scalar_max,n_relevant,n_total,sparsity_pct,"
        "uploaded_mb,epsilon,frames,render_mean_ms,render_min_ms,render_max_ms,"
        "fps_mean,fps_latest,gpu_mem_bytes,latency_latest_ms";
    const char kHeaderTail[] =
        ",vol_samples,dist_reads,leaps,total_steps_bounds,skip_ratio_bounds,snapshot_path\n";
    std::string header = kHeaderBase;
    if (buildMs >= 0.0)
        header += ",build_time_ms";
    header += kHeaderTail;

    std::FILE* f = std::fopen(path, "a");
    if (!f)
    {
        std::cerr << "Failed to open summary CSV " << path << "\n";
        return;
    }

    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) == 0)
        std::fputs(header.c_str(), f);

    const double latency = metrics.hasLatency() ? metrics.latestLatencyMs() : -1.0;
    const float volumeMB = static_cast<float>(volume.data.size() * sizeof(float) / (1024.0 * 1024.0));

    std::fprintf(f,
        "%s,%s,%s,%s,%d,%s,%dx%d,"
        "%dx%dx%d,(%g,%g,%g),(%g,%g,%g),"
        "%g,%g,%zu,%zu,%.4f,"
        "%g,%g,%u,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%zu,%.3f,",
        dataset ? dataset : "", strategy ? strategy : "", mode ? mode : "",
        seriesPath.c_str(), volume.dimZ, modality.c_str(), volume.dimX, volume.dimY,
        volume.dimX, volume.dimY, volume.dimZ,
        volume.spacingX, volume.spacingY, volume.spacingZ,
        volume.originX, volume.originY, volume.originZ,
        scalarMin, scalarMax, relCount, relTotal, relSparse,
        volumeMB, epsilon, metrics.numFramesRecorded(),
        metrics.meanRenderMs(), metrics.minRenderMs(), metrics.maxRenderMs(),
        metrics.meanFps(), metrics.latestFps(), metrics.gpuMemoryBytes(), latency);

    if (buildMs >= 0.0)
        std::fprintf(f, "%.6f,", buildMs);

    std::fprintf(f,
        "%u,%u,%u,%u,%.6f,%s\n",
        dbg[0], dbg[1], dbg[2], dbg[3], skipRatio,
        snapshotPath.c_str());

    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    std::string seriesPath = argv[1];
    TraceMode traceMode = TraceMode::MANUAL;
    std::string metricsPath;
    std::string snapshotPath;
    unsigned int metricsWindow = 120;
    int brickSize = 16;
    float brickEpsilon = 0.01f;
    int octreeLeafSize = 8;
    unsigned int maxFrames = 0;     // 0 = run until window closes
    bool nanovdbNearest = false;    // 1 = nearest, 0 = trilinear (NanoVDB, default)
    bool adaptiveMarch = false;     // adaptive-step inner march in region strategies
    bool dbgCounterBounds = false;   // log per-frame skip_ratio_bounds to CSV
    std::string summaryPath;        // aggregate one-row-per-run CSV (appended on exit)
    std::string summaryDataset;     // dataset label for the aggregate row
    std::string summaryStrategy;    // strategy/config label for the aggregate row
    long long summaryDiskBytes = -1; // size-on-disk of the series (bytes) for the row
    bool summaryBuildTime = false;  // --build-time: time load-time construction

    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            ++i;
            if (std::strcmp(argv[i], "optix") == 0)
                traceMode = TraceMode::OPTIX;
            else if (std::strcmp(argv[i], "manual") == 0)
                traceMode = TraceMode::MANUAL;
            else if (std::strcmp(argv[i], "adaptive") == 0)
                traceMode = TraceMode::ADAPTIVE;
            else if (std::strcmp(argv[i], "bricked-regions") == 0)
                traceMode = TraceMode::BRICKED_REGIONS;
            else if (std::strcmp(argv[i], "octree") == 0)
                traceMode = TraceMode::OCTREE;
            else if (std::strcmp(argv[i], "octree-regions") == 0)
                traceMode = TraceMode::OCTREE_REGIONS;
            else if (std::strcmp(argv[i], "nanovdb") == 0)
                traceMode = TraceMode::NANOVDB;
            else
            {
                std::cerr << "Unknown mode: " << argv[i] << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--brick-size") == 0 && i + 1 < argc)
        {
            brickSize = std::max(1, std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc)
        {
            brickEpsilon = std::strtof(argv[++i], nullptr);
        }
        else if (std::strcmp(argv[i], "--leaf-size") == 0 && i + 1 < argc)
        {
            octreeLeafSize = std::max(1, std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--metrics") == 0 && i + 1 < argc)
        {
            metricsPath = argv[++i];
        }
        else if (std::strcmp(argv[i], "--snapshot") == 0 && i + 1 < argc)
        {
            snapshotPath = argv[++i];
        }
        else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc)
        {
            metricsWindow = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
        {
            maxFrames = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (std::strcmp(argv[i], "--dbg-counter-bounds") == 0)
        {
            dbgCounterBounds = true;
        }
        else if (std::strcmp(argv[i], "--nanovdb-sampler") == 0 && i + 1 < argc)
        {
            const std::string v = argv[++i];
            if (v == "nearest")
                nanovdbNearest = true;
            else if (v == "trilinear")
                nanovdbNearest = false;
            else
            {
                std::cerr << "Unknown nanovdb sampler: " << v
                          << " (expected 'nearest' or 'trilinear')\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--adaptive-march") == 0 && i + 1 < argc)
        {
            const std::string v = argv[++i];
            if (v == "on")
                adaptiveMarch = true;
            else if (v == "off")
                adaptiveMarch = false;
            else
            {
                std::cerr << "Unknown adaptive-march value: " << v
                          << " (expected 'on' or 'off')\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--summary") == 0 && i + 1 < argc)
        {
            summaryPath = argv[++i];
        }
        else if (std::strcmp(argv[i], "--dataset") == 0 && i + 1 < argc)
        {
            summaryDataset = argv[++i];
        }
        else if (std::strcmp(argv[i], "--strategy") == 0 && i + 1 < argc)
        {
            summaryStrategy = argv[++i];
        }
        else if (std::strcmp(argv[i], "--disk-bytes") == 0 && i + 1 < argc)
        {
            summaryDiskBytes = std::atoll(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--build-time") == 0)
        {
            summaryBuildTime = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    const char* modeLabel =
        (traceMode == TraceMode::OPTIX)    ? "optix"    :
        (traceMode == TraceMode::ADAPTIVE) ? "adaptive" :
        (traceMode == TraceMode::BRICKED_REGIONS) ? "bricked-regions" :
        (traceMode == TraceMode::OCTREE)   ? "octree"   :
        (traceMode == TraceMode::OCTREE_REGIONS) ? "octree-regions" :
        (traceMode == TraceMode::NANOVDB)  ? "nanovdb" : "manual";

    try
    {
        std::cout << "Trace mode: " << modeLabel << "\n";
        if (traceMode == TraceMode::BRICKED_REGIONS)
            std::cout << "Brick size: " << brickSize << "^3 voxels, epsilon="
                      << brickEpsilon << "\n";
        if (traceMode == TraceMode::ADAPTIVE)
            std::cout << "Adaptive-step distance map, epsilon=" << brickEpsilon << "\n";
        if (traceMode == TraceMode::OCTREE || traceMode == TraceMode::BRICKED_REGIONS ||
            traceMode == TraceMode::OCTREE_REGIONS)
            std::cout << "Adaptive inner march: " << (adaptiveMarch ? "on" : "off") << "\n";
        std::cout << "Loading DICOM series from: " << seriesPath << "\n";
        DicomSeries series = loadDicomSeries(seriesPath);

        Volume volume;
        volume.buildFromSeries(series);

        TransferFunction tf;
        if (volume.modality == "CT")
            tf.buildDefaultCT();
        else
            tf.buildDefaultMR();

        GlWindow window;
        if (!window.init(1920, 1080, "OptiX Volume Renderer"))
            return 1;

        GL_CHECK(glViewport(0, 0, 1920, 1080));

        Display display;
        display.init(1920, 1080);

        OptixContext optixContext;
        optixContext.init();

        MetricsCollector metrics;
        metrics.init(optixContext.stream(),
                     metricsPath.empty() ? nullptr : metricsPath.c_str(),
                     metricsWindow);

        WindowUserData windowData;
        windowData.metrics = &metrics;

        unsigned int width = 1920;
        unsigned int height = 1080;

        InteropBuffer interopBuffer;
        interopBuffer.init(width, height);

        Camera camera;
        camera.aspect = static_cast<float>(width) / static_cast<float>(height);
        {
            float volSpanX = volume.dimX * volume.spacingX;
            float volSpanY = volume.dimY * volume.spacingY;
            float volSpanZ = volume.dimZ * volume.spacingZ;
            camera.radius = fmaxf(volSpanX, fmaxf(volSpanY, volSpanZ)) * 1.5f;
            camera.lookAt = make_float3(
                volume.originX + volSpanX * 0.5f,
                volume.originY + volSpanY * 0.5f,
                volume.originZ + volSpanZ * 0.5f);
            camera.theta = 0.0f;
            camera.phi   = 0.3f;
        }

        float vmin = 1e30f, vmax = -1e30f;
        {
            size_t total = volume.data.size();
            for (size_t i = 0; i < total; ++i)
            {
                float v = volume.data[i];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            std::cout << "Volume stats: range=[" << vmin << ", " << vmax << "]\n";
        }

        if (traceMode == TraceMode::MANUAL)
        {
            size_t total = volume.data.size();
            size_t nonzero = 0;
            for (size_t i = 0; i < total; ++i)
                if (volume.data[i] != 0.0f) nonzero++;
            std::cout << "Volume stats: " << nonzero << "/" << total << " nonzero\n";

            int cx = volume.dimX / 2, cy = volume.dimY / 2, cz = volume.dimZ / 2;
            std::cout << "Center voxel [" << cx << "," << cy << "," << cz << "] = "
                      << volume.data[cz * volume.dimX * volume.dimY + cy * volume.dimX + cx] << "\n";
            std::cout << "Origin voxel [0,0,0] = " << volume.data[0] << "\n";

            int midZ = volume.dimZ / 2;
            int nnz = 0;
            for (int y = 0; y < volume.dimY; ++y)
                for (int x = 0; x < volume.dimX; ++x)
                    if (volume.data[midZ * volume.dimX * volume.dimY + y * volume.dimX + x] != 0.0f)
                        nnz++;
            std::cout << "Middle slice (z=" << midZ << "): " << nnz << "/"
                      << volume.dimX * volume.dimY << " nonzero\n";
        }

        windowData.camera  = &camera;
        glfwSetWindowUserPointer(window.handle(), &windowData);
        glfwSetMouseButtonCallback(window.handle(), glfwMouseButtonCallback);
        glfwSetCursorPosCallback(window.handle(), glfwCursorPosCallback);
        glfwSetScrollCallback(window.handle(), glfwScrollCallback);

        double xpos, ypos;
        glfwGetCursorPos(window.handle(), &xpos, &ypos);
        camera.lastPos = make_float2(static_cast<float>(xpos),
                                     static_cast<float>(ypos));

        std::chrono::steady_clock::time_point tBuildStart = {};
        if (summaryBuildTime)
            tBuildStart = std::chrono::steady_clock::now();

        cudaArray* d_volumeArray = nullptr;
        cudaTextureObject_t volumeTex = 0;
        volume.uploadToDevice(&d_volumeArray, &volumeTex);

        float4* d_tfData = nullptr;
        {
            size_t tfBytes = sizeof(float4) * TransferFunction::LUT_SIZE;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tfData), tfBytes));
            CUDA_CHECK(cudaMemcpy(d_tfData, tf.lut, tfBytes, cudaMemcpyHostToDevice));
        }

        float scalarMin, scalarMax;
        if (volume.modality == "CT") {
            scalarMin = -1000.0f;
            scalarMax =  1000.0f;
        } else {
            scalarMin = 0.0f;
            scalarMax = 4096.0f;
        }

        // Rendering-relevant sparsity: fraction of voxels whose transfer-function
        // opacity is below the epsilon threshold. This is the same classification
        // every sparse strategy uses (see adaptive_grid_marcher.cpp, bricked_regions,
        // octree, nanovdb), so it reflects how much empty space the strategy can skip.
        // Values are hoisted to this scope so the end-of-run summary can reuse them.
        size_t relCount = 0, relTotal = 0;
        double relSparse = 100.0;
        {
            const float invRange  = 1.0f / (scalarMax - scalarMin);
            relTotal = volume.data.size();
            for (size_t i = 0; i < relTotal; ++i)
            {
                const float s   = volume.data[i];
                const float tft = (s - scalarMin) * invRange * 2047.0f;
                int idx = static_cast<int>(std::rint(tft));
                idx = std::max(0, std::min(2047, idx));
                if (tf.lut[idx].w >= brickEpsilon) ++relCount;
            }
            relSparse = relTotal > 0 ? 100.0 * (1.0 - static_cast<double>(relCount) / static_cast<double>(relTotal)) : 100.0;
            std::cout << "Volume stats: rendering-relevant (eps=" << brickEpsilon
                      << "): " << relCount << "/" << relTotal << " relevant, "
                      << relSparse << "% sparse\n";
        }

        const char* irPath = std::getenv("OPTIXIR_PATH");

        // Per-strategy setup: scene (acceleration structure) and the OptiX
        // program entry points that implement the strategy's traversal.
        std::unique_ptr<VolumeScene> volScene;
        std::unique_ptr<PlaceholderScene> placeScene;
        std::unique_ptr<AdaptiveGridMarcher> adaptiveGrid;
        std::unique_ptr<BrickedRegionsScene> brickedRegionsScene;
        std::unique_ptr<OctreeVolume> octreeVol;
        std::unique_ptr<OctreeRegionsScene> ocRegionsScene;
        std::unique_ptr<NanoVDBVolume> nanovdbVol;
        std::unique_ptr<PipelineBase> pipeline;

        Params params = {};
        params.image_width  = width;
        params.image_height = height;
        params.volumeTex    = volumeTex;
        params.tfData       = d_tfData;
        params.volumeDims   = make_int3(volume.dimX, volume.dimY, volume.dimZ);
        params.volumeSpacing = make_float3(volume.spacingX, volume.spacingY, volume.spacingZ);
        params.volumeOrigin  = make_float3(volume.originX, volume.originY, volume.originZ);
        params.volumeMax     = make_float3(volume.maxX(), volume.maxY(), volume.maxZ());
        params.scalarMin     = scalarMin;
        params.scalarMax     = scalarMax;

        // Builds and uploads the Chebyshev distance map for the strategies
        // whose intra-region march uses it (octree / bricked-regions /
        // octree-regions) and enables the adaptive-step gate. Only runs when
        // --adaptive-march is on; otherwise the shared region march keeps the
        // dense-baseline fixed step.
        auto enableAdaptiveStep = [&]()
        {
            if (!adaptiveMarch)
                return;
            adaptiveGrid.reset(new AdaptiveGridMarcher);
            adaptiveGrid->build(volume, tf, scalarMin, scalarMax, brickEpsilon);
            params.distanceTex = adaptiveGrid->distanceTex();
            params.epsilon     = brickEpsilon;
            params.useAdaptive = 1u;
            std::cout << "Distance map: " << adaptiveGrid->dims().x << "x"
                      << adaptiveGrid->dims().y << "x" << adaptiveGrid->dims().z
                      << ", " << adaptiveGrid->distanceBytes() / (1024.0 * 1024.0)
                      << " MB\n";
        };

        if (traceMode == TraceMode::OPTIX)
        {
            volScene.reset(new VolumeScene);
            volScene->init(optixContext.context(), optixContext.stream(),
                           make_float3(volume.originX, volume.originY, volume.originZ),
                           make_float3(volume.maxX(), volume.maxY(), volume.maxZ()));
            params.handle = volScene->handle();

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_optix", "__miss__ms", "__closesthit__ch_vol",
                           "__intersection__is", TraceMode::OPTIX);
        }
        else if (traceMode == TraceMode::ADAPTIVE)
        {
            // Single AABB GAS, identical to the OptiX-traced baseline, but the
            // adaptive raygen never optixTrace-s it: the whole march, including
            // the empty-space step-size decision, runs in software.
            volScene.reset(new VolumeScene);
            volScene->init(optixContext.context(), optixContext.stream(),
                           make_float3(volume.originX, volume.originY, volume.originZ),
                           make_float3(volume.maxX(), volume.maxY(), volume.maxZ()));
            params.handle = volScene->handle();

            adaptiveGrid.reset(new AdaptiveGridMarcher);
            adaptiveGrid->build(volume, tf, scalarMin, scalarMax, brickEpsilon);
            params.distanceTex = adaptiveGrid->distanceTex();
            params.epsilon     = brickEpsilon;
            params.useAdaptive = 1u;

            std::cout << "Distance map: " << adaptiveGrid->dims().x << "x"
                      << adaptiveGrid->dims().y << "x" << adaptiveGrid->dims().z
                      << ", " << adaptiveGrid->distanceBytes() / (1024.0 * 1024.0)
                      << " MB\n";

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_adaptive", "__miss__ms", "__closesthit__ch",
                           nullptr, TraceMode::ADAPTIVE);
        }
        else if (traceMode == TraceMode::BRICKED_REGIONS)
        {
            // One AABB primitive per non-empty brick (bricked, but with
            // hardware empty-skip). RT-core BVH traversal skips empty space;
            // the raygen loops with a utility ray (depth 1) over the bricks a
            // ray actually crosses, so numPayloadValues=4 (entry/exit/prim +
            // miss flag) and the region-style miss program.
            brickedRegionsScene.reset(new BrickedRegionsScene);
            brickedRegionsScene->init(optixContext.context(), optixContext.stream(),
                                      volume, tf, brickSize, scalarMin, scalarMax, brickEpsilon);
            params.handle               = brickedRegionsScene->handle();
            params.brickedRegionsAabbs  = brickedRegionsScene->deviceAabbs();

            enableAdaptiveStep();

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_bricked_regions", "__miss__ms_region",
                           "__closesthit__ch_bricked_regions", "__intersection__is_bricked_regions",
                           TraceMode::BRICKED_REGIONS, 4, 1);
            std::cout << "Bricked-regions scene: " << brickedRegionsScene->numRegions()
                      << " primitives\n";
        }
        else if (traceMode == TraceMode::OCTREE)
        {
            // True hierarchical octree traversal: single top-level AABB GAS,
            // the closest-hit descends the octree skipping empty subtrees.
            volScene.reset(new VolumeScene);
            volScene->init(optixContext.context(), optixContext.stream(),
                           make_float3(volume.originX, volume.originY, volume.originZ),
                           make_float3(volume.maxX(), volume.maxY(), volume.maxZ()));
            params.handle = volScene->handle();

            octreeVol.reset(new OctreeVolume);
            octreeVol->build(volume, tf, octreeLeafSize, scalarMin, scalarMax, brickEpsilon);
            params.octreeNodes    = octreeVol->deviceNodes();
            params.octreeChild    = octreeVol->deviceChild();
            params.octreeNodeCount = static_cast<unsigned int>(octreeVol->numNodes());

            enableAdaptiveStep();

            std::cout << "Octree: " << octreeVol->numNodes() << " nodes, "
                      << octreeVol->nodeBytes() / (1024.0 * 1024.0) << " MB, "
                      << octreeVol->numLeaves() << " leaves, "
                      << octreeVol->numRelevant() << " relevant, "
                      << octreeVol->numEmpty() << " empty, maxDepth="
                      << octreeVol->maxDepth() << "\n";

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_optix", "__miss__ms", "__closesthit__ch_octree",
                           "__intersection__is", TraceMode::OCTREE);
        }
        else if (traceMode == TraceMode::OCTREE_REGIONS)
        {
            // Octree leaves emitted as GAS AABBs: RT-core hardware skip with
            // data-adaptive granularity (non-empty leaves, variable size).
            ocRegionsScene.reset(new OctreeRegionsScene);
            ocRegionsScene->init(optixContext.context(), optixContext.stream(),
                                 volume, tf, octreeLeafSize, scalarMin, scalarMax,
                                 brickEpsilon);
            params.handle        = ocRegionsScene->handle();
            params.ocRegionAabbs = ocRegionsScene->deviceAabbs();
            params.ocRegionNode  = ocRegionsScene->deviceNodeMap();

            enableAdaptiveStep();

            std::cout << "Octree-regions scene: " << ocRegionsScene->numRegions()
                      << " leaf primitives\n";

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_region_octree", "__miss__ms_region",
                           "__closesthit__ch_ocregion", "__intersection__is_region_octree",
                           TraceMode::OCTREE_REGIONS, 4, 1);
        }
        else if (traceMode == TraceMode::NANOVDB)
        {
            // NanoVDB sparse GPU volume. A single top-level AABB GAS (identical
            // to the other GAS-based strategies); the closest-hit marches with
            // NanoVDB's device accessor, which performs its own empty-space
            // skipping internally (thesis \ref tab:strategy-optix-mapping).
            volScene.reset(new VolumeScene);
            volScene->init(optixContext.context(), optixContext.stream(),
                           make_float3(volume.originX, volume.originY, volume.originZ),
                           make_float3(volume.maxX(), volume.maxY(), volume.maxZ()));
            params.handle = volScene->handle();

            nanovdbVol.reset(new NanoVDBVolume);
            nanovdbVol->build(volume, tf, scalarMin, scalarMax, brickEpsilon);
            params.nanovdbGrid   = nanovdbVol->deviceGrid();
            params.nanovdbNearest = nanovdbNearest;

            std::cout << "NanoVDB grid: " << nanovdbVol->hostBytes() / (1024.0 * 1024.0)
                      << " MB on GPU, " << nanovdbVol->numActiveVoxels() << " active voxels, "
                      << "sparsity=" << (100.0f * nanovdbVol->sparsityRatio()) << "%\n";
            std::cout << "NanoVDB sampler: "
                      << (nanovdbNearest ? "nearest (fast)" : "trilinear (dense-equivalent)") << "\n";

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg_optix", "__miss__ms", "__closesthit__ch_nanovdb",
                           "__intersection__is", TraceMode::NANOVDB);
        }
        else
        {
            placeScene.reset(new PlaceholderScene);
            placeScene->init(optixContext.context(), optixContext.stream());
            params.handle = placeScene->handle();

            pipeline.reset(new PipelineBase);
            pipeline->init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                           "__raygen__rg", "__miss__ms", "__closesthit__ch",
                           nullptr, TraceMode::MANUAL);
        }

        CUdeviceptr d_params = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));

        // Diagnostic sample counters (volume samples, distance-map reads,
        // leaps, total-steps bounds) so the traversal cost and skip ratio of
        // each strategy can be compared.
        unsigned int* d_dbgCounters = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dbgCounters), sizeof(unsigned int) * 4));
        CUDA_CHECK(cudaMemset(d_dbgCounters, 0, sizeof(unsigned int) * 4));
        params.dbgCounters = d_dbgCounters;

        // Wire the per-frame counter CSV logging (opt-in via flag).
        metrics.enablePerFrameCounters(d_dbgCounters, dbgCounterBounds);

        metrics.recordGpuMemory();

        double buildMs = -1.0;
        if (summaryBuildTime)
        {
            buildMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tBuildStart)
                          .count();
            std::cout << "Construction time: " << buildMs << " ms\n";
        }

        std::cout << "Starting render loop (" << modeLabel << " mode)...\n";

        runRenderLoop(window, display, interopBuffer, optixContext, *pipeline,
                      metrics, camera, params, d_params, maxFrames,
                      snapshotPath.empty() ? nullptr : snapshotPath.c_str());

        printMetricsSummary(metrics, modeLabel);

        {
            unsigned int dbg[4] = { 0, 0, 0, 0 };
            CUDA_CHECK(cudaMemcpy(dbg, d_dbgCounters, sizeof(unsigned int) * 4,
                                  cudaMemcpyDeviceToHost));
            const unsigned int denom = dbg[0] > dbg[3] ? dbg[0] : dbg[3];
            const double ratio = denom ? 1.0 - static_cast<double>(dbg[0]) / denom : 0.0;
            std::cout << "  traversal : " << dbg[0] << " vol-samples, " << dbg[1]
                      << " dist-reads, " << dbg[2] << " leaps, " << dbg[3]
                      << " bounds-steps (per frame: "
                      << dbg[0] << " samples, " << dbg[1] << " reads, " << dbg[2]
                      << " leaps, " << dbg[3] << " bounds-steps)\n"
                      << "  skip ratio : bounds=" << ratio << "\n";

            writeSummaryRow(summaryPath.c_str(),
                            summaryDataset.c_str(), summaryStrategy.c_str(),
                            summaryDiskBytes, modeLabel, seriesPath,
                            volume, volume.modality,
                            relCount, relTotal, relSparse,
                            scalarMin, scalarMax, brickEpsilon, buildMs,
                            metrics, dbg, ratio,
                            snapshotPath);
        }
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_dbgCounters)));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
        volume.destroyDevice(d_volumeArray, volumeTex);
        if (d_tfData) cudaFree(d_tfData);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}