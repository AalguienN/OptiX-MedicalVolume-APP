#pragma once

#include <optix.h>
#include <vector_functions.h>

#include <string>

#include "shared_device.h"

enum class TraceMode { MANUAL, OPTIX, BRICKED, ADAPTIVE, REGION, OCTREE, OCTREE_REGIONS, NANOVDB };

class PipelineBase
{
public:
    PipelineBase()      = default;
    ~PipelineBase();

    PipelineBase(const PipelineBase&)            = delete;
    PipelineBase& operator=(const PipelineBase&) = delete;

    void init(OptixDeviceContext context,
              const std::string& optixIrPath,
              const char*        raygenEntry,
              const char*        missEntry,
              const char*        closestHitEntry,
              const char*        intersectionEntry,
              TraceMode          mode,
              unsigned int       numPayloadValues = 3,
              unsigned int       maxTraceDepth    = 1);

    OptixPipeline            pipeline() { return pipeline_; }
    OptixShaderBindingTable* sbt() { return &sbt_; }
    TraceMode                mode() const { return mode_; }

    void updateRayGenRecord(CUstream stream, const RayGenData& data);

private:
    void createModule(const std::string& optixIrPath);
    void createProgramGroups(const char* raygenEntry, const char* missEntry,
                             const char* closestHitEntry, const char* intersectionEntry);
    void createPipeline();
    void createSbt();

    TraceMode                mode_         = TraceMode::MANUAL;
    OptixDeviceContext       context_      = nullptr;
    OptixPipelineCompileOptions compileOptions_ = {};
    OptixModule              module_       = nullptr;
    OptixProgramGroup        raygenPG_     = nullptr;
    OptixProgramGroup        missPG_       = nullptr;
    OptixProgramGroup        hitgroupPG_   = nullptr;
    OptixPipeline            pipeline_     = nullptr;
    OptixShaderBindingTable  sbt_          = {};
    CUdeviceptr              raygenRecord_   = 0;
    CUdeviceptr              missRecord_     = 0;
    CUdeviceptr              hitgroupRecord_ = 0;

    unsigned int             maxTraceDepth_    = 1;
};
