#pragma once

#include <optix.h>
#include <vector_functions.h>

#include <string>

#include "shared_device.h"

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
              const char*        closestHitEntry);

    OptixPipeline            pipeline() { return pipeline_; }
    OptixShaderBindingTable* sbt() { return &sbt_; }

    void updateRayGenRecord(CUstream stream, const RayGenData& data);

private:
    void createModule(const std::string& optixIrPath);
    void createProgramGroups(const char* raygenEntry, const char* missEntry, const char* closestHitEntry);
    void createPipeline();
    void createSbt();

    OptixDeviceContext             context_      = nullptr;
    OptixPipelineCompileOptions    compileOptions_ = {};
    OptixModule                    module_       = nullptr;
    OptixProgramGroup              raygenPG_     = nullptr;
    OptixProgramGroup              missPG_       = nullptr;
    OptixProgramGroup              hitgroupPG_   = nullptr;
    OptixPipeline                  pipeline_     = nullptr;
    OptixShaderBindingTable        sbt_          = {};
    CUdeviceptr                    raygenRecord_   = 0;
    CUdeviceptr                    missRecord_     = 0;
    CUdeviceptr                    hitgroupRecord_ = 0;
};
