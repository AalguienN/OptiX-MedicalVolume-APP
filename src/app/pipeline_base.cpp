#include "check_macros.h"
#include "pipeline_base.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <fstream>
#include <stdexcept>
#include <vector>

PipelineBase::~PipelineBase()
{
    if (raygenRecord_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(raygenRecord_)));
    if (missRecord_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(missRecord_)));
    if (hitgroupRecord_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(hitgroupRecord_)));
    if (pipeline_)
        OPTIX_CHECK_NOEXCEPT(optixPipelineDestroy(pipeline_));
    if (hitgroupPG_)
        OPTIX_CHECK_NOEXCEPT(optixProgramGroupDestroy(hitgroupPG_));
    if (missPG_)
        OPTIX_CHECK_NOEXCEPT(optixProgramGroupDestroy(missPG_));
    if (raygenPG_)
        OPTIX_CHECK_NOEXCEPT(optixProgramGroupDestroy(raygenPG_));
    if (module_)
        OPTIX_CHECK_NOEXCEPT(optixModuleDestroy(module_));
}

void PipelineBase::init(OptixDeviceContext context,
                        const std::string& optixIrPath,
                        const char*        raygenEntry,
                        const char*        missEntry,
                        const char*        closestHitEntry)
{
    context_ = context;

    createModule(optixIrPath);
    createProgramGroups(raygenEntry, missEntry, closestHitEntry);
    createPipeline();
    createSbt();
}

void PipelineBase::createModule(const std::string& optixIrPath)
{
    OptixModuleCompileOptions module_compile_options = {};

    compileOptions_.usesMotionBlur              = false;
    compileOptions_.traversableGraphFlags       = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    compileOptions_.numPayloadValues            = 3;
    compileOptions_.numAttributeValues          = 2;
    compileOptions_.exceptionFlags              = OPTIX_EXCEPTION_FLAG_NONE;
    compileOptions_.pipelineLaunchParamsVariableName = "params";
    compileOptions_.usesPrimitiveTypeFlags      = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string optixIr;
    {
        const char* path = std::getenv("OPTIXIR_PATH");
        if (!path)
            path = optixIrPath.c_str();

        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error(std::string("Cannot open OptiX IR file: ") + path);

        optixIr.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    char   LOG[2048];
    size_t LOG_SIZE = sizeof(LOG);
    OPTIX_CHECK_LOG(optixModuleCreate(context_, &module_compile_options, &compileOptions_,
                                      optixIr.data(), optixIr.size(),
                                      LOG, &LOG_SIZE, &module_));
}

void PipelineBase::createProgramGroups(const char* raygenEntry, const char* missEntry, const char* closestHitEntry)
{
    OptixProgramGroupOptions program_group_options = {};

    char   LOG[2048];
    size_t LOG_SIZE = sizeof(LOG);

    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module         = module_;
    raygen_desc.raygen.entryFunctionName = raygenEntry;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &raygen_desc, 1, &program_group_options, LOG, &LOG_SIZE, &raygenPG_));

    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module           = module_;
    miss_desc.miss.entryFunctionName = missEntry;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context_, &miss_desc, 1, &program_group_options, LOG, &LOG_SIZE, &missPG_));

    if (closestHitEntry)
    {
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH     = module_;
        hitgroup_desc.hitgroup.entryFunctionNameCH = closestHitEntry;
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context_, &hitgroup_desc, 1, &program_group_options, LOG, &LOG_SIZE, &hitgroupPG_));
    }
}

void PipelineBase::createPipeline()
{
    OptixProgramGroup program_groups[] = { raygenPG_, missPG_, hitgroupPG_ };
    const unsigned int num_groups = hitgroupPG_ ? 3 : 2;

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth            = 1;

    char   LOG[2048];
    size_t LOG_SIZE = sizeof(LOG);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_, &compileOptions_, &link_options,
        program_groups, num_groups, LOG, &LOG_SIZE, &pipeline_));

    OptixStackSizes stack_sizes = {};
    for (unsigned int i = 0; i < num_groups; ++i)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_groups[i], &stack_sizes, pipeline_));

    uint32_t css;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &css, &css, &css));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, 0, 0, css, 1));
}

void PipelineBase::createSbt()
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygenRecord_), sizeof(SbtRecord<RayGenData>)));
    SbtRecord<RayGenData> rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygenRecord_), &rg_sbt,
                          sizeof(SbtRecord<RayGenData>), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&missRecord_), sizeof(SbtRecord<MissData>)));
    SbtRecord<MissData> ms_sbt = {};
    ms_sbt.data = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(missRecord_), &ms_sbt,
                          sizeof(SbtRecord<MissData>), cudaMemcpyHostToDevice));

    if (hitgroupPG_)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecord_), sizeof(SbtRecord<HitGroupData>)));
        SbtRecord<HitGroupData> hg_sbt = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG_, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecord_), &hg_sbt,
                              sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice));
    }

    sbt_.raygenRecord                = raygenRecord_;
    sbt_.missRecordBase              = missRecord_;
    sbt_.missRecordStrideInBytes     = sizeof(SbtRecord<MissData>);
    sbt_.missRecordCount             = 1;
    sbt_.hitgroupRecordBase          = hitgroupRecord_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
    sbt_.hitgroupRecordCount         = hitgroupPG_ ? 1 : 0;
}

void PipelineBase::updateRayGenRecord(CUstream stream, const RayGenData& data)
{
    SbtRecord<RayGenData> rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &rg_sbt));
    rg_sbt.data = data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(raygenRecord_), &rg_sbt,
                               sizeof(SbtRecord<RayGenData>), cudaMemcpyHostToDevice, stream));
}
