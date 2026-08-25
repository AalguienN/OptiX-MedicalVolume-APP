#include "volume_scene.h"
#include "check_macros.h"

#include <optix.h>
#include <optix_stubs.h>

VolumeScene::~VolumeScene()
{
    if (d_aabb_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_aabb_)));
    if (d_gasOutput_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
}

void VolumeScene::init(OptixDeviceContext context, CUstream stream,
                       float3 bmin, float3 bmax)
{
    context_ = context;
    stream_  = stream;

    float aabb[6] = {
        bmin.x, bmin.y, bmin.z,
        bmax.x, bmax.y, bmax.z
    };
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_), sizeof(aabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_), aabb,
                          sizeof(aabb), cudaMemcpyHostToDevice));

    uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput buildInput = {};
    buildInput.type                      = OPTIX_BUILD_INPUT_TYPE_AABB;
    buildInput.aabbArray.aabbs           = reinterpret_cast<const float*>(d_aabb_);
    buildInput.aabbArray.numAabbs        = 1;
    buildInput.aabbArray.strideInBytes   = sizeof(float) * 6;
    buildInput.aabbArray.inputFlags      = inputFlags;
    buildInput.aabbArray.numSbtRecords   = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accelOptions,
                                             &buildInput, 1, &gasBufferSizes));

    CUdeviceptr d_tempBuffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer),
                          gasBufferSizes.tempSizeInBytes));

    size_t compactedSizeOffset =
        (gasBufferSizes.outputSizeInBytes + 7) & ~static_cast<size_t>(7);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput_),
                          compactedSizeOffset + sizeof(size_t)));

    OptixAccelEmitDesc emitProp = {};
    emitProp.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProp.result = d_gasOutput_ + compactedSizeOffset;

    OPTIX_CHECK(optixAccelBuild(context_, stream_, &accelOptions,
                                &buildInput, 1,
                                d_tempBuffer, gasBufferSizes.tempSizeInBytes,
                                d_gasOutput_, gasBufferSizes.outputSizeInBytes,
                                &handle_, &emitProp, 1));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));

    size_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize,
                          reinterpret_cast<void*>(emitProp.result),
                          sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compactedSize < gasBufferSizes.outputSizeInBytes)
    {
        CUdeviceptr d_compacted = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted), compactedSize));
        OPTIX_CHECK(optixAccelCompact(context_, stream_, handle_,
                                      d_compacted, compactedSize, &handle_));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
        d_gasOutput_ = d_compacted;
    }
}
