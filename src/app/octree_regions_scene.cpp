#include "octree_regions_scene.h"

#include "octree_volume.h"
#include "check_macros.h"
#include "vec_math.h"

#include <algorithm>
#include <cmath>
#include <iostream>

OctreeRegionsScene::~OctreeRegionsScene()
{
    if (d_aabb_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_aabb_)));
    if (d_nodeMap_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_nodeMap_)));
    if (d_gasOutput_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
}

void OctreeRegionsScene::init(OptixDeviceContext context, CUstream stream,
                              const Volume& volume, const TransferFunction& tf,
                              int leafSize, float scalarMin, float scalarMax, float epsilon)
{
    context_ = context;
    stream_  = stream;

    // Build the octree (Variant A construction), then flatten its relevant
    // leaves into GAS AABB primitives.
    OctreeVolume octree;
    octree.build(volume, tf, leafSize, scalarMin, scalarMax, epsilon);

    const auto& nodes = octree.hostNodes();
    const float3 volOrigin = make_float3(volume.originX, volume.originY, volume.originZ);
    const float3 volSpacing = make_float3(volume.spacingX, volume.spacingY, volume.spacingZ);

    aabbs_.clear();
    nodeMap_.clear();

    unsigned int numRelevantLeaves = 0;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const OctreeNode& n = nodes[i];
        if (!n.isLeaf || !n.relevant)
            continue;
        ++numRelevantLeaves;

        float3 bmin = volOrigin + make_float3(
            static_cast<float>(n.voxelMin[0]) * volSpacing.x,
            static_cast<float>(n.voxelMin[1]) * volSpacing.y,
            static_cast<float>(n.voxelMin[2]) * volSpacing.z);
        float3 bmax = volOrigin + make_float3(
            static_cast<float>(n.voxelMax[0]) * volSpacing.x,
            static_cast<float>(n.voxelMax[1]) * volSpacing.y,
            static_cast<float>(n.voxelMax[2]) * volSpacing.z);

        aabbs_.push_back({ bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z });
        nodeMap_.push_back(static_cast<unsigned int>(i));
    }

    std::cout << "Octree-regions: " << numRelevantLeaves << "/" << octree.numLeaves()
              << " relevant leaves flattened to GAS primitives\n";

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_),
                          aabbs_.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_), aabbs_.data(),
                          aabbs_.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nodeMap_),
                          nodeMap_.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_nodeMap_), nodeMap_.data(),
                          nodeMap_.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    buildGAS(context, stream);
}

void OctreeRegionsScene::buildGAS(OptixDeviceContext context, CUstream stream)
{
    if (aabbs_.empty())
    {
        OptixAabb dummy = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_), sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_), &dummy,
                              sizeof(OptixAabb), cudaMemcpyHostToDevice));
        uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

        OptixBuildInput buildInput = {};
        buildInput.type                            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        buildInput.customPrimitiveArray.aabbBuffers = &d_aabb_;
        buildInput.customPrimitiveArray.numPrimitives        = 1;
        buildInput.customPrimitiveArray.strideInBytes        = 0;
        buildInput.customPrimitiveArray.flags                = inputFlags;
        buildInput.customPrimitiveArray.numSbtRecords        = 1;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions,
                                                 &buildInput, 1, &gasBufferSizes));
        CUdeviceptr d_temp = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), gasBufferSizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput_),
                              gasBufferSizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions,
                                    &buildInput, 1,
                                    d_temp, gasBufferSizes.tempSizeInBytes,
                                    d_gasOutput_, gasBufferSizes.outputSizeInBytes,
                                    &handle_, nullptr, 0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));
        return;
    }

    uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput buildInput = {};
    buildInput.type                            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers = &d_aabb_;
    buildInput.customPrimitiveArray.numPrimitives        = static_cast<unsigned int>(aabbs_.size());
    buildInput.customPrimitiveArray.strideInBytes        = sizeof(OptixAabb);
    buildInput.customPrimitiveArray.flags                = inputFlags;
    buildInput.customPrimitiveArray.numSbtRecords        = 1;
    buildInput.customPrimitiveArray.sbtIndexOffsetBuffer           = 0;
    buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes      = 0;
    buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes    = 0;
    buildInput.customPrimitiveArray.primitiveIndexOffset           = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions,
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

    OPTIX_CHECK(optixAccelBuild(context, stream, &accelOptions,
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
        OPTIX_CHECK(optixAccelCompact(context, stream, handle_,
                                      d_compacted, compactedSize, &handle_));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
        d_gasOutput_ = d_compacted;
    }
}
