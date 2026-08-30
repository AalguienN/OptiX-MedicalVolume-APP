#include "regions_scene.h"

#include "check_macros.h"
#include "vec_math.h"

#include <algorithm>
#include <cmath>
#include <iostream>

RegionsScene::~RegionsScene()
{
    if (d_aabb_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_aabb_)));
    if (d_brickMap_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_brickMap_)));
    if (d_gasOutput_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
}

void RegionsScene::init(OptixDeviceContext context, CUstream stream,
                        const Volume& volume, const TransferFunction& tf,
                        int brickSize, float scalarMin, float scalarMax, float epsilon)
{
    context_ = context;
    stream_  = stream;
    epsilon_ = epsilon;

    const int bs = std::max(1, brickSize);

    brickDims_  = make_int3(bs, bs, bs);
    brickCount_ = make_int3((volume.dimX + bs - 1) / bs,
                            (volume.dimY + bs - 1) / bs,
                            (volume.dimZ + bs - 1) / bs);
    brickSize_  = make_float3(bs * volume.spacingX,
                              bs * volume.spacingY,
                              bs * volume.spacingZ);

    const int nx = brickCount_.x;
    const int ny = brickCount_.y;
    const int nz = brickCount_.z;
    numBricks_ = nx * ny * nz;

    const float invRange = 1.0f / (scalarMax - scalarMin);

    // Sanity-check the per-brick world origin convention against the rest of
    // the application. A brick (bx,by,bz) covers voxels [bx*bs,(bx+1)*bs) etc.
    // and its world AABB is
    //   origin + (bx,by,bz)*brickSize .. origin + (bx+1,by+1,bz+1)*brickSize
    // clamped to the volume max, so an AABB primitive never extends past the
    // volume bounds that the sampling code clamps to.
    const float3 volOrigin = make_float3(volume.originX, volume.originY, volume.originZ);
    const float3 volMax    = make_float3(volume.maxX(), volume.maxY(), volume.maxZ());

    brickMap_.clear();
    aabbs_.clear();

    for (int bz = 0; bz < nz; ++bz)
    {
        for (int by = 0; by < ny; ++by)
        {
            for (int bx = 0; bx < nx; ++bx)
            {
                const int vx0 = bx * bs, vy0 = by * bs, vz0 = bz * bs;
                const int vx1 = std::min(volume.dimX, vx0 + bs);
                const int vy1 = std::min(volume.dimY, vy0 + bs);
                const int vz1 = std::min(volume.dimZ, vz0 + bs);

                const int brickLinear = (bz * ny + by) * nx + bx;

                float maxOpacity = -1e30f;
                for (int vz = vz0; vz < vz1; ++vz)
                    for (int vy = vy0; vy < vy1; ++vy)
                        for (int vx = vx0; vx < vx1; ++vx)
                        {
                            const float s = volume.data[
                                (static_cast<size_t>(vz) * volume.dimY + vy) * volume.dimX + vx];
                            const float tf_t = (s - scalarMin) * invRange * 2047.0f;
                            int idx = static_cast<int>(std::rint(tf_t));
                            idx = std::max(0, std::min(2047, idx));
                            maxOpacity = std::max(maxOpacity, tf.lut[idx].w);
                        }

                if (maxOpacity < epsilon)
                    continue;  // empty brick -> no primitive

                // World AABB of this brick, clamped to the volume max so a
                // boundary brick that runs past the last voxel does not sample
                // (and this matches the clamp-to-[0,1] in the samplers).
                float3 bmin = volOrigin + make_float3(
                    static_cast<float>(bx) * brickSize_.x,
                    static_cast<float>(by) * brickSize_.y,
                    static_cast<float>(bz) * brickSize_.z);
                float3 bmax = volOrigin + make_float3(
                    static_cast<float>(bx + 1) * brickSize_.x,
                    static_cast<float>(by + 1) * brickSize_.y,
                    static_cast<float>(bz + 1) * brickSize_.z);

                bmax.x = std::min(bmax.x, volMax.x);
                bmax.y = std::min(bmax.y, volMax.y);
                bmax.z = std::min(bmax.z, volMax.z);

                aabbs_.push_back({ bmin.x, bmin.y, bmin.z,
                                   bmax.x, bmax.y, bmax.z });
                brickMap_.push_back(static_cast<unsigned int>(brickLinear));
            }
        }
    }

    numRegions_ = static_cast<int>(aabbs_.size());

    std::cout << "Region strategy: " << numRegions_ << "/" << numBricks_
              << " bricks are relevant (epsilon=" << epsilon << ")\n";

    // Upload per-primitive AABBs and primitive->brick mapping.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_),
                          aabbs_.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_), aabbs_.data(),
                          aabbs_.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_brickMap_),
                          brickMap_.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_brickMap_), brickMap_.data(),
                          brickMap_.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

    buildGAS(context, stream);
}

void RegionsScene::buildGAS(OptixDeviceContext context, CUstream stream)
{
    if (aabbs_.empty())
    {
        // No relevant regions. Build a degenerate single primitive so a valid
        // traversable handle always exists (optixLaunch requires one).
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
        buildInput.customPrimitiveArray.primitiveIndexOffset = 0;

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
    buildInput.customPrimitiveArray.numPrimitives        = numRegions_;
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
