#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <optix.h>
#include <cuda_runtime.h>

#include <vector>

// Host-side construction and GAS build of the "region" strategy
// (--mode region, Variant B of the thesis).
//
// The volume is subdivided into fixed-size bricks (reusing the bricked
// strategy's classification). Every brick whose metadata classifies it as
// rendering-relevant (maxOpacity >= epsilon) is emitted as a single custom
// AABB primitive in the GAS; empty bricks are simply omitted. The OptiX
// BVH / RT cores therefore perform empty-space skipping in hardware: a ray
// that crosses empty space never intersects any primitive, and the closest-hit
// program is invoked only for the (small) non-empty regions. This is in
// contrast to the bricked strategy, which keeps one top-level AABB and does
// the relevance check in software during the march.
class RegionsScene
{
public:
    RegionsScene() = default;
    ~RegionsScene();

    RegionsScene(const RegionsScene&)            = delete;
    RegionsScene& operator=(const RegionsScene&) = delete;

    // Classifies the volume into bricks, keeps only the relevant ones,
    // uploads the per-primitive AABBs and the primitive->brick mapping, and
    // builds the multi-primitive GAS. Returns via handle() the traversable.
    void init(OptixDeviceContext context, CUstream stream,
              const Volume& volume, const TransferFunction& tf,
              int brickSize, float scalarMin, float scalarMax, float epsilon);

    OptixTraversableHandle handle() const { return handle_; }

    OptixAabb*     deviceAabbs() const { return reinterpret_cast<OptixAabb*>(d_aabb_); }
    unsigned int*  deviceBrickMap() const { return reinterpret_cast<unsigned int*>(d_brickMap_); }
    int3           brickCount() const { return brickCount_; }
    float3         brickSize() const { return brickSize_; }
    int            numRegions() const { return numRegions_; }
    int            numBricks() const { return numBricks_; }
    float          epsilon() const { return epsilon_; }

private:
    void buildGAS(OptixDeviceContext context, CUstream stream);

    OptixDeviceContext context_ = nullptr;
    CUstream           stream_  = nullptr;

    int3   brickDims_   = {};
    int3   brickCount_  = {};
    float3 brickSize_   = {};
    int    numRegions_  = 0;
    int    numBricks_   = 0;
    float  epsilon_     = 0.0f;

    std::vector<OptixAabb>  aabbs_;
    std::vector<unsigned int> brickMap_;

    CUdeviceptr        d_aabb_      = 0;
    CUdeviceptr        d_brickMap_  = 0;
    CUdeviceptr        d_gasOutput_ = 0;
    OptixTraversableHandle handle_  = 0;
};
