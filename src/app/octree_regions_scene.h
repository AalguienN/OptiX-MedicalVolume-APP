#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <optix.h>
#include <cuda_runtime.h>

#include <vector>

// Host-side construction and GAS build of the "octree-regions" strategy
// (--mode octree-regions, Variant B of the thesis).
//
// It combines the hierarchical octree with the hardware-skip idea of the
// bricked-regions strategy: the octree is built (reusing OctreeVolume), and
// every rendering-relevant leaf is emitted as a single custom AABB primitive
// in the GAS. Empty subtrees are simply not represented, so the OptiX BVH /
// RT cores skip them in hardware, but with a data-adaptive granularity (leaves
// are variable-sized, unlike the fixed bricks of the bricked-regions
// strategy). The closest-hit program marches the voxel region of the hit
// leaf.
class OctreeRegionsScene
{
public:
    OctreeRegionsScene() = default;
    ~OctreeRegionsScene();

    OctreeRegionsScene(const OctreeRegionsScene&)            = delete;
    OctreeRegionsScene& operator=(const OctreeRegionsScene&) = delete;

    void init(OptixDeviceContext context, CUstream stream,
              const Volume& volume, const TransferFunction& tf,
              int leafSize, float scalarMin, float scalarMax, float epsilon);

    OptixTraversableHandle handle() const { return handle_; }

    OptixAabb*     deviceAabbs() const { return reinterpret_cast<OptixAabb*>(d_aabb_); }
    unsigned int*  deviceNodeMap() const { return reinterpret_cast<unsigned int*>(d_nodeMap_); }
    int            numRegions() const { return static_cast<int>(aabbs_.size()); }

private:
    void buildGAS(OptixDeviceContext context, CUstream stream);

    OptixDeviceContext context_ = nullptr;
    CUstream           stream_  = nullptr;

    std::vector<OptixAabb>    aabbs_;
    std::vector<unsigned int> nodeMap_;

    CUdeviceptr        d_aabb_      = 0;
    CUdeviceptr        d_nodeMap_   = 0;
    CUdeviceptr        d_gasOutput_ = 0;
    OptixTraversableHandle handle_  = 0;
};
