#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <cuda_runtime.h>

#include <vector>

// Host-side construction of the bricked / tiled volume representation
// (Section "Bricked / tiled volume" of the thesis). The volume is
// subdivided into fixed-size bricks of n^3 voxels. Each brick carries a
// BrickMeta: minimum and maximum scalar value, minimum and maximum
// opacity after transfer-function application, and a flag indicating
// whether the brick contains any rendering-relevant voxel (maxOpacity
// >= epsilon). The voxel data itself stays in the dense 3D texture; the
// metadata array is uploaded once to GPU memory.
class BrickedVolume
{
public:
    struct HostMeta
    {
        float  minScalar  = 0.0f;
        float  maxScalar  = 0.0f;
        float  minOpacity = 0.0f;
        float  maxOpacity = 0.0f;
        bool   relevant   = false;
    };

    BrickedVolume() = default;
    ~BrickedVolume();

    BrickedVolume(const BrickedVolume&)            = delete;
    BrickedVolume& operator=(const BrickedVolume&) = delete;

    void build(const Volume& volume, const TransferFunction& tf,
               int brickSize, float scalarMin, float scalarMax, float epsilon);

    int3   brickDims()  const { return brickDims_; }
    int3   brickCount() const { return brickCount_; }
    float3 brickSize()  const { return brickSize_; }
    BrickMeta* deviceMeta() const { return d_meta_; }

    int    numBricks()         const { return static_cast<int>(meta_.size()); }
    int    numRelevantBricks() const { return numRelevant_; }
    size_t metadataBytes()     const { return meta_.size() * sizeof(BrickMeta); }

private:
    void uploadToDevice();

    std::vector<HostMeta> meta_;
    int3   brickDims_  = {};
    int3   brickCount_ = {};
    float3 brickSize_  = {};
    int    numRelevant_ = 0;
    BrickMeta* d_meta_ = nullptr;
};