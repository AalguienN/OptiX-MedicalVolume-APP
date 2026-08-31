#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <cuda_runtime.h>

#include <vector>

// Host-side construction of the adaptive-step grid marcher representation
// (Section "Adaptive-step grid marcher" of the thesis).
//
// The underlying dense voxel grid is left unmodified; a separate distance
// map of the same resolution is precomputed, storing at each voxel the
// Chebyshev (L-infinity) distance, in voxel units, to the nearest
// rendering-relevant (non-empty) voxel. "Relevant" is decided exactly like
// the bricked strategy: a voxel is relevant if its transfer-function
// opacity reaches the configured epsilon. Relevant voxels get distance 0.
//
// The distance map is uploaded to GPU memory as a 3D texture so the device
// marcher can sample it with the same normalized trilinear addressing used
// for the volume. Unlike the bricked strategy this adds an auxiliary array
// on top of the dense volume rather than replacing or compressing it, so it
// is not expected to reduce memory footprint versus the baseline; it targets
// traversal cost only.
class AdaptiveGridMarcher
{
public:
    AdaptiveGridMarcher() = default;
    ~AdaptiveGridMarcher();

    AdaptiveGridMarcher(const AdaptiveGridMarcher&)            = delete;
    AdaptiveGridMarcher& operator=(const AdaptiveGridMarcher&) = delete;

    // Computes the occupancy-driven Chebyshev distance map and uploads it to
    // a 3D texture. epsilon is the opacity threshold classifying a voxel as
    // rendering-relevant, matching the bricked strategy's definition.
    void build(const Volume& volume, const TransferFunction& tf,
               float scalarMin, float scalarMax, float epsilon);

    int3 dims() const { return dims_; }

    // Device-resident 3D texture object holding the distance map.
    cudaTextureObject_t distanceTex() const { return d_distanceTex_; }

    // Bytes of the host-side distance map (its GPU footprint is the same,
    // plus one byte per voxel).
    size_t distanceBytes() const { return dist_.size(); }

private:
    void uploadToDevice();

    int3   dims_ = {};
    std::vector<unsigned char> dist_;      // Chebyshev distance (voxels) per voxel
    cudaArray*           d_distanceArray_ = nullptr;
    cudaTextureObject_t  d_distanceTex_   = 0;
};
