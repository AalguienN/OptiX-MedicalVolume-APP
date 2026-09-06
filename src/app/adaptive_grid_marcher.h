#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <cuda_runtime.h>

#include <array>
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
//
// A directional (quantized) variant of the same map is also supported. The
// sphere of ray directions is partitioned into 6 face sectors
// (dominant axis + sign, i.e. the +-x/+-y/+-z cones), and for every voxel the
// six distances to the nearest rendering-relevant voxel along each of the six
// principal directions are stored (one unsigned char per direction, capped to
// 255 voxels). During the march the device side quantizes the ray direction
// to its sector and uses that distance as the safe leap size. This is the
// thesis proposal: a quantized directional distance map for empty-space
// skipping. The scalar (all-direction) Chebyshev map is kept unchanged and
// used as a directional safety floor on the device.
class AdaptiveGridMarcher
{
public:
    // Supported directional discretizations (thesis sect. 3.2.3).
    enum DirectionalSectors
    {
        DIR_SCALAR = 0, // scalar Chebyshev map only (original adaptive strategy)
        DIR_FACES  = 6, // six face (axis + sign) sectors
    };

    AdaptiveGridMarcher() = default;
    ~AdaptiveGridMarcher();

    AdaptiveGridMarcher(const AdaptiveGridMarcher&)            = delete;
    AdaptiveGridMarcher& operator=(const AdaptiveGridMarcher&) = delete;

    // Computes the occupancy-driven Chebyshev distance map and uploads it to
    // a 3D texture. epsilon is the opacity threshold classifying a voxel as
    // rendering-relevant, matching the bricked strategy's definition.
    void build(const Volume& volume, const TransferFunction& tf,
               float scalarMin, float scalarMax, float epsilon);

    // Directional (quantized) distance map. Reuses the occupancy computed by
    // build() and uploads one 3D texture per sector. sectors must be 0 (no
    // directional map) or 6 (face sectors); call after build(). Returns true
    // if a directional map was produced.
    bool buildDirectional(int sectors);

    int3 dims() const { return dims_; }

    // Device-resident 3D texture object holding the scalar distance map.
    cudaTextureObject_t distanceTex() const { return d_distanceTex_; }

    // Number of directional sectors (0 if none), and the per-sector device
    // 3D texture objects. sectorTexture(i) is only valid for i < sectors().
    int  sectors() const { return dirSectorCount_; }
    cudaTextureObject_t sectorTexture(int i) const { return dirTextures_[i]; }

    // Bytes of the host-side distance map (its GPU footprint is the same,
    // plus one byte per voxel).
    size_t distanceBytes() const { return dist_.size(); }

private:
    void uploadToDevice();
    void uploadDirectionalToDevice();

    // Computes the six axis-aligned directional distances at (x,y,z) into out
    // ([0..5] = +x,-x,+y,-y,+z,+z), in voxel Chebyshev units, capped to 255.
    void axisFaceDistances(int x, int y, int z,
                           const std::vector<unsigned char>& occupied,
                           std::array<unsigned char, 6>& out) const;

    int3   dims_ = {};
    std::vector<unsigned char> dist_;      // Chebyshev distance (voxels) per voxel
    cudaArray*           d_distanceArray_ = nullptr;
    cudaTextureObject_t  d_distanceTex_   = 0;

    int dirSectorCount_ = 0;                       // 0 = none, else must be 6
    std::vector<std::vector<unsigned char>> dirDist_;  // sectors x V distances
    std::vector<cudaArray*>            dirArrays_;
    std::vector<cudaTextureObject_t>   dirTextures_;
};
