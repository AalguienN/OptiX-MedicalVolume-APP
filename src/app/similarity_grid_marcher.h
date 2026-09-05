#pragma once

#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <cuda_runtime.h>

#include <vector>

// Host-side construction of the homogeneous-region (similarity-based)
// adaptive marcher representation.
//
// Faithful to the "Adaptive-step grid marcher" design (AdaptiveGridMarcher)
// in that it keeps the dense grid untouched and adds a single same-resolution
// per-voxel distance texture that the device reads once per leap at runtime.
// It is NOT an octree: no hierarchy is ever built or traversed at render time.
//
// The distinction is the meaning of the distance:
//   * AdaptiveGridMarcher:  Chebyshev distance to the nearest RENDERING-
//     IRRELEVANT-vs-RELEVANT boundary (opacity crossing epsilon).
//   * SimilarityGridMarcher: Chebyshev distance to the nearest voxel that
//     differs from THIS voxel by >= delta in transfer-function opacity OR in
//     any color channel. This anchors the leap on the sample's own value, so a
//     single sample can provably stand in for a whole window that is
//     "similar" to it (thesis "Homogeneous Regions": max-min < delta).
//
// Because "similar to me" is a relative predicate (voxel u may be within
// delta of v but not of w), it cannot be expressed as a single global binary
// occupancy mask the way the empty-vs-relevant map is. Instead the build does
// a bounded per-voxel outward Chebyshev shell search (capped at maxRadius) for
// the nearest dissimilar neighbour; voxels whose whole capped window is
// similar get the cap value. This is offline (build time is not part of the
// real-time comparison) and embarrassingly parallel.
//
// The map is uploaded exactly like the adaptive distance map: a normalized
// unsigned-char 3D texture read with point filtering, rescaled by 255 on the
// device.
class SimilarityGridMarcher
{
public:
    SimilarityGridMarcher() = default;
    ~SimilarityGridMarcher();

    SimilarityGridMarcher(const SimilarityGridMarcher&)            = delete;
    SimilarityGridMarcher& operator=(const SimilarityGridMarcher&) = delete;

    // deltaAlpha: opacity spread below which a window is similar (absolute,
    // i.e. max-min of transfer-function opacity over the window).
    // deltaColor: per-channel colour spread (absolute, 0..1) below which a
    // window is similar. maxRadius caps the outward search (leap distance).
    // epsilon is the opacity threshold marking a voxel rendering-relevant; only
    // relevant voxels get a non-zero similarity distance (empty voxels are
    // skipped by the Chebyshev empty leap at runtime instead).
    void build(const Volume& volume, const TransferFunction& tf,
               float scalarMin, float scalarMax, float epsilon,
               float deltaAlpha, float deltaColor, int maxRadius);

    int3 dims() const { return dims_; }

    // Device-resident 3D texture object holding the similarity distance map.
    cudaTextureObject_t similarityTex() const { return d_similarityTex_; }

    size_t similarityBytes() const { return sim_.size(); }

private:
    void uploadToDevice();

    int3   dims_ = {};
    std::vector<unsigned char> sim_;       // similarity distance (voxels) per voxel
    cudaArray*          d_simArray_ = nullptr;
    cudaTextureObject_t d_similarityTex_ = 0;
};
