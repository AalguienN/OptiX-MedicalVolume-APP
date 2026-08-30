#pragma once

#include "transfer_function.h"
#include "volume.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

// Host-side construction of the NanoVDB sparse-volume representation
// (--mode nanovdb). NanoVDB retains OpenVDB's hierarchical tree layout but in
// a compact, pointer-free, contiguous buffer that can be uploaded to the GPU
// "as-is" and traversed directly from OptiX device code with its own accessor
// and empty-space-skipping HDDA traversal.
//
// This class converts the dense CT/MRI volume to a NanoVDB FloatGrid using
// only NanoVDB's own host tools (tools::GridBuilder + tools::createNanoGrid),
// i.e. without an OpenVDB dependency. Every rendering-relevant voxel (the same
// TF-opacity >= epsilon test used by the other sparse strategies) is stored
// with its raw scalar value; all other voxels are left at the grid background,
// which maps to zero opacity in the shared transfer function. The transform
// (Map) is set so that NanoVDB index space maps exactly onto the world space
// convention shared by the other samplers (world = index * spacing + origin).
class NanoVDBVolume
{
public:
    NanoVDBVolume() = default;
    ~NanoVDBVolume();

    NanoVDBVolume(const NanoVDBVolume&)            = delete;
    NanoVDBVolume& operator=(const NanoVDBVolume&) = delete;

    void build(const Volume& volume, const TransferFunction& tf,
               float scalarMin, float scalarMax, float epsilon);

    // Pointer to the uploaded NanoVDB FloatGrid on device (castable on the
    // device side to const nanovdb::FloatGrid*).
    const void* deviceGrid() const { return reinterpret_cast<const void*>(d_grid_); }

    // Byte size of the flat NanoVDB buffer (stored on GPU).
    size_t gridBytes()  const { return hostGrid_.bufferSize(); }
    size_t hostBytes()  const { return hostGrid_.bufferSize(); }

    // Number of active (stored) voxels and derived sparsity statistics.
    size_t numActiveVoxels() const { return numActiveVoxels_; }
    size_t numRelevant()     const { return numRelevant_; }
    float  sparsityRatio()   const
    {
        return totalVoxels_ ? 1.0f - static_cast<float>(numRelevant_) /
                                        static_cast<float>(totalVoxels_) : 0.0f;
    }
    size_t totalVoxels() const { return totalVoxels_; }

    // Grid extent in index space (used to size the volume GAS consistently
    // with the dense world-space AABB used by the other strategies).
    nanovdb::CoordBBox indexBBox() const;

private:
    void uploadToDevice();

    nanovdb::GridHandle<nanovdb::HostBuffer> hostGrid_;
    const nanovdb::FloatGrid*                d_grid_ = nullptr;

    size_t numActiveVoxels_ = 0;
    size_t numRelevant_     = 0;
    size_t totalVoxels_     = 0;
    float  invRange_        = 0.0f;
    float  scalarMin_       = 0.0f;
    float  scalarMax_       = 0.0f;
};
