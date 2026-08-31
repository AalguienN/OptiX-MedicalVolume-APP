#include "nanovdb_volume.h"

#include "check_macros.h"

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <algorithm>
#include <cmath>
#include <iostream>

void NanoVDBVolume::build(const Volume& volume, const TransferFunction& tf,
                          float scalarMin, float scalarMax, float epsilon)
{
    using nanovdb::Coord;
    using nanovdb::Vec3d;

    scalarMin_ = scalarMin;
    scalarMax_ = scalarMax;
    invRange_  = 1.0f / (scalarMax - scalarMin);

    numActiveVoxels_ = 0;
    numRelevant_     = 0;
    totalVoxels_     = static_cast<size_t>(volume.dimX) * volume.dimY * volume.dimZ;

    // The grid background is the scalar at the low end of the CT/MR window,
    // which maps to (near) zero opacity in the shared transfer function.
    const float background = scalarMin;
    nanovdb::tools::build::Grid<float> grid(
        background, "medical_volume", nanovdb::GridClass::VoxelVolume);

    // Set per-axis voxel size (spacing) and origin so that NanoVDB index space
    // maps onto the world space shared by all other samplers:
    //   world = index * spacing + origin
    {
        const double sx = volume.spacingX;
        const double sy = volume.spacingY;
        const double sz = volume.spacingZ;
        const double mat[3][3] = { { sx, 0.0, 0.0 },
                                   { 0.0, sy, 0.0 },
                                   { 0.0, 0.0, sz } };
        const double inv[3][3] = { { 1.0 / sx, 0.0,      0.0 },
                                   { 0.0,      1.0 / sy, 0.0 },
                                   { 0.0,      0.0,      1.0 / sz } };
        grid.mMap.set(mat, inv, Vec3d(volume.originX, volume.originY, volume.originZ), 1.0);
    }

    auto acc = grid.getAccessor();

    // Populate the tree. Only voxels whose TF opacity >= epsilon (the same rule
    // used by every other sparse strategy to classify a voxel as relevant) are
    // stored with their raw scalar; everything else remains at the background,
    // yielding a sparse tree that NanoVDB's HDDA traversal can skip.
    size_t releCount = 0;
    size_t activeCount = 0;
    for (int z = 0; z < volume.dimZ; ++z)
    {
        for (int y = 0; y < volume.dimY; ++y)
        {
            for (int x = 0; x < volume.dimX; ++x)
            {
                const float s = volume.data[
                    (static_cast<size_t>(z) * volume.dimY + y) * volume.dimX + x];

                const float tf_t = (s - scalarMin) * invRange_ * 2047.0f;
                int idx = static_cast<int>(std::rint(tf_t));
                idx = std::max(0, std::min(2047, idx));
                const float op = tf.lut[idx].w;

                if (op < epsilon)
                    continue;   // leave as background (empty)

                ++releCount;
                ++activeCount;
                acc.setValue(Coord(x, y, z), s);
            }
        }
    }

    numRelevant_ = releCount;
    numActiveVoxels_ = activeCount;

    // Flatten the source tree into the linearized NanoVDB layout.
    hostGrid_ = nanovdb::tools::createNanoGrid(grid);
    if (!hostGrid_)
        throw std::runtime_error("NanoVDB: createNanoGrid returned an empty handle");

    uploadToDevice();
}

void NanoVDBVolume::uploadToDevice()
{
    if (d_grid_)
        return;

    const uint64_t bytes = hostGrid_.bufferSize();
    void* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, bytes));
    CUDA_CHECK(cudaMemcpy(devPtr, hostGrid_.data(), bytes, cudaMemcpyHostToDevice));
    d_grid_ = reinterpret_cast<const nanovdb::FloatGrid*>(devPtr);
}

nanovdb::CoordBBox NanoVDBVolume::indexBBox() const
{
    if (!hostGrid_)
        return nanovdb::CoordBBox();
    return hostGrid_.grid<nanovdb::FloatGrid>()->indexBBox();
}

NanoVDBVolume::~NanoVDBVolume()
{
    if (d_grid_)
        CUDA_CHECK_NOEXCEPT(cudaFree(const_cast<nanovdb::FloatGrid*>(d_grid_)));
}
