#include "adaptive_grid_marcher.h"

#include "check_macros.h"

#include <algorithm>
#include <array>
#include <cmath>

AdaptiveGridMarcher::~AdaptiveGridMarcher()
{
    if (d_distanceTex_)
        CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(d_distanceTex_));
    if (d_distanceArray_)
        CUDA_CHECK_NOEXCEPT(cudaFreeArray(d_distanceArray_));
}

void AdaptiveGridMarcher::build(const Volume& volume, const TransferFunction& tf,
                                float scalarMin, float scalarMax, float epsilon)
{
    const int nx = volume.dimX;
    const int ny = volume.dimY;
    const int nz = volume.dimZ;
    dims_ = make_int3(nx, ny, nz);

    // Occupancy: 1 for rendering-relevant (non-empty) voxels, 0 otherwise.
    // Relevant is defined by the transfer-function opacity reaching epsilon,
    // mirroring the bricked strategy so the two empty-space definitions agree.
    std::vector<unsigned char> occupied(static_cast<size_t>(nx) * ny * nz, 0);

    const float invRange = 1.0f / (scalarMax - scalarMin);
    const size_t sliceSize = static_cast<size_t>(nx) * ny;

    for (int z = 0; z < nz; ++z)
    {
        const size_t zOff = static_cast<size_t>(z) * sliceSize;
        for (int y = 0; y < ny; ++y)
        {
            const size_t off = zOff + static_cast<size_t>(y) * nx;
            for (int x = 0; x < nx; ++x)
            {
                const float s = volume.data[off + x];
                const float tf_t = (s - scalarMin) * invRange * 2047.0f;
                int idx = static_cast<int>(std::rint(tf_t));
                idx = std::max(0, std::min(2047, idx));
                occupied[off + x] = (tf.lut[idx].w >= epsilon) ? 1 : 0;
            }
        }
    }

    // Phases: higher than any possible distance so empty voxels start at
    // "infinity" before the propagation sweeps assign real values.
    const int INF = 255;
    dist_.assign(static_cast<size_t>(nx) * ny * nz, INF);

    // Chebyshev (L-infinity) distance transform by two raster sweeps.
    // The value at a voxel is the number of Chebyshev "hops" (max over the
    // three axes) to the nearest occupied voxel; occupied voxels are 0.
    // For the forward (lexicographic) sweep the already-computed neighbours
    // are those with dz=-1 (any dx,dy), dz=0 with dy=-1 (any dx), and
    // dz=0,dy=0 with dx=-1: 9 + 3 + 1 = 13 in total.
    const int N = 13;
    std::array<int, N> fdx = { -1,0,1, -1,0,1, -1,0,1, -1,0,1, -1 };
    std::array<int, N> fdy = { -1,-1,-1, 0,0,0, 1,1,1, -1,-1,-1, 0 };
    std::array<int, N> fdz = { -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0, 0 };

    // Forward sweep: x,y,z ascending, consider 13 lower-orthant neighbours.
    for (int z = 0; z < nz; ++z)
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
                unsigned char v = occupied[idx] ? 0 : INF;

                for (int k = 0; k < N; ++k)
                {
                    int ox = x + fdx[k], oy = y + fdy[k], oz = z + fdz[k];
                    if (ox < 0 || oy < 0 || oz < 0 || ox >= nx || oy >= ny || oz >= nz)
                        continue;
                    unsigned char d = dist_[(static_cast<size_t>(oz) * ny + oy) * nx + ox];
                    v = static_cast<unsigned char>(std::min<int>(v, d + 1));
                }

                dist_[idx] = v;
            }
        }
    }

    // Backward neighbours are the mirror image: dz=+1 (any dx,dy), dz=0 with
    // dy=+1 (any dx), and dz=0,dy=0 with dx=+1.
    std::array<int, N> bdx = { 1,0,-1, 1,0,-1, 1,0,-1, 1,0,-1, 1 };
    std::array<int, N> bdy = { 1,1,1, 0,0,0, -1,-1,-1, 1,1,1, 0 };
    std::array<int, N> bdz = { 1,1,1, 1,1,1, 1,1,1, 0,0,0, 0 };

    // Backward sweep: x,y,z descending, consider 13 upper-orthant neighbours.
    for (int z = nz - 1; z >= 0; --z)
    {
        for (int y = ny - 1; y >= 0; --y)
        {
            for (int x = nx - 1; x >= 0; --x)
            {
                size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
                unsigned char v = dist_[idx];

                for (int k = 0; k < N; ++k)
                {
                    int ox = x + bdx[k], oy = y + bdy[k], oz = z + bdz[k];
                    if (ox < 0 || oy < 0 || oz < 0 || ox >= nx || oy >= ny || oz >= nz)
                        continue;
                    unsigned char d = dist_[(static_cast<size_t>(oz) * ny + oy) * nx + ox];
                    v = static_cast<unsigned char>(std::min<int>(v, d + 1));
                }

                dist_[idx] = v;
            }
        }
    }

    {
        // Report how much of the volume is rendering-relevant and the range of
        // leap distances available, to confirm the skip structure is populated.
        unsigned int relevant = 0;
        unsigned int maxD = 0;
        const size_t total = dist_.size();
        for (size_t i = 0; i < total; ++i)
        {
            const unsigned char d = dist_[i];
            if (d == 0) ++relevant;
            if (d > maxD) maxD = d;
        }
        std::cout << "Adaptive occupancy: " << (relevant * 100.0 / total) << "% relevant, "
                  << "max Chebyshev distance " << maxD << " voxels\n";
    }

    uploadToDevice();
}

void AdaptiveGridMarcher::uploadToDevice()
{
    if (d_distanceTex_)
        return;

    cudaExtent extent = make_cudaExtent(dims_.x, dims_.y, dims_.z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaMalloc3DArray(&d_distanceArray_, &channelDesc, extent));

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr(
        const_cast<unsigned char*>(dist_.data()),
        dims_.x * sizeof(unsigned char),
        dims_.x,
        dims_.y);
    copyParams.dstArray = d_distanceArray_;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = d_distanceArray_;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    // Point (nearest) filtering: the device marcher must read the exact
    // stored Chebyshev distance so the leap guarantee is honoured. Linear
    // interpolation across an occupied boundary could shrink the value and,
    // more importantly, reading the value of the nearest voxel keeps the
    // (D-1) leap strictly within the guaranteed-empty region.
    texDesc.filterMode     = cudaFilterModePoint;
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&d_distanceTex_, &resDesc, &texDesc, nullptr));

    std::cout << "Adaptive distance map: " << dims_.x << "x" << dims_.y << "x" << dims_.z
              << " (" << dist_.size() / (1024.0 * 1024.0) << " MB)\n";
}
