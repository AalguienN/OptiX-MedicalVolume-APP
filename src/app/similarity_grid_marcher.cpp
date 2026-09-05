#include "similarity_grid_marcher.h"

#include "check_macros.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

// Safety cap on the outward Chebyshev shell search. Leaps beyond this many
// voxels are not attempted.
constexpr int kMaxRadius = 16;

// Coarse-cell size for the min/max acceleration grid. A Chebyshev ball of
// radius <= kMaxRadius has width <= 2*kMaxRadius+1, so with this cell size it
// overlaps at most 3 cells per axis and therefore <= 27 cells total.
constexpr int kCellSize = 2 * kMaxRadius;

struct CellStats
{
    float minA, maxA;
    float minR, maxR, minG, maxG, minB, maxB;
};

}  // namespace

SimilarityGridMarcher::~SimilarityGridMarcher()
{
    if (d_similarityTex_)
        CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(d_similarityTex_));
    if (d_simArray_)
        CUDA_CHECK_NOEXCEPT(cudaFreeArray(d_simArray_));
}

void SimilarityGridMarcher::build(const Volume& volume, const TransferFunction& tf,
                                  float scalarMin, float scalarMax, float epsilon,
                                  float deltaAlpha, float deltaColor, int maxRadius)
{
    const int nx = volume.dimX;
    const int ny = volume.dimY;
    const int nz = volume.dimZ;
    dims_ = make_int3(nx, ny, nz);

    const int R = std::max(1, std::min(maxRadius, kMaxRadius));
    const float invRange = 1.0f / (scalarMax - scalarMin);

    const size_t total = static_cast<size_t>(nx) * ny * nz;
    const size_t sliceSize = static_cast<size_t>(nx) * ny;

    auto mapTF = [&](float s) -> float4
    {
        const float t = (s - scalarMin) * invRange * 2047.0f;
        int idx = static_cast<int>(std::rint(t));
        idx = std::max(0, std::min(2047, idx));
        return tf.lut[idx];
    };

    // Partition the volume into C^3 cells and, in one O(N) pass, record the
    // true min/max of opacity and each colour channel per cell. A Chebyshev
    // ball of radius k overlaps a small set of cells; the combined min/max of
    // those cells over-approximates the ball's true range. This is the key:
    // it is conservative (can only shrink the leap, never make it unsafely
    // large) and turns the intractable O(N*R^3) brute force into an
    // O(N*R*27) build with MB-scale memory.
    const int nxc = (nx + kCellSize - 1) / kCellSize;
    const int nyc = (ny + kCellSize - 1) / kCellSize;
    const int nzc = (nz + kCellSize - 1) / kCellSize;
    const size_t nCells = static_cast<size_t>(nxc) * nyc * nzc;
    std::vector<CellStats> cells(nCells, CellStats{ 1e30f, -1e30f, 1e30f, -1e30f,
                                                     1e30f, -1e30f, 1e30f, -1e30f });

    for (int z = 0; z < nz; ++z)
    {
        const size_t zOff = static_cast<size_t>(z) * sliceSize;
        const int czc = z / kCellSize;
        const size_t cellBase = static_cast<size_t>(czc) * nyc * nxc;
        for (int y = 0; y < ny; ++y)
        {
            const size_t row = zOff + static_cast<size_t>(y) * nx;
            const int cyc = y / kCellSize;
            const size_t rowBase = (cellBase + static_cast<size_t>(cyc) * nxc);
            for (int x = 0; x < nx; ++x)
            {
                const float4 tfv = mapTF(volume.data[row + static_cast<size_t>(x)]);
                CellStats& c = cells[rowBase + static_cast<size_t>(x / kCellSize)];
                c.minA = std::min(c.minA, tfv.w); c.maxA = std::max(c.maxA, tfv.w);
                c.minR = std::min(c.minR, tfv.x); c.maxR = std::max(c.maxR, tfv.x);
                c.minG = std::min(c.minG, tfv.y); c.maxG = std::max(c.maxG, tfv.y);
                c.minB = std::min(c.minB, tfv.z); c.maxB = std::max(c.maxB, tfv.z);
            }
        }
    }

    auto cellAt = [&](int czc_, int cyc_, int cxc_) -> const CellStats&
    {
        return cells[(static_cast<size_t>(czc_) * nyc + cyc_) * nxc + cxc_];
    };

    // Second pass: the similarity distance for each rendering-relevant voxel.
    // sim[v] = largest radius r in [1,R] whose whole Chebyshev ball is within
    // delta of v's opacity/colour (over-approximated via cells). A voxel with
    // no similar radius >= 2 (or that is empty) gets a value that disables the
    // leap. Irrelevant voxels are set to 0 and skipped, matching the device
    // predicate (only opacity>=epsilon reaches the homogeneity leap).
    sim_.assign(total, static_cast<unsigned char>(0));

    unsigned long long relevant = 0;
    for (int z = 0; z < nz; ++z)
    {
        const size_t zOff = static_cast<size_t>(z) * sliceSize;
        for (int y = 0; y < ny; ++y)
        {
            const size_t row = zOff + static_cast<size_t>(y) * nx;
            for (int x = 0; x < nx; ++x)
            {
                const size_t i = row + static_cast<size_t>(x);
                const float4 anchor = mapTF(volume.data[i]);

                // Empty (rendering-irrelevant): no homogeneity leap ever uses
                // this voxel, and the coarsened min/max can only over-approximate
                // the range, so skip it entirely.
                if (anchor.w < epsilon) continue;
                ++relevant;

                // Range of the radius-k ball accumulates monotonically as k
                // grows (the covered cell set is a superset for larger k), so
                // track the running min/max of opacity and colour.
                float mnA = anchor.w, mxA = anchor.w;
                float mnR = anchor.x, mxR = anchor.x;
                float mnG = anchor.y, mxG = anchor.y;
                float mnB = anchor.z, mxB = anchor.z;

                unsigned char sim = static_cast<unsigned char>(R);
                for (int k = 1; k <= R; ++k)
                {
                    const int bx0 = std::max(0, x - k), bx1 = std::min(nx - 1, x + k);
                    const int by0 = std::max(0, y - k), by1 = std::min(ny - 1, y + k);
                    const int bz0 = std::max(0, z - k), bz1 = std::min(nz - 1, z + k);
                    const int gz0 = bz0 / kCellSize, gz1 = bz1 / kCellSize;
                    const int gy0 = by0 / kCellSize, gy1 = by1 / kCellSize;
                    const int gx0 = bx0 / kCellSize, gx1 = bx1 / kCellSize;

                    for (int czc2 = gz0; czc2 <= gz1; ++czc2)
                        for (int cyc2 = gy0; cyc2 <= gy1; ++cyc2)
                            for (int cxc2 = gx0; cxc2 <= gx1; ++cxc2)
                            {
                                const CellStats& c = cellAt(czc2, cyc2, cxc2);
                                mnA = std::min(mnA, c.minA); mxA = std::max(mxA, c.maxA);
                                mnR = std::min(mnR, c.minR); mxR = std::max(mxR, c.maxR);
                                mnG = std::min(mnG, c.minG); mxG = std::max(mxG, c.maxG);
                                mnB = std::min(mnB, c.minB); mxB = std::max(mxB, c.maxB);
                            }

                    // If the (over-approximated) radius-k range leaves delta
                    // versus the anchor, the largest fully-similar radius is k-1.
                    if (mxA - anchor.w >= deltaAlpha || anchor.w - mnA >= deltaAlpha ||
                        mxR - anchor.x >= deltaColor || anchor.x - mnR >= deltaColor ||
                        mxG - anchor.y >= deltaColor || anchor.y - mnG >= deltaColor ||
                        mxB - anchor.z >= deltaColor || anchor.z - mnB >= deltaColor)
                    {
                        sim = static_cast<unsigned char>(k - 1);
                        break;
                    }
                }
                sim_[i] = sim;
            }
        }
    }

    unsigned int maxD = 0;
    unsigned long long d2 = 0, d4 = 0;
    for (size_t i = 0; i < total; ++i)
    {
        const unsigned char d = sim_[i];
        if (d > maxD) maxD = d;
        if (d >= 2) ++d2;
        if (d >= 4) ++d4;
    }
    std::cout << "Similarity map: " << relevant << "/" << total << " relevant, "
              << "max similar radius " << static_cast<int>(maxD) << " voxels; "
              << (relevant ? d2 * 100.0 / relevant : 0.0) << "% relevant voxels D>=2, "
              << (relevant ? d4 * 100.0 / relevant : 0.0) << "% D>=4\n";

    uploadToDevice();
}

void SimilarityGridMarcher::uploadToDevice()
{
    if (d_similarityTex_)
        return;

    cudaExtent extent = make_cudaExtent(dims_.x, dims_.y, dims_.z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaMalloc3DArray(&d_simArray_, &channelDesc, extent));

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr(
        const_cast<unsigned char*>(sim_.data()),
        dims_.x * sizeof(unsigned char),
        dims_.x,
        dims_.y);
    copyParams.dstArray = d_simArray_;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = d_simArray_;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    // Point (nearest) filtering: the device must read the exact stored
    // distance so the (D-1) leap stays inside the similar window. The channel
    // is unsigned char, read in normalized float mode and rescaled by 255.
    texDesc.filterMode     = cudaFilterModePoint;
    texDesc.readMode       = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&d_similarityTex_, &resDesc, &texDesc, nullptr));

    std::cout << "Similarity distance map: " << dims_.x << "x" << dims_.y << "x" << dims_.z
              << " (" << sim_.size() / (1024.0 * 1024.0) << " MB)\n";
}
