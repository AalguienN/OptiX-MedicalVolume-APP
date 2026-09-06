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
    for (cudaTextureObject_t t : dirTextures_)
        if (t) CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(t));
    for (cudaArray* a : dirArrays_)
        if (a) CUDA_CHECK_NOEXCEPT(cudaFreeArray(a));
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

        unsigned long long d2 = 0, d10 = 0;
        for (size_t i = 0; i < total; ++i)
        {
            const unsigned char d = dist_[i];
            if (d >= 2) ++d2;
            if (d >= 10) ++d10;
        }
        std::cout << "Distance distribution: " << (d2 * 100.0 / total) << "% voxels D>=2, "
                  << (d10 * 100.0 / total) << "% voxels D>=10\n";
    }

    uploadToDevice();
}

// ============================================================
// Directional (quantized) distance map.
//
// The sphere of directions is partitioned into 6 face sectors (+-x, +-y,
// +-z). For every voxel the directional distance along a sector is the
// distance, in Chebyshev voxel steps along the sector's principal axis, to
// the nearest rendering-relevant voxel seen in that direction (255 if none
// within the capped radius). These six per-voxel values are independent of
// the ray direction, so the whole field is a K x V auxiliary grid computed
// with six directional scans (forward and backward along each axis).
// ============================================================

// Per-voxel axis-aligned directional distances (voxel Chebyshev units along
// each of the six principal directions), capped to 255.
void AdaptiveGridMarcher::axisFaceDistances(
    int x, int y, int z, const std::vector<unsigned char>& occupied,
    std::array<unsigned char, 6>& out) const
{
    const int nx = dims_.x, ny = dims_.y, nz = dims_.z;
    std::array<unsigned char, 6> dist = { 255, 255, 255, 255, 255, 255 };

    // Along X.
    {
        const int row = (z * ny + y);
        const int base = row * nx;
        unsigned char d = occupied[base + x] ? 0 : nx;
        for (int i = x + 1; i < nx && d < 255; ++i)
            if (occupied[base + i]) { d = i - x; break; }
        dist[0] = d; // +x
        d = occupied[base + x] ? 0 : nx;
        for (int i = x - 1; i >= 0 && d < 255; --i)
            if (occupied[base + i]) { d = x - i; break; }
        dist[1] = d; // -x
    }
    // Along Y.
    {
        const int base = (z * ny + y) * nx + x;
        unsigned char d = occupied[base] ? 0 : ny;
        for (int i = y + 1; i < ny && d < 255; ++i)
            if (occupied[(static_cast<size_t>(z) * ny + i) * nx + x]) { d = i - y; break; }
        dist[2] = d; // +y
        d = occupied[base] ? 0 : ny;
        for (int i = y - 1; i >= 0 && d < 255; --i)
            if (occupied[(static_cast<size_t>(z) * ny + i) * nx + x]) { d = y - i; break; }
        dist[3] = d; // -y
    }
    // Along Z.
    {
        const int base = (z * ny + y) * nx + x;
        unsigned char d = occupied[base] ? 0 : nz;
        for (int i = z + 1; i < nz && d < 255; ++i)
            if (occupied[i * ny * nx + y * nx + x]) { d = i - z; break; }
        dist[4] = d; // +z
        d = occupied[base] ? 0 : nz;
        for (int i = z - 1; i >= 0 && d < 255; --i)
            if (occupied[i * ny * nx + y * nx + x]) { d = z - i; break; }
        dist[5] = d; // -z
    }

    out = dist;
}

bool AdaptiveGridMarcher::buildDirectional(int sectors)
{
    if (sectors != DIR_FACES)
    {
        std::cout << "Adaptive directional: unsupported sector count " << sectors
                  << " (only " << DIR_FACES << " implemented)\n";
        return false;
    }
    if (dist_.empty())
        throw std::runtime_error("AdaptiveGridMarcher::buildDirectional called before build()");

    const int nx = dims_.x, ny = dims_.y, nz = dims_.z;
    const size_t V = dist_.size();

    // Recover the occupancy (dist_ == 0) used to build the scalar map. The
    // directional maps reuse the same rendering-relevant classification.
    std::vector<unsigned char> occupied(V, 0);
    for (size_t i = 0; i < V; ++i)
        occupied[i] = (dist_[i] == 0) ? 1 : 0;

    std::vector<unsigned char> axisDist(6 * V, 255);
    std::array<unsigned char, 6> face;
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
            {
                axisFaceDistances(x, y, z, occupied, face);
                const size_t idx = (static_cast<size_t>(z) * ny + y) * nx + x;
                for (int s = 0; s < 6; ++s)
                    axisDist[s * V + idx] = face[s];
            }

    // Fill the per-sector maps from the axis distances. A sector selects the
    // single axis distance that matches its direction (face sectors are
    // axis-aligned by construction); the host index layout matches the device
    // faceSectorFor() quantization (+-x, +-y, +-z).
    dirDist_.clear();
    dirDist_.resize(6, std::vector<unsigned char>(V, 255));
    for (size_t i = 0; i < V; ++i)
        for (int s = 0; s < 6; ++s)
            dirDist_[s][i] = axisDist[s * V + i];

    // Diagnostic: report per-sector mean distance of the empty voxels that
    // would produce a larger leap than the all-direction scalar map.
    {
        unsigned long long better = 0, empty = 0;
        for (size_t i = 0; i < V; ++i)
        {
            if (dist_[i] == 0) continue;
            ++empty;
            for (int s = 0; s < 6; ++s)
                if (dirDist_[s][i] > dist_[i]) { ++better; break; }
        }
        std::cout << "Directional sector map: " << 6 << " sectors x " << V / (1024.0 * 1024.0)
                  << " Mvoxels = " << 6 * V / (1024.0 * 1024.0) << " MB, "
                  << (better * 100.0 / (empty ? empty : 1)) << "% of empty voxels have a sector "
                  << "distance above the scalar map\n";
    }

    dirSectorCount_ = 6;
    uploadDirectionalToDevice();
    return true;
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
    // NOTE: the channel is unsigned char, so it must be read in normalized
    // float mode (value/255) and rescaled by 255 on the device; reading a
    // uchar texture with cudaReadModeElementType returns garbage via tex3D.
    texDesc.filterMode     = cudaFilterModePoint;
    texDesc.readMode       = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&d_distanceTex_, &resDesc, &texDesc, nullptr));

    std::cout << "Adaptive distance map: " << dims_.x << "x" << dims_.y << "x" << dims_.z
              << " (" << dist_.size() / (1024.0 * 1024.0) << " MB)\n";
}

void AdaptiveGridMarcher::uploadDirectionalToDevice()
{
    cudaExtent extent = make_cudaExtent(dims_.x, dims_.y, dims_.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

    dirArrays_.resize(dirSectorCount_, nullptr);
    dirTextures_.resize(dirSectorCount_, 0);

    for (int s = 0; s < dirSectorCount_; ++s)
    {
        cudaArray* array = nullptr;
        CUDA_CHECK(cudaMalloc3DArray(&array, &channelDesc, extent));

        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr(
            const_cast<unsigned char*>(dirDist_[s].data()),
            dims_.x * sizeof(unsigned char), dims_.x, dims_.y);
        copyParams.dstArray = array;
        copyParams.extent   = extent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));

        cudaResourceDesc resDesc = {};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0]  = cudaAddressModeClamp;
        texDesc.addressMode[1]  = cudaAddressModeClamp;
        texDesc.addressMode[2]  = cudaAddressModeClamp;
        texDesc.filterMode      = cudaFilterModePoint;
        texDesc.readMode        = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

        dirArrays_[s]   = array;
        dirTextures_[s] = tex;
    }

    std::cout << "Directional sector textures: " << dirSectorCount_ << " x "
              << dims_.x << "x" << dims_.y << "x" << dims_.z << " ("
              << (dirSectorCount_ * dirDist_[0].size()) / (1024.0 * 1024.0) << " MB total)\n";
}
