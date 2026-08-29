#include "bricked_volume.h"

#include "check_macros.h"

#include <algorithm>
#include <cmath>

BrickedVolume::~BrickedVolume()
{
    if (d_meta_)
        CUDA_CHECK_NOEXCEPT(cudaFree(d_meta_));
}

void BrickedVolume::build(const Volume& volume, const TransferFunction& tf,
                          int brickSize, float scalarMin, float scalarMax, float epsilon)
{
    const int bs = std::max(1, brickSize);

    brickDims_  = make_int3(bs, bs, bs);
    brickCount_ = make_int3((volume.dimX + bs - 1) / bs,
                            (volume.dimY + bs - 1) / bs,
                            (volume.dimZ + bs - 1) / bs);
    brickSize_  = make_float3(bs * volume.spacingX,
                              bs * volume.spacingY,
                              bs * volume.spacingZ);

    const int nx = brickCount_.x;
    const int ny = brickCount_.y;
    const int nz = brickCount_.z;

    meta_.assign(static_cast<size_t>(nx) * ny * nz, HostMeta{});
    numRelevant_ = 0;

    const float invRange = 1.0f / (scalarMax - scalarMin);

    for (int bz = 0; bz < nz; ++bz)
        for (int by = 0; by < ny; ++by)
            for (int bx = 0; bx < nx; ++bx)
            {
                const int vx0 = bx * bs, vy0 = by * bs, vz0 = bz * bs;
                const int vx1 = std::min(volume.dimX, vx0 + bs);
                const int vy1 = std::min(volume.dimY, vy0 + bs);
                const int vz1 = std::min(volume.dimZ, vz0 + bs);

                HostMeta& m = meta_[bz * ny * nx + by * nx + bx];
                m.minScalar  =  1e30f;
                m.maxScalar  = -1e30f;
                m.minOpacity =  1e30f;
                m.maxOpacity = -1e30f;

                bool found = false;
                for (int vz = vz0; vz < vz1; ++vz)
                    for (int vy = vy0; vy < vy1; ++vy)
                        for (int vx = vx0; vx < vx1; ++vx)
                        {
                            const float s = volume.data[
                                (static_cast<size_t>(vz) * volume.dimY + vy) * volume.dimX + vx];
                            m.minScalar = std::min(m.minScalar, s);
                            m.maxScalar = std::max(m.maxScalar, s);

                            const float tf_t = (s - scalarMin) * invRange * 2047.0f;
                            int idx = static_cast<int>(std::rint(tf_t));
                            idx = std::max(0, std::min(2047, idx));
                            const float op = tf.lut[idx].w;
                            m.minOpacity = std::min(m.minOpacity, op);
                            m.maxOpacity = std::max(m.maxOpacity, op);
                            found = true;
                        }

                if (!found)
                {
                    m.minScalar = m.maxScalar = 0.0f;
                    m.minOpacity = m.maxOpacity = 0.0f;
                    m.relevant  = false;
                }
                else
                {
                    m.relevant = (m.maxOpacity >= epsilon);
                    if (m.relevant)
                        ++numRelevant_;
                }
            }

    uploadToDevice();
}

void BrickedVolume::uploadToDevice()
{
    if (d_meta_)
        return;

    const size_t count = meta_.size();
    const size_t bytes = count * sizeof(BrickMeta);

    std::vector<BrickMeta> tmp(count);
    for (size_t i = 0; i < count; ++i)
    {
        const HostMeta& h = meta_[i];
        tmp[i] = { h.minScalar, h.maxScalar, h.minOpacity, h.maxOpacity,
                   h.relevant ? 1u : 0u };
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_meta_), bytes));
    CUDA_CHECK(cudaMemcpy(d_meta_, tmp.data(), bytes, cudaMemcpyHostToDevice));
}