// NanoVDB strategy device programs (--mode nanovdb).
//
// Following the single-AABB-GAS + closest-hit-march pattern shared by the
// OPTIX / OCTREE strategies: the shared __raygen__rg_optix and
// __intersection__is report the volume entry/exit t-range, and this closest-hit
// program runs the march. The march keeps the SAME fixed step size, transfer
// function and front-to-back compositing as the dense baseline (volumeMarch) so
// the output is comparable, but replaces the dense 3D-texture lookup with a
// NanoVDB device Accessor lookup into the sparse grid, and skips fully-empty
// (background) tiles via NanoVDB's getDim() — its built-in empty-space skip.
//
// Empty-space skipping: getDim(ijk) returns the extent (a power of two, in
// voxels) of the tree node or constant tile that contains ijk, and 1 inside a
// leaf. For dim > 1 the whole node is guaranteed to be background (empty from
// the renderer's point of view), so the ray warps straight to the far face of
// that dim-aligned node along its own direction instead of sampling through it.
// For dim == 1 the ray is inside a populated leaf and is marched at the same
// density as the dense baseline. dbgCounters[0] counts actually-sampled voxels
// and dbgCounters[2] counts empty-node leaps.

#pragma once

#include "shared_device_programs.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/SampleFromVoxels.h>

static __device__ float3 nanovdbMarch(float3 worig, float3 wdir, float tmin, float tmax)
{
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(params.nanovdbGrid);
    auto acc = grid->tree().getAccessor();

    float minSpacing = fminf(params.volumeSpacing.x,
                     fminf(params.volumeSpacing.y, params.volumeSpacing.z));
    float stepSize = minSpacing * 0.5f;

    // Index-space representation of this ray. The NanoVDB map is the diagonal
    // world = index * spacing + origin set in NanoVDBVolume::build(), so a
    // world point at parameter t maps to index = iOrig + mDir * (t - tmin),
    // where mDir_j = wdir_j / spacing_j (index voxels per world mm).
    const nanovdb::Vec3f iOrig = grid->worldToIndexF(
        nanovdb::Vec3f(worig.x + wdir.x * tmin,
                       worig.y + wdir.y * tmin,
                       worig.z + wdir.z * tmin));
    const nanovdb::Vec3f mDir = grid->worldToIndexDirF(
        nanovdb::Vec3f(wdir.x, wdir.y, wdir.z));

    // World-time needed to traverse one index voxel along the ray, per axis
    // (0 when that direction component vanishes, i.e. never exit that axis).
    const float invM[3] = { fabsf(mDir[0]) > 1e-12f ? 1.0f / mDir[0] : 0.0f,
                            fabsf(mDir[1]) > 1e-12f ? 1.0f / mDir[1] : 0.0f,
                            fabsf(mDir[2]) > 1e-12f ? 1.0f / mDir[2] : 0.0f };

    // Index-space ray over the whole [tmin, tmax] span, passed to getDim for
    // forward compatibility (this NanoVDB revision does not use it internally).
    const nanovdb::Vec3f iDir  = mDir * (tmax - tmin);
    nanovdb::math::Ray<float> iRay(iOrig, iDir, 0.0f, iDir.length());

    // Trilinear sampler over the NanoVDB accessor (same interpolation order as
    // the baseline's hardware trilinear texture lookup).
    nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::AccessorType, 1> sample(acc);

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    unsigned int nSamples = 0, nLeaps = 0;

    int maxSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
    if (maxSteps > 4096) maxSteps = 4096;

    if (params.dbgCounters)
    {
        // [3] interval and [4] bounds coincide for NanoVDB: it marches the full
        // volume span, so the would-be fixed-step count is the same either way.
        atomicAdd(&params.dbgCounters[3], static_cast<unsigned int>(maxSteps));
        atomicAdd(&params.dbgCounters[4], static_cast<unsigned int>(maxSteps));
    }

    const float tEnd = tmax;
    float t = tmin;
    float tLeafExit = -1.0f;  // world t past which the current leaf run ends
    while (t < tEnd && accumA < 0.99f && maxSteps-- > 0)
    {
        // When no leaf run is active, classify the current position: either it
        // lies in a fully-empty node/tile (dim > 1, leap over it) or we have
        // entered a 8^3 leaf (dim == 1) and can march inside it in one run,
        // only re-querying getDim on the leaf exit plane.
        if (t >= tLeafExit)
        {
            const nanovdb::Vec3f indexPos = iOrig + mDir * (t - tmin);
            const nanovdb::Coord ijk = nanovdb::math::RoundDown<nanovdb::Coord>(indexPos);

            const uint32_t dim = acc.getDim(ijk, iRay);
            if (dim > 1)
            {
                // The sample lies in a fully-empty node or constant tile of extent
                // dim voxels. Its index-space region is [o, o + dim) along every
                // axis, with a dim-aligned origin o = ijk & ~(dim - 1) (VDB node
                // origins are aligned to their power-of-two size). Warp to the first
                // face of that box the ray exits: per axis, the world time at which
                // indexPos crosses the far boundary, then take the earliest one.
                const int mask = static_cast<int>(dim) - 1;
                const int o[3] = { ijk[0] & ~mask, ijk[1] & ~mask, ijk[2] & ~mask };

                float tExit = tEnd;
                #pragma unroll
                for (int a = 0; a < 3; ++a)
                {
                    if (mDir[a] > 0.0f)
                        tExit = fminf(tExit, t + ((float)(o[a] + (int)dim) - indexPos[a]) * invM[a]);
                    else if (mDir[a] < 0.0f)
                        tExit = fminf(tExit, t + ((float)o[a] - indexPos[a]) * invM[a]);
                }

                // Guarantee forward progress even under degenerate direction/mask
                // combinations, and never overshoot the volume exit.
                float tNext = tExit + fmaxf(stepSize, 1e-3f);
                if (tNext <= t) tNext = t + fmaxf(stepSize, 1e-3f);
                if (tNext > tEnd) tNext = tEnd;

                t = tNext;
                ++nLeaps;
                continue;
            }

            // Entered a populated leaf: compute the exit t of its 8-aligned box
            // so the upcoming samples run inside it without per-sample getDim.
            const int lo[3] = { ijk[0] & ~7, ijk[1] & ~7, ijk[2] & ~7 };
            float tExit = tEnd;
            #pragma unroll
            for (int a = 0; a < 3; ++a)
            {
                if (mDir[a] > 0.0f)
                    tExit = fminf(tExit, t + ((float)(lo[a] + 8) - indexPos[a]) * invM[a]);
                else if (mDir[a] < 0.0f)
                    tExit = fminf(tExit, t + ((float)lo[a] - indexPos[a]) * invM[a]);
            }
            if (tExit <= t) tExit = t + stepSize;  // guard degenerate exit
            tLeafExit = tExit;
        }

        // Sample at the current world position inside the leaf run.
        const nanovdb::Vec3f indexPos = iOrig + mDir * (t - tmin);
        const nanovdb::Coord ijk = nanovdb::math::RoundDown<nanovdb::Coord>(indexPos);
        float scalar;
        if (params.nanovdbNearest)
        {
            // Single accessor probe at the nearest voxel (box-filtered sampling).
            const nanovdb::Coord vox(nanovdb::math::RoundDown<nanovdb::Coord>(
                nanovdb::Vec3f(indexPos[0] + 0.5f, indexPos[1] + 0.5f, indexPos[2] + 0.5f)));
            scalar = acc.getValue(vox);
        }
        else
        {
            // tex3D reads at texel centers (i+0.5)/N, i.e. at integer index j it
            // interpolates between voxels j-1 and j; SampleFromVoxels interpolates
            // between floor(p) and floor(p)+1, so subtracting 0.5 reproduces the
            // dense sampler's convention at the same world positions.
            scalar = sample(nanovdb::Vec3f(indexPos[0] - 0.5f,
                                           indexPos[1] - 0.5f,
                                           indexPos[2] - 0.5f));
        }
        ++nSamples;

        float tf_t = (scalar - params.scalarMin) / (params.scalarMax - params.scalarMin) * 2047.0f;
        int tfIdx = __float2int_rn(tf_t);
        tfIdx = max(0, min(2047, tfIdx));
        float4 tfVal = params.tfData[tfIdx];

        float r = tfVal.x;
        float g = tfVal.y;
        float b = tfVal.z;
        float a = tfVal.w * stepSize;

        if (a > 0.001f)
        {
            float opacityFactor = (1.0f - accumA) * a;
            accumR += opacityFactor * r;
            accumG += opacityFactor * g;
            accumB += opacityFactor * b;
            accumA += opacityFactor;
        }

        t += stepSize;
    }

    if (params.dbgCounters)
    {
        atomicAdd(&params.dbgCounters[0], nSamples);
        atomicAdd(&params.dbgCounters[2], nLeaps);
    }

    return make_float3(accumR, accumG, accumB);
}

// Closest-hit program for the NanoVDB strategy. The shared __intersection__is
// reports entry/exit t-values as attributes; we march that span.
extern "C" __global__ void __closesthit__ch_nanovdb()
{
    float tmin = __uint_as_float(optixGetAttribute_0());
    float tmax = __uint_as_float(optixGetAttribute_1());

    float3 color = nanovdbMarch(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), tmin, tmax);
    setPayload(color);
}
