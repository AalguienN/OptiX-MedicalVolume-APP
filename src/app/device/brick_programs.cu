///////////////////////////////////////////////////////////////////////////////
// Bricked / tiled volume strategy device programs.
//
// This file is intentionally kept separate from the shared dense-baseline
// programs in device_programs.cu. It is NOT an independent compilation
// unit: it relies on the shared helpers / declarations from
// shared_device_programs.h and is #included from device_programs.cu so that
// everything still lands in a single OptiX IR module (one OptixModule) that
// the pipeline loads. Strategy-specific  programs live here, keeping the
// dense baseline file focused on the implementations it shares across modes.
//
// See "Bricked / tiled volume" in the thesis. The volume is subdivided into
// fixed-size bricks of n^3 voxels, each carrying BrickMeta metadata. The ray
// marches with the same fixed step size as the dense baseline, but whenever
// the current position falls inside a brick whose metadata marks it empty
// (no rendering-relevant voxels) the whole brick is skipped in a single step
// by jumping to the brick's exit along the ray.
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

static __forceinline__ __device__ int3 brickFromPos(float3 p)
{
    float3 local = componentDiv(p - params.volumeOrigin, params.brickSize);

    int3 b;
    b.x = min(params.brickCount.x - 1, max(0, __float2int_rd(local.x)));
    b.y = min(params.brickCount.y - 1, max(0, __float2int_rd(local.y)));
    b.z = min(params.brickCount.z - 1, max(0, __float2int_rd(local.z)));
    return b;
}

static __forceinline__ __device__ int brickLinearIndex(int3 b)
{
    return b.x + params.brickCount.x * (b.y + params.brickCount.y * b.z);
}

static __forceinline__ __device__ void brickWorldBounds(int3 b, float3& bmin, float3& bmax)
{
    bmin = params.volumeOrigin
         + componentMul(make_float3(static_cast<float>(b.x),
                                    static_cast<float>(b.y),
                                    static_cast<float>(b.z)), params.brickSize);
    bmax = bmin + params.brickSize;
}

static __forceinline__ __device__ float3 brickMarch(float3 origin, float3 direction, float tmin, float tmax)
{
    float3 volumeSize = make_float3(
        params.volumeDims.x * params.volumeSpacing.x,
        params.volumeDims.y * params.volumeSpacing.y,
        params.volumeDims.z * params.volumeSpacing.z);

    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;

    float t = tmin;
    int prevBrick = -1;

    for (int i = 0; i < 4096 && t <= tmax && accumA < 0.99f; ++i)
    {
        float3 samplePos = origin + direction * t;
        int3 brick = brickFromPos(samplePos);
        int bIdx = brickLinearIndex(brick);

        if (bIdx != prevBrick)
        {
            prevBrick = bIdx;

            if (!params.brickMeta[bIdx].relevant)
            {
                // Entire brick is empty: skip it in one step by jumping
                // to its exit intersection along the ray.
                float3 bmin, bmax;
                brickWorldBounds(brick, bmin, bmax);
                float bt0, bt1;
                if (intersectAABB(origin, direction, bmin, bmax, bt0, bt1))
                {
                    float tJump = fminf(tmax, fmaxf(tmin, bt1)) + stepSize * 0.01f;
                    int3 nb = brickFromPos(origin + direction * tJump);
                    if (brickLinearIndex(nb) == bIdx)
                        t += stepSize;  // exit face not cleared -> force real progress
                    else
                        t = tJump;
                }
                else
                {
                    t += stepSize;
                }
                prevBrick = -1;  // force brick re-lookup next iteration
                continue;
            }
        }

        float3 texCoord = componentDiv(samplePos - params.volumeOrigin, volumeSize);

        texCoord.x = fmaxf(0.0f, fminf(1.0f, texCoord.x));
        texCoord.y = fmaxf(0.0f, fminf(1.0f, texCoord.y));
        texCoord.z = fmaxf(0.0f, fminf(1.0f, texCoord.z));

        float scalar = tex3D<float>(params.volumeTex,
                                    texCoord.x, texCoord.y, texCoord.z);

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

    return make_float3(accumR, accumG, accumB);
}

// ============================================================
// Closest-hit program for the bricked / tiled strategy. The
// single-AABB intersection reports the volume entry/exit t-values;
// this program marches with empty-brick skipping via brickMarch().
// ============================================================
extern "C" __global__ void __closesthit__ch_brick()
{
    float tmin = __uint_as_float(optixGetAttribute_0());
    float tmax = __uint_as_float(optixGetAttribute_1());

    float3 color = brickMarch(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), tmin, tmax);
    setPayload(color);
}
