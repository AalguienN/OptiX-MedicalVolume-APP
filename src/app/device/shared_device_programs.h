#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../shared_device.h"
#include "helpers.h"         // make_color, vec_math (float3 operators)

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 componentMul(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3 componentDiv(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static __forceinline__ __device__ bool intersectAABB(
    float3 origin, float3 direction,
    float3 bmin, float3 bmax,
    float& tmin, float& tmax)
{
    float3 inv_dir = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    float3 t0 = componentMul(bmin - origin, inv_dir);
    float3 t1 = componentMul(bmax - origin, inv_dir);

    float3 tmin3 = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax3 = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
    tmax = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);

    return tmin <= tmax && tmax > 0.0f;
}

static __forceinline__ __device__ float3 volumeMarch(float3 origin, float3 direction, float tmin, float tmax)
{
    float3 volumeSize = make_float3(
        params.volumeDims.x * params.volumeSpacing.x,
        params.volumeDims.y * params.volumeSpacing.y,
        params.volumeDims.z * params.volumeSpacing.z);

    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    unsigned int nSamples = 0;

    int maxSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
    if (maxSteps > 4096) maxSteps = 4096;

    for (int i = 0; i < maxSteps && accumA < 0.99f; ++i)
    {
        float t = tmin + i * stepSize;
        float3 samplePos = origin + direction * t;
        float3 texCoord = componentDiv(samplePos - params.volumeOrigin, volumeSize);

        texCoord.x = fmaxf(0.0f, fminf(1.0f, texCoord.x));
        texCoord.y = fmaxf(0.0f, fminf(1.0f, texCoord.y));
        texCoord.z = fmaxf(0.0f, fminf(1.0f, texCoord.z));

        float scalar = tex3D<float>(params.volumeTex,
                                    texCoord.x, texCoord.y, texCoord.z);
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
    }

    if (params.dbgCounters)
        atomicAdd(&params.dbgCounters[0], nSamples);

    return make_float3(accumR, accumG, accumB);
}
