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
    // Robust per-axis slab test. Handles rays whose direction component is
    // exactly 0 (e.g. a camera perfectly aligned with an axis): in that case
    // the ray's coordinate must lie within the slab on that axis, otherwise it
    // never enters the box. A plain 1/d slab test would compute 0 * inf = NaN
    // when the ray's coordinate equals a box face, wrongly rejecting the hit.
    tmin = -3.402823466e+38f;   // -FLT_MAX
    tmax =  3.402823466e+38f;   // +FLT_MAX

    if (fabsf(direction.x) < 1e-12f)
    {
        if (origin.x < bmin.x || origin.x > bmax.x) return false;
    }
    else
    {
        float inv = 1.0f / direction.x;
        float t0 = (bmin.x - origin.x) * inv;
        float t1 = (bmax.x - origin.x) * inv;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }

    if (fabsf(direction.y) < 1e-12f)
    {
        if (origin.y < bmin.y || origin.y > bmax.y) return false;
    }
    else
    {
        float inv = 1.0f / direction.y;
        float t0 = (bmin.y - origin.y) * inv;
        float t1 = (bmax.y - origin.y) * inv;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }

    if (fabsf(direction.z) < 1e-12f)
    {
        if (origin.z < bmin.z || origin.z > bmax.z) return false;
    }
    else
    {
        float inv = 1.0f / direction.z;
        float t0 = (bmin.z - origin.z) * inv;
        float t1 = (bmax.z - origin.z) * inv;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }

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
