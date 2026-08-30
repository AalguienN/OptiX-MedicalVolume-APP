///////////////////////////////////////////////////////////////////////////////
// Adaptive-step grid marcher strategy device programs.
//
// This file is intentionally kept separate from the shared dense-baseline
// programs in device_programs.cu. It is NOT an independent compilation
// unit: it relies on the shared helpers / declarations from
// shared_device_programs.h and is #included from device_programs.cu so that
// everything still lands in a single OptiX IR module (one OptixModule) that
// the pipeline loads.
//
// See "Adaptive-step grid marcher" in the thesis. The dense voxel grid is
// left unmodified; a same-resolution Chebyshev distance map is precomputed
// (host side, AdaptiveGridMarcher) storing, per voxel, the distance to the
// nearest rendering-relevant (non-empty) voxel. During the march every step
// reads the distance map at the current position and advances by that
// distance (minus one, to stay strictly inside the guaranteed-empty region),
// so whole empty regions are skipped in a single sample while material is
// sampled at the same density as the dense baseline. The entire loop,
// including the empty-space step-size decision, runs in software inside a
// manual raygen program (no RT-core traversal); the single-AABB GAS handle is
// present but never optixTrace-d, mirroring the manual dense baseline.
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

// Adaptive-step volume marcher. origin/direction define the ray, [tmin,tmax]
// the volume entry/exit intersection. Returns the front-to-back composited
// colour. The distance map is sampled with point filtering to honour the leap
// guarantee: a sampled Chebyshev distance D means every voxel strictly within
// Chebyshev radius D of the sampled voxel is empty, so advancing by (D-1)
// voxels of world length never crosses a non-empty voxel.
static __forceinline__ __device__ float3 adaptiveMarch(
    float3 origin, float3 direction, float tmin, float tmax)
{
    float3 volumeSize = make_float3(
        params.volumeDims.x * params.volumeSpacing.x,
        params.volumeDims.y * params.volumeSpacing.y,
        params.volumeDims.z * params.volumeSpacing.z);

    // Base step matches the dense baseline (half the smallest voxel spacing).
    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    // World length of one "safe voxel step" along this ray, taking each axis's
    // own spacing into account. The Chebyshev distance map value D is a voxel
    // count valid independently in every axis, so advancing by (D-1) of these
    // per-axis voxel units keeps every axis within (D-1) of its own voxels and
    // therefore strictly inside the guaranteed-empty L-infinity ball.
    float3 invDir = make_float3(
        direction.x != 0.0f ? 1.0f / fabsf(direction.x) : 1e30f,
        direction.y != 0.0f ? 1.0f / fabsf(direction.y) : 1e30f,
        direction.z != 0.0f ? 1.0f / fabsf(direction.z) : 1e30f);
    float voxelWorldStep = fminf(params.volumeSpacing.x * invDir.x,
                          fminf(params.volumeSpacing.y * invDir.y,
                                params.volumeSpacing.z * invDir.z));

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    unsigned int nSamples = 0;
    unsigned int nDistReads = 0;
    unsigned int nLeaps = 0;

    float t = tmin;
    int maxSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
    if (maxSteps > 4096) maxSteps = 4096;

    for (int i = 0; i < maxSteps && t <= tmax && accumA < 0.99f; ++i)
    {
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

        // Adaptive step sizing. Advance at least the dense baseline step every
        // iteration. Only when the current sample is transparent (its transfer
        // function opacity is below the relevance threshold, i.e. it cannot be a
        // rendering-relevant voxel) do we consult the distance map to leap over
        // the surrounding empty space. Reading the distance map in this
        // conditional way keeps the cost inside opaque material identical to the
        // dense baseline (no extra texture fetch per step), while retaining the
        // empty-space skip. If D >= 2 the voxel is two or more voxels from any
        // relevant voxel, so (D-1) of these per-axis voxel units advance strictly
        // inside the guaranteed-empty region and never skip a non-empty voxel.
        float D = 0.0f;
        if (tfVal.w < params.epsilon)
        {
            D = 255.0f * tex3D<float>(params.distanceTex, texCoord.x, texCoord.y, texCoord.z);
            ++nDistReads;
        }

        int iD = __float2int_rd(D);
        if (iD >= 2)
        {
            t += (iD - 1) * voxelWorldStep;
            ++nLeaps;
        }
        else
        {
            t += stepSize;
        }
    }

    if (params.dbgCounters)
    {
        atomicAdd(&params.dbgCounters[0], nSamples);
        atomicAdd(&params.dbgCounters[1], nDistReads);
        atomicAdd(&params.dbgCounters[2], nLeaps);
    }

    return make_float3(accumR, accumG, accumB);
}

// ============================================================
// Manual raygen program for the adaptive-step strategy (no optixTrace).
// Performs the same software AABB entry/exit test as the manual dense
// baseline, then runs adaptiveMarch() for the volume traversal.
// ============================================================
extern "C" __global__ void __raygen__rg_adaptive()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    float3 origin    = rtData->cam_eye;
    float3 direction = normalize(d.x * rtData->camera_u + d.y * rtData->camera_v + rtData->camera_w);

    float3 color = make_float3(0.1f, 0.1f, 0.2f);

    float tmin, tmax;
    if (intersectAABB(origin, direction, params.volumeOrigin, params.volumeMax, tmin, tmax))
    {
        if (tmin < 0.0f) tmin = 0.0f;
        color = adaptiveMarch(origin, direction, tmin, tmax);
    }

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}
