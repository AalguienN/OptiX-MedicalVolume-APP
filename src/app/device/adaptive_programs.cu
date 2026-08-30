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

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;

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

        // Adaptive step: read the Chebyshev distance to the nearest
        // non-empty voxel (in voxel units). If comfortably inside an empty
        // region (D >= 2) leap the safe (D-1) voxel length in one go;
        // otherwise fall back to the dense baseline half-voxel step so
        // material is sampled at identical density to the baseline.
        float D = tex3D<float>(params.distanceTex,
                               texCoord.x, texCoord.y, texCoord.z);
        int iD = __float2int_rd(D);
        if (iD >= 2)
        {
            t += (iD - 1) * params.minSpacing;
        }
        else
        {
            t += stepSize;
        }
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
