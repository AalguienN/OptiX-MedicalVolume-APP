#include <optix.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../shared_device.h"
#include "helpers.h"

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

    return make_float3(accumR, accumG, accumB);
}

// ============================================================
// Manual raygen program (no optixTrace — dense baseline)
// ============================================================
extern "C" __global__ void __raygen__rg()
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
        color = volumeMarch(origin, direction, tmin, tmax);
    }

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

// ============================================================
// OptiX raygen program — calls optixTrace, relies on
// intersection + closest-hit programs for volume traversal.
// ============================================================
extern "C" __global__ void __raygen__rg_optix()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    float3 origin    = rtData->cam_eye;
    float3 direction = normalize(d.x * rtData->camera_u + d.y * rtData->camera_v + rtData->camera_w);

    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);

    optixTrace(params.handle,
               origin,
               direction,
               0.0f,           // tmin
               1e30f,          // tmax
               0.0f,           // ray time
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_NONE,
               0,              // SBT offset
               1,              // SBT stride
               0,              // miss SBT index
               p0, p1, p2);

    float3 color = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2));

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

// ============================================================
// Intersection program — AABB slab test for the volume box.
// Reports entry (tmin) and exit (tmax) as OptiX attributes.
// ============================================================
extern "C" __global__ void __intersection__is()
{
    float3 origin    = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float  tmin_ray  = optixGetRayTmin();
    float  tmax_ray  = optixGetRayTmax();

    float tmin, tmax;
    if (!intersectAABB(origin, direction, params.volumeOrigin, params.volumeMax, tmin, tmax))
        return;

    tmin = fmaxf(tmin, tmin_ray);
    tmax = fminf(tmax, tmax_ray);

    if (tmin <= tmax)
    {
        optixReportIntersection(tmin, 0,
            __float_as_uint(tmin),
            __float_as_uint(tmax));
    }
}

// ============================================================
// Closest-hit program for volume path — receives entry/exit
// t-values via attributes and performs the full volume march.
// ============================================================
extern "C" __global__ void __closesthit__ch_vol()
{
    float tmin = __uint_as_float(optixGetAttribute_0());
    float tmax = __uint_as_float(optixGetAttribute_1());

    float3 color = volumeMarch(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), tmin, tmax);
    setPayload(color);
}

// ============================================================
// Stub programs (retained for pipeline validity)
// ============================================================
extern "C" __global__ void __miss__ms()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    float t = static_cast<float>(idx.y) / static_cast<float>(dim.y);
    float3 bg = make_float3(0.1f + 0.2f * t, 0.1f + 0.2f * t, 0.2f + 0.3f * t);
    setPayload(bg);
}

extern "C" __global__ void __closesthit__ch()
{
    setPayload(make_float3(1.0f, 0.0f, 1.0f));
}
