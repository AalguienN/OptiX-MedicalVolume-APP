// Dense-baseline / shared OptiX device programs.
//
// This file holds the implementations that the dense baseline uses and that
// are shared/reused across rendering modes: the manual raygen program, the
// OptiX-traced raygen program, the volume AABB intersection program, the
// volume closest-hit program and the stub miss/closest-hit programs. The
// common device helpers and __constant__ Params live in
// shared_device_programs.h. Strategy-specific programs (e.g. the bricked /
// tiled volume) live in dedicated .cu files which are #included below so that
// everything is still compiled into a single OptiX IR module (one OptixModule)
// that the pipeline loads.
#include "shared_device_programs.h"

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

// Strategy-specific device programs. #included so that all programs land in
// the same OptiX IR module (single OptixModule) that the pipeline loads.
#include "brick_programs.cu"
#include "adaptive_programs.cu"
#include "region_programs.cu"
#include "octree_programs.cu"
#include "octree_region_programs.cu"
#include "nanovdb_programs.cu"
