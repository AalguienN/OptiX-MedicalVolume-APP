#include <optix.h>

#include "optixConsole.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

constexpr unsigned int SBT_STRIDE_COLLAPSE = 0;

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float3*                prd)
{
    unsigned int p0, p1, p2;
    p0 = __float_as_uint(prd->x);
    p1 = __float_as_uint(prd->y);
    p2 = __float_as_uint(prd->z);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,
        SBT_STRIDE_COLLAPSE,
        0,
        p0, p1, p2);
    prd->x = __uint_as_float(p0);
    prd->y = __uint_as_float(p1);
    prd->z = __uint_as_float(p2);
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    const float3 origin    = rtData->cam_eye;
    const float3 direction = normalize(d.x * U + d.y * V + W);
    float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    trace(params.handle,
          origin,
          direction,
          0.00f,
          1e16f,
          &payload_rgb);

    params.image[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
}

extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3    payload = getPayload();
    setPayload(make_float3(rt_data->r, rt_data->g, rt_data->b));
}

extern "C" __global__ void __closesthit__ch()
{
    float t_hit = optixGetRayTmax();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);

    float3 world_raypos = ray_orig + t_hit * ray_dir;
    float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace(world_raypos);
    float3 obj_normal   = (obj_raypos - make_float3(q)) / q.w;
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));

    setPayload(world_normal * 0.5f + 0.5f);
}
