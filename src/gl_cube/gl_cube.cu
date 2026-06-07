#include <optix.h>

#include "gl_cube.h"
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

    unsigned int u0 = 0, u1 = 0, u2 = 0;

    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,
        0,
        0,
        u0, u1, u2);

    float3 color = make_float3(
        __uint_as_float(u0),
        __uint_as_float(u1),
        __uint_as_float(u2));

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

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
    float3 v[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        optixGetPrimitiveIndex(),
        optixGetSbtGASIndex(),
        0.0f,
        v);

    float3 n = normalize(cross(v[1] - v[0], v[2] - v[0]));

    float3 ray_dir = optixGetWorldRayDirection();
    if (dot(n, ray_dir) > 0.0f)
        n = -n;

    setPayload(n * 0.5f + 0.5f);
}
