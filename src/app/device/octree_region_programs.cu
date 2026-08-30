///////////////////////////////////////////////////////////////////////////////
// Octree-regions strategy (--mode octree-regions, Variant B of the thesis)
// device programs.
//
// Combines Variant A (octree) with the hardware-skip idea of the
// bricked-regions strategy: the GAS contains one custom AABB primitive per
// rendering-relevant octree leaf (built by OctreeRegionsScene), so the OptiX
// BVH / RT cores skip empty space with the data-adaptive granularity of the
// octree rather than fixed bricks. Ray traversal is the same utility-ray loop
// as the bricked-regions strategy: each optixTrace returns the nearest
// remaining relevant leaf as [entry t, exit t, primitive index] in the
// payload, and the ray-generation program marches exactly that leaf's span
// before re-tracing from just past its exit. The march is identical to the
// dense baseline so the output is visually unchanged.
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

// regionMarchSeed (defined in bricked_regions_programs.cu, same translation
// unit) marches the span [tmin,tmax] with the fixed dense-baseline step,
// continuing from the accumulated RGBA. It is reused verbatim here.

// ============================================================
// Ray-generation program: utility-ray loop over non-empty octree leaves.
// ============================================================
extern "C" __global__ void __raygen__rg_region_octree()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    float3 origin    = rtData->cam_eye;
    float3 direction = normalize(d.x * rtData->camera_u + d.y * rtData->camera_v + rtData->camera_w);

    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;

    float t_cur = 0.0f;
    {
        float v0, v1;
        if (intersectAABB(origin, direction, params.volumeOrigin, params.volumeMax, v0, v1))
            t_cur = fmaxf(0.0f, v0);
    }

    for (int hops = 0; hops < 256 && accumA < 0.99f; ++hops)
    {
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u;  // [entry t, exit t, prim index]
        unsigned int missFlag = 0u;

        optixTrace(params.handle,
                   origin,
                   direction,
                   t_cur,
                   1e30f,
                   0.0f,
                   OptixVisibilityMask(0xFF),
                   OPTIX_RAY_FLAG_NONE,
                   0,
                   1,
                   0,
                   p0, p1, p2, missFlag);

        if (missFlag == 1u)
            break;

        float tHit  = __uint_as_float(p0);
        float tExit = __uint_as_float(p1);

        if (tHit >= 1e29f)
            break;

        regionMarchSeed(origin, direction, tHit, tExit, stepSize,
                        accumR, accumG, accumB, accumA);

        float advance = tExit + stepSize * 0.01f;
        if (tExit - tHit <= stepSize * 0.01f)
            advance = tExit + stepSize;
        t_cur = fminf(advance, 1e30f);

        if (tHit <= 0.0f && tExit <= tHit)
            break;
    }

    float3 color = make_float3(accumR, accumG, accumB);
    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

// ============================================================
// Intersection program: reads this primitive's octree-leaf AABB from the
// ocRegion array and reports entry/exit t (attributes 0,1) plus primitive
// index (attribute 2).
// ============================================================
extern "C" __global__ void __intersection__is_region_octree()
{
    const unsigned int primIdx = optixGetPrimitiveIndex();

    OptixAabb aabb = params.ocRegionAabbs[primIdx];
    float3 bmin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    float3 bmax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    float3 origin    = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float  tmin_ray  = optixGetRayTmin();
    float  tmax_ray  = optixGetRayTmax();

    float tmin, tmax;
    if (!intersectAABB(origin, direction, bmin, bmax, tmin, tmax))
        return;

    tmin = fmaxf(tmin, tmin_ray);
    tmax = fminf(tmax, tmax_ray);

    if (tmin <= tmax)
    {
        optixReportIntersection(tmin, 0,
            __float_as_uint(tmin),
            __float_as_uint(tmax),
            primIdx);
    }
}

// ============================================================
// Closest-hit program: forward the intersection-reported entry/exit t and
// primitive index to the ray-generation payload. Does not march.
// ============================================================
extern "C" __global__ void __closesthit__ch_ocregion()
{
    optixSetPayload_0(optixGetAttribute_0());
    optixSetPayload_1(optixGetAttribute_1());
    optixSetPayload_2(optixGetAttribute_2());
}

// ============================================================
// Miss program for the octree-regions ray loop: sets the miss flag so the
// ray-generation loop knows there are no more relevant leaves along the ray.
// ============================================================
extern "C" __global__ void __miss__ms_region_octree()
{
    optixSetPayload_3(1u);
}
