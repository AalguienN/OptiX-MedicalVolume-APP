///////////////////////////////////////////////////////////////////////////////
// Bricked-regions strategy (--mode bricked-regions) device programs: the
// bricked method but with hardware RT-core empty-skip.
//
// The GAS contains one custom AABB primitive per rendering-relevant brick
// (built by BrickedRegionsScene); empty bricks have no primitive, so the OptiX
// BVH / RT cores skip empty space in hardware. Ray traversal is a *utility-ray
// loop* running in the ray-generation program: each optixTrace (with
// maxTraceDepth = 1) returns the nearest remaining non-empty brick as
// [entry t, exit t, primitive index] carried through the payload; the raygen
// then marches exactly that brick's span and re-traces from just past its
// exit until opacity saturates or the ray misses. This composites multiple
// separated bricks front-to-back without any recursive tracing, keeping the
// pipeline non-recursive (shallow continuation stack).
//
// The closest-hit program does NOT march; it only forwards the intersection-
// reported entry/exit t and primitive index to the ray-generation payload.
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

// regionMarch (defined in adaptive_programs.cu, same translation unit) marches
// the span [tmin,tmax] continuing from the accumulated RGBA. It is the shared
// intra-region traversal: the adaptive-step variant when params.useAdaptive is
// set (--adaptive-march on, default) and the dense-baseline fixed-step variant
// otherwise.

// ============================================================
// Ray-generation program: utility-ray loop over non-empty bricks.
// Each optixTrace returns the nearest remaining brick's [entry, exit, prim]
// via the payload; the raygen marches that span and continues.
// ============================================================
extern "C" __global__ void __raygen__rg_bricked_regions()
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

    // Start from the volume entry so re-traces never re-visit geometry in
    // front of the current brick.
    float t_cur = 0.0f;
    {
        float v0, v1;
        if (intersectAABB(origin, direction, params.volumeOrigin, params.volumeMax, v0, v1))
        {
            t_cur = fmaxf(0.0f, v0);

            // total_steps_bounds ([3]): steps over the full volume AABB span
            // for this ray, counted once here (before any empty-space skip).
            float vmin = fmaxf(0.0f, v0), vmax = v1;
            int bSteps = static_cast<int>((vmax - vmin) / stepSize) + 1;
            if (bSteps > 4096) bSteps = 4096;
            if (params.dbgCounters)
                atomicAdd(&params.dbgCounters[3], static_cast<unsigned int>(bSteps));
        }
    }

    unsigned int nRegionSkips = 0;
    for (int hops = 0; hops < 256 && accumA < 0.99f; ++hops)
    {
        unsigned int p0 = 0u, p1 = 0u, p2 = 0u;  // [entry t, exit t, prim index]
        unsigned int missFlag = 0u;              // 1 => miss (set by miss program)

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

        if (missFlag == 1u)   // miss: no more non-empty bricks along the ray
            break;

        float tHit   = __uint_as_float(p0);
        float tExit  = __uint_as_float(p1);

        if (tHit >= 1e29f)
            break;

        // Each found non-empty brick means the hardware BVH skipped the empty
        // region preceding it, so count it as a leap (empty-region skip).
        ++nRegionSkips;

        regionMarch(origin, direction, tHit, tExit, stepSize,
                    accumR, accumG, accumB, accumA);

        // Advance past this hit. For a legitimate non-empty brick the exit t is
        // well beyond the entry, and a tiny nudge past the exit is enough. But a
        // ray lying exactly on a brick face can produce a degenerate corner-
        // grazing hit whose entry/exit are identical (zero-length span); the tiny
        // nudge would then loop forever on that single point. In that case force
        // a full sample-step advance so the ray escapes and continues to the next
        // real brick (same robustness as the shared intersectAABB zero-direction
        // fix used by the octree traversal).
        float advance = tExit + stepSize * 0.01f;
        if (tExit - tHit <= stepSize * 0.01f)
            advance = tExit + stepSize;
        t_cur = fminf(advance, 1e30f);
    }

    if (params.dbgCounters)
        atomicAdd(&params.dbgCounters[2], nRegionSkips);

    float3 color = make_float3(accumR, accumG, accumB);
    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

// ============================================================
// Intersection program: reads this primitive's brick AABB from the
// brickedRegions array and reports entry/exit t (attributes 0,1) plus
// primitive index (attribute 2). Invoked only for primitives (non-empty
// bricks) whose AABB the hardware determined the ray may overlap.
// ============================================================
extern "C" __global__ void __intersection__is_bricked_regions()
{
    const unsigned int primIdx = optixGetPrimitiveIndex();

    OptixAabb aabb = params.brickedRegionsAabbs[primIdx];
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
extern "C" __global__ void __closesthit__ch_bricked_regions()
{
    optixSetPayload_0(optixGetAttribute_0());
    optixSetPayload_1(optixGetAttribute_1());
    optixSetPayload_2(optixGetAttribute_2());
}

// ============================================================
// Miss program for the bricked-regions (and octree-regions) ray loop: sets
// the miss flag (payload 3) so the ray-generation loop knows there are no more
// non-empty bricks along the ray. Shared with the octree-regions strategy.
extern "C" __global__ void __miss__ms_region()
{
    optixSetPayload_3(1u);
}
