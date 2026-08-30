///////////////////////////////////////////////////////////////////////////////
// Region strategy (--mode region, Variant B) device programs.
//
// The GAS contains one custom AABB primitive per rendering-relevant brick
// (built by RegionsScene); empty bricks have no primitive, so the OptiX BVH
// / RT cores skip empty space in hardware. Ray traversal is a *utility-ray
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

// March the span [tmin,tmax] with the dense-baseline fixed step size,
// continuing from an existing accumulated RGBA so multiple bricks on one ray
// composite correctly. Mirrors shared_device_programs.h volumeMarch().
static __forceinline__ __device__ void regionMarchSeed(
    float3 origin, float3 direction, float tmin, float tmax,
    float stepSize, float& accumR, float& accumG, float& accumB, float& accumA)
{
    float3 volumeSize = make_float3(
        params.volumeDims.x * params.volumeSpacing.x,
        params.volumeDims.y * params.volumeSpacing.y,
        params.volumeDims.z * params.volumeSpacing.z);

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
}

// ============================================================
// Ray-generation program: utility-ray loop over non-empty bricks.
// Each optixTrace returns the nearest remaining brick's [entry, exit, prim]
// via the payload; the raygen marches that span and continues.
// ============================================================
extern "C" __global__ void __raygen__rg_region()
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
            t_cur = fmaxf(0.0f, v0);
    }

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

        regionMarchSeed(origin, direction, tHit, tExit, stepSize,
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

    float3 color = make_float3(accumR, accumG, accumB);
    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}

// ============================================================
// Intersection program: reads this primitive's brick AABB from the region
// array and reports entry/exit t (attributes 0,1) plus primitive index
// (attribute 2). Invoked only for primitives (non-empty bricks) whose AABB
// the hardware determined the ray may overlap.
// ============================================================
extern "C" __global__ void __intersection__is_region()
{
    const unsigned int primIdx = optixGetPrimitiveIndex();

    OptixAabb aabb = params.regionAabbs[primIdx];
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
extern "C" __global__ void __closesthit__ch_region()
{
    optixSetPayload_0(optixGetAttribute_0());
    optixSetPayload_1(optixGetAttribute_1());
    optixSetPayload_2(optixGetAttribute_2());
}

// ============================================================
// Miss program for the region ray loop: sets the miss flag (payload 3) so the
// ray-generation loop knows there are no more non-empty bricks along the ray.
// ============================================================
extern "C" __global__ void __miss__ms_region()
{
    optixSetPayload_3(1u);
}
