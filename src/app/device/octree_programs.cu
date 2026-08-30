///////////////////////////////////////////////////////////////////////////////
// SVO / Octree strategy (--mode octree, Variant A) device programs.
//
// True hierarchical octree ray traversal, in contrast to the original slow
// badoctree experiment which performed a flat dense-grid march with a full
// root-to-leaf octree descent per sample. Here the ray descends the octree,
// skipping empty subtrees at the coarsest containing level, marching only the
// (relevant) leaf regions it actually crosses, front-to-back.
//
// The GAS is a single top-level AABB (same as the dense baseline); the
// intersection program reports the volume entry/exit t-values and the
// closest-hit program octreeTraverse() walks the octree over that span.
// Traversal is iterative with an explicit stack (bounded by octree depth x 8)
// to avoid device-stack recursion. Within a relevant leaf the dense grid is
// sampled on the same fixed step grid as the baseline, so the output is
// visually identical to the dense baseline.
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

#define OCT_MAX_STACK 512u

namespace {

struct OctStackEntry
{
    unsigned int node;
    float t0;
    float t1;
};

__forceinline__ __device__ OctreeNode octreeNodeAt(unsigned int i)
{
    return params.octreeNodes[i];
}

// World-space AABB of a node's voxel extent [voxelMin, voxelMax) * spacing.
__forceinline__ __device__ void octreeNodeWorldBounds(
    const OctreeNode& node, float3& bmin, float3& bmax)
{
    bmin = params.volumeOrigin + make_float3(
        static_cast<float>(node.voxelMin[0]) * params.volumeSpacing.x,
        static_cast<float>(node.voxelMin[1]) * params.volumeSpacing.y,
        static_cast<float>(node.voxelMin[2]) * params.volumeSpacing.z);
    bmax = params.volumeOrigin + make_float3(
        static_cast<float>(node.voxelMax[0]) * params.volumeSpacing.x,
        static_cast<float>(node.voxelMax[1]) * params.volumeSpacing.y,
        static_cast<float>(node.voxelMax[2]) * params.volumeSpacing.z);
}

// March the span [tmin,tmax] with the dense-baseline fixed step size,
// continuing from an existing accumulated RGBA (front-to-back compositing,
// early termination, same transfer function as the baseline).
__forceinline__ __device__ void marchLeafRegion(
    float3 origin, float3 direction, float tmin, float tmax, float stepSize,
    float& accumR, float& accumG, float& accumB, float& accumA)
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

}  // namespace

// Iterative front-to-back octree traversal over [tmin,tmax].
static __forceinline__ __device__ float3 octreeTraverse(
    float3 origin, float3 direction, float tmin, float tmax)
{
    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    unsigned int nLeaps = 0;

    OctStackEntry stack[OCT_MAX_STACK];
    unsigned int sp = 0;
    stack[sp++] = { 0u, tmin, tmax };

    OctStackEntry childT[8];

    while (sp > 0 && accumA < 0.99f && sp < OCT_MAX_STACK)
    {
        OctStackEntry cur = stack[--sp];
        OctreeNode node = octreeNodeAt(cur.node);

        if (!node.relevant)
        {
            ++nLeaps;         // empty subtree skipped in one step
            continue;
        }

        if (node.isLeaf)
        {
            marchLeafRegion(origin, direction, cur.t0, cur.t1, stepSize,
                            accumR, accumG, accumB, accumA);
            continue;
        }

        // Internal node: gather the relevant children the ray crosses, ordered
        // front-to-back, and push them so the nearest is processed first.
        float3 bmin, bmax;
        unsigned int n = 0;
        for (unsigned int k = 0; k < 8; ++k)
        {
            // childOffset is an index into octreeChild[]; the element is the
            // child's node index into octreeNodes[].
            unsigned int cidx = params.octreeChild[node.childOffset + k];
            OctreeNode child = octreeNodeAt(cidx);
            if (!child.relevant)
                continue;

            octreeNodeWorldBounds(child, bmin, bmax);
            float ct0, ct1;
            if (!intersectAABB(origin, direction, bmin, bmax, ct0, ct1))
                continue;
            ct0 = fmaxf(ct0, cur.t0);
            ct1 = fminf(ct1, cur.t1);
            if (ct0 <= ct1 + 1e-6f)
                childT[n++] = { cidx, ct0, ct1 };
        }

        // Simple insertion sort by ascending entry t (n <= 8).
        for (unsigned int i = 1; i < n; ++i)
        {
            OctStackEntry key = childT[i];
            int j = static_cast<int>(i) - 1;
            while (j >= 0 && childT[j].t0 > key.t0)
            {
                childT[j + 1] = childT[j];
                --j;
            }
            childT[j + 1] = key;
        }

        // Push in reverse so the nearest (smallest t0) is on top.
        for (int i = static_cast<int>(n) - 1; i >= 0; --i)
            stack[sp++] = childT[i];
    }

    if (params.dbgCounters)
        atomicAdd(&params.dbgCounters[2], nLeaps);

    return make_float3(accumR, accumG, accumB);
}

// ============================================================
// Closest-hit program for the SVO / Octree strategy. The single-AABB
// intersection reports the volume entry/exit t-values; this program performs
// the hierarchical octree traversal.
// ============================================================
extern "C" __global__ void __closesthit__ch_octree()
{
    float tmin = __uint_as_float(optixGetAttribute_0());
    float tmax = __uint_as_float(optixGetAttribute_1());

    float3 color = octreeTraverse(optixGetWorldRayOrigin(), optixGetWorldRayDirection(),
                                  tmin, tmax);
    setPayload(color);
}
