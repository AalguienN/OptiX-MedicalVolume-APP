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
// to avoid device-stack recursion. Within a relevant leaf the shared
// regionMarch() is used (see adaptive_programs.cu): the dense grid is sampled
// on the same step grid as the baseline (fixed step by default, or with the
// adaptive empty-space leap when --adaptive-march is on), so the output is
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
            // Shared intra-region march (adaptive-step when params.useAdaptive
            // is set, fixed dense-baseline step otherwise) over this leaf's
            // span, continuing the accumulated RGBA.
            regionMarch(origin, direction, cur.t0, cur.t1, stepSize,
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
            {
                ++nLeaps;     // empty subtree skipped in one step
                continue;
            }

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

    // total_steps_bounds ([4]): steps over the full volume AABB span for this
    // ray, counted once here (before any empty-subtree skip).
    {
        float stepSize = fminf(params.volumeSpacing.x,
                       fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;
        int bSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
        if (bSteps > 4096) bSteps = 4096;
        if (params.dbgCounters)
            atomicAdd(&params.dbgCounters[4], static_cast<unsigned int>(bSteps));
    }

    float3 color = octreeTraverse(optixGetWorldRayOrigin(), optixGetWorldRayDirection(),
                                  tmin, tmax);
    setPayload(color);
}
