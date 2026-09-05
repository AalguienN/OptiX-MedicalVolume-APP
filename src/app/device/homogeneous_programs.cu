///////////////////////////////////////////////////////////////////////////////
// Homogeneous-region (similarity) strategy device programs.
//
// Faithful to the adaptive-step strategy: the dense grid is left unmodified
// and a same-resolution per-voxel similarity distance map is precomputed on
// the host (SimilarityGridMarcher). At runtime each step reads the map and
// advances. There is NO octree/hierarchy at render time.
//
// The similarity map stores, per voxel, the Chebyshev distance D to the
// nearest voxel whose transfer-function opacity differs by >= deltaAlpha OR
// any colour channel differs by >= deltaColor (i.e. the nearest *dissimilar*
// voxel, anchored on each voxel's own value). Therefore every voxel strictly
// within Chebyshev radius D of a sample is similar to it (within delta of its
// opacity and colour), so the sample's own (colour, opacity) stands in for the
// whole surrounding window.
//
// Two leap paths coexist in the shared span march:
//   * Empty region (opacity < epsilon): reuses the Chebyshev empty map leap
//     exactly as the adaptive strategy (params::distanceTex / useAdaptive).
//   * Relevant + similar region (opacity >= epsilon and D >= 2): leap (D-1)
//     voxel steps and add the *closed-form* discrete front-to-back composite
//     of that span using the sample's own colour/opacity over N constant
//     samples. For a truly constant region this reproduces the dense baseline
//     exactly (up to float rounding); with a < delta spread it is within a
//     bounded delta-induced error, which the thesis's fidelity gate measures.
//
// A standalone raygen (__raygen__rg_homogeneous) performs a software AABB
// test and the same march, matching the adaptive standalone program. The
// homogeneity leap is also exposed as an optional inner march for the other
// region strategies via params.useSimilarity (like --adaptive-march).
///////////////////////////////////////////////////////////////////////////////
#include "shared_device_programs.h"

// Closed-form front-to-back composite for a span of N identical discrete
// samples of (r,g,b) with per-sample opacity a, continuing from the running
// accumulation. Derived from the discrete recurrence; reproduces marching the
// N steps one-by-one.
static __forceinline__ __device__ void compositeConstSpan(
    float r, float g, float b, float a, int N,
    float& accumR, float& accumG, float& accumB, float& accumA)
{
    if (N <= 0) return;
    // Transmission after N identical samples: (1 - a)^N, computed in a
    // log-space-safe way. For the small per-step opacities used here the
    // direct product loop is exact and avoids powf precision issues for N<=20.
    float T = 1.0f;
    for (int i = 0; i < N && i < 1024; ++i)
        T *= (1.0f - a);
    // Guard: if a region is large enough to fully occlude, clamp.
    const float alphaAdd = (1.0f - accumA) * (1.0f - T);
    accumR += alphaAdd * r;
    accumG += alphaAdd * g;
    accumB += alphaAdd * b;
    accumA += alphaAdd;
}

// Homogeneity leap advance. sampleTexCoord is the current sample's texCoord;
// posVoxelSpace is the sample expressed in voxel units (for the baseline fixed
// step grid alignment). Uses the sample's transfer-function value (tfVal) to
// decide empty vs similar. Updates accum via the closed form on a leap.
static __forceinline__ __device__ float homogeneousAdvance(
    float t, float3 texCoord, float4 tfVal,
    float stepSize, float voxelWorldStep, float startVoxelOffset,
    float& accumR, float& accumG, float& accumB, float& accumA,
    unsigned int& nDistReads, unsigned int& nLeaps)
{
    // Relevant sample (opacity >= epsilon): try the homogeneity leap. If the
    // similarity distance map says a whole window around this sample is within
    // delta of its opacity/colour, leap the window in closed form.
    if (tfVal.w >= params.epsilon)
    {
        float D = 255.0f * tex3D<float>(params.similarityTex,
                                        texCoord.x, texCoord.y, texCoord.z);
        ++nDistReads;
        int iD = __float2int_rd(D);
        if (iD >= 2)
        {
            ++nLeaps;
            // Length of the similar window in voxels that we leap (D-1),
            // converted to world time.
            const float leapT = (iD - 1) * voxelWorldStep;
            // Number of baseline fixed samples that fit in the leap, measured
            // from the current sample's step-grid position so the remainder is
            // later picked up by a normal fixed step.
            float r = startVoxelOffset + (t / voxelWorldStep);
            int N = static_cast<int>((iD - 1) - r);
            N = max(N, 0);
            const float a = tfVal.w * stepSize;
            if (N >= 1 && a > 0.0f)
                compositeConstSpan(tfVal.x, tfVal.y, tfVal.z, a, N,
                                   accumR, accumG, accumB, accumA);
            return t + leapT;
        }
        return t + stepSize;
    }

    // Empty (opacity below relevance): consult the Chebyshev empty map.
    float De = 255.0f * tex3D<float>(params.distanceTex, texCoord.x, texCoord.y, texCoord.z);
    ++nDistReads;
    int iDe = __float2int_rd(De);
    if (iDe >= 2)
    {
        ++nLeaps;
        return t + (iDe - 1) * voxelWorldStep;
    }
    return t + stepSize;
}

// Shared homogeneous span march: same span semantics as spanMarchFixed /
// spanMarchAdaptive, but with an additional homogeneity leap. When the current
// sample is relevant (opacity >= epsilon) and a similar window surrounds it,
// leap it in closed form; otherwise fall back to the empty-map leap / fixed
// step. Dispatched from regionMarch() when params.useSimilarity is set.
static __forceinline__ __device__ void spanMarchHomogeneous(
    float3 origin, float3 direction, float tmin, float tmax, float stepSize,
    float& accumR, float& accumG, float& accumB, float& accumA)
{
    float voxelWorldStep = voxelWorldStepFor(direction);

    unsigned int nSamples = 0;
    unsigned int nDistReads = 0;
    unsigned int nLeaps = 0;

    int maxSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
    if (maxSteps > 4096) maxSteps = 4096;

    float t = tmin;
    for (int i = 0; i < maxSteps && t <= tmax && accumA < 0.99f; ++i)
    {
        float3 samplePos = origin + direction * t;
        float3 texCoord = sampleTexCoord(samplePos);
        float4 tfVal = compositeSample(texCoord, stepSize,
                                       accumR, accumG, accumB, accumA);
        ++nSamples;

        // startVoxelOffset: where the current sample falls on the fixed step
        // grid in voxel units (voxelWorldStep maps world time to voxels).
        float startVoxelOffset = (t / voxelWorldStep);
        t = homogeneousAdvance(t, texCoord, tfVal, stepSize, voxelWorldStep,
                               startVoxelOffset,
                               accumR, accumG, accumB, accumA,
                               nDistReads, nLeaps);
    }

    if (params.dbgCounters)
    {
        atomicAdd(&params.dbgCounters[0], nSamples);
        atomicAdd(&params.dbgCounters[1], nDistReads);
        atomicAdd(&params.dbgCounters[2], nLeaps);
    }
}

// Extend the shared gated region march to include the homogeneity variant.
// regionMarch() lives in adaptive_programs.cu (same translation unit); we
// augment the dispatch here by checking params.useSimilarity ahead of it.
// To keep a single definition, the region strategies call the same
// regionMarch(); we instead route via the params.useSimilarity flag handled
// inside regionMarch's callers. For the standalone homogeneous raygen we call
// spanMarchHomogeneous directly.

static __forceinline__ __device__ float3 homogeneousMarch(
    float3 origin, float3 direction, float tmin, float tmax)
{
    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    spanMarchHomogeneous(origin, direction, tmin, tmax, stepSize,
                         accumR, accumG, accumB, accumA);
    return make_float3(accumR, accumG, accumB);
}

// ============================================================
// Manual raygen program for the homogeneous strategy (no optixTrace).
// Performs the same software AABB entry/exit test as the manual dense
// baseline and the adaptive standalone, then runs homogeneousMarch().
// ============================================================
extern "C" __global__ void __raygen__rg_homogeneous()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;

    float3 origin    = rtData->cam_eye;
    float3 direction = normalize(d.x * rtData->camera_u + d.y * rtData->camera_v + rtData->camera_w);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    float tmin, tmax;
    if (intersectAABB(origin, direction, params.volumeOrigin, params.volumeMax, tmin, tmax))
    {
        if (tmin < 0.0f) tmin = 0.0f;

        {
            float stepSize = fminf(params.volumeSpacing.x,
                           fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;
            int bSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
            if (bSteps > 4096) bSteps = 4096;
            if (params.dbgCounters)
                atomicAdd(&params.dbgCounters[3], static_cast<unsigned int>(bSteps));
        }

        color = homogeneousMarch(origin, direction, tmin, tmax);
    }

    params.image[idx.y * params.image_width + idx.x] = make_color(color);
}
