#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <vector_types.h>

// Octree node for the SVO / Octree strategies (Variants A and AB). The octree
// is used purely as a spatial acceleration structure over the dense grid:
// each node records whether its subtree contains any rendering-relevant voxel,
// and leaf nodes reference a half-open voxel region [voxelMin, voxelMax) of the
// unmodified dense grid rather than duplicating it.
struct OctreeNode
{
    unsigned int relevant;      // 1 if the subtree has a rendering-relevant voxel
    unsigned int isLeaf;        // 1 = leaf (dense-grid region), 0 = internal
    unsigned int childOffset;   // octreeChild[] offset of this node's 8 children
    unsigned int voxelMin[3];   // half-open voxel bounds of this node's region
    unsigned int voxelMax[3];
};

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    OptixTraversableHandle handle;

    cudaTextureObject_t    volumeTex;
    float4*                tfData;
    int3                   volumeDims;
    float3                 volumeSpacing;
    float3                 volumeOrigin;
    float3                 volumeMax;
    float                  scalarMin;
    float                  scalarMax;

    // Adaptive-step fields. distanceTex is a 3D texture holding the per-voxel
    // Chebyshev distance (in voxel units) to the nearest rendering-relevant
    // (non-empty) voxel; epsilon is the opacity threshold that defines a voxel
    // as rendering-relevant, matching the threshold used to build the distance
    // map / occupancy. Host sets these for TraceMode::ADAPTIVE and, when
    // --adaptive-march is on, for the octree / bricked-regions / octree-regions
    // modes. useAdaptive gates the shared intra-region march: 1 selects the
    // adaptive-step variant, 0 the fixed dense-baseline step.
    cudaTextureObject_t    distanceTex;
    float                  epsilon;
    unsigned int           useAdaptive;

    // Bricked-regions strategy fields (host sets them only for
    // TraceMode::BRICKED_REGIONS). regionAabbs is the per-primitive AABB array
    // of the multi-primitive GAS; each primitive is one rendering-relevant
    // brick, so the intersection program reads its own bounds and the
    // closest-hit marches that brick span.
    OptixAabb*             brickedRegionsAabbs;

    // SVO / Octree strategy fields (host sets them only for TraceMode::OCTREE
    // / OCTREE_REGIONS). Nodes are stored in a preorder array with the root at
    // index 0; octreeChild holds the 8 child indices of each internal node.
    OctreeNode*            octreeNodes;
    unsigned int*          octreeChild;
    unsigned int           octreeNodeCount;

    // Octree-regions (Variant B): per-primitive world AABBs of the non-empty
    // octree leaves, uploaded flat, plus the node index each primitive belongs
    // to so the closest-hit can march that leaf's voxel region.
    OptixAabb*             ocRegionAabbs;
    unsigned int*          ocRegionNode;

    // NanoVDB strategy fields (host sets them only for TraceMode::NANOVDB).
    // nanovdbGrid is a device pointer to an uploaded nanovdb::FloatGrid (cast on
    // the device side to const nanovdb::FloatGrid*). The closest-hit samples it
    // with NanoVDB's device Accessor and skips empty leaf tiles via getDim().
    // nanovdbNearest selects the per-sample interpolation order: 1 = nearest
    // voxel (1 accessor probe, ~4x faster), 0 = trilinear (dense-equivalent).
    const void*            nanovdbGrid;
    unsigned int           nanovdbNearest;

    // Optional per-frame diagnostic counters (device, 4 x unsigned int):
    // [0] = volume samples (actual voxels visited)
    // [1] = distance-map reads
    // [2] = leaps (empty-region / node / subtree skips)
    // [3] = total_steps_bounds   (steps over the full volume AABB span, no
    //                             early termination; counted once per ray)
    // Null to disable. Only incremented by the device marchers for traversal
    // cost analysis. The single derived skip ratio is:
    //   skip_ratio = 1 - [0]/[3]
    unsigned int*          dbgCounters;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
