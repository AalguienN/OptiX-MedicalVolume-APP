#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <vector_types.h>

// Per-brick metadata for the bricked / tiled strategy
// (Section "Bricked / tiled volume" of the thesis). Each fixed-size brick
// of n^3 voxels stores the min/max scalar value, the min/max opacity after
// transfer-function application, and a flag telling whether the brick
// contains any rendering-relevant voxel.
struct BrickMeta
{
    float              minScalar;
    float              maxScalar;
    float              minOpacity;
    float              maxOpacity;
    unsigned int       relevant;   // 1 if maxOpacity >= epsilon
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

    // Bricked strategy fields (host sets them only for TraceMode::BRICKED).
    BrickMeta*             brickMeta;      // device array, brickCount.x*y*z
    int3                   brickDims;      // voxels per brick per axis
    int3                   brickCount;     // bricks per axis (ceil)
    float3                 brickSize;      // world-space extent per brick

    // Adaptive-step strategy fields (host sets them only for
    // TraceMode::ADAPTIVE). distanceTex is a 3D texture holding the per-voxel
    // Chebyshev distance (in voxel units) to the nearest rendering-relevant
    // (non-empty) voxel. minSpacing is the smallest voxel spacing, used to
    // convert the distance-map voxel value into a safe world-space advance.
    cudaTextureObject_t    distanceTex;
    float                  minSpacing;
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
