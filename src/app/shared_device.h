#pragma once

#include <optix.h>
#include <vector_types.h>

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
