#pragma once

#include <optix.h>
#include <vector_types.h>

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    int                    origin_x;
    int                    origin_y;
    OptixTraversableHandle handle;
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
