#pragma once

#include <optix.h>
#include <vector_types.h>

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
};

struct RayGenData
{
};

struct MissData
{
    float r, g, b;
};
