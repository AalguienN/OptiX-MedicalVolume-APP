#include <optix.h>

#include "base.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 color = make_float3(
        (float)idx.x / dim.x,
        (float)idx.y / dim.y,
        0.0f);

    params.image[idx.y * params.image_width + idx.x] = make_color(clamp(color, 0.0f, 1.0f));
}

extern "C" __global__ void __miss__ms()
{
}
