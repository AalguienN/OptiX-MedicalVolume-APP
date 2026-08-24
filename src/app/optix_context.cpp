#include "check_macros.h"
#include "optix_context.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

OptixContext::~OptixContext()
{
    if (context_)
        OPTIX_CHECK_NOEXCEPT(optixDeviceContextDestroy(context_));
    if (stream_)
        CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream_));
}

void OptixContext::init()
{
    CUcontext cuCtx = 0;

    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));

    CUDA_CHECK(cudaStreamCreate(&stream_));
}
