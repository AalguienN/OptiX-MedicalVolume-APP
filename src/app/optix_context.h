#pragma once

#include <cuda_runtime.h>
#include <optix.h>

class OptixContext
{
public:
    OptixContext()      = default;
    ~OptixContext();

    OptixContext(const OptixContext&)            = delete;
    OptixContext& operator=(const OptixContext&) = delete;

    void init();

    OptixDeviceContext context() const { return context_; }
    CUstream           stream() const { return stream_; }

private:
    OptixDeviceContext context_ = nullptr;
    CUstream           stream_  = nullptr;
};
