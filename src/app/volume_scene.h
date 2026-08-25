#pragma once

#include <optix.h>
#include <cuda_runtime.h>

class VolumeScene
{
public:
    VolumeScene() = default;
    ~VolumeScene();

    VolumeScene(const VolumeScene&) = delete;
    VolumeScene& operator=(const VolumeScene&) = delete;

    void init(OptixDeviceContext context, CUstream stream,
              float3 bmin, float3 bmax);

    OptixTraversableHandle handle() const { return handle_; }

private:
    OptixDeviceContext context_ = nullptr;
    CUstream           stream_  = nullptr;
    CUdeviceptr        d_aabb_  = 0;
    CUdeviceptr        d_gasOutput_ = 0;
    OptixTraversableHandle handle_ = 0;
};
