#pragma once

#include <optix.h>
#include <vector_functions.h>

#include <vector>

class PlaceholderScene
{
public:
    PlaceholderScene()      = default;
    ~PlaceholderScene();

    PlaceholderScene(const PlaceholderScene&)            = delete;
    PlaceholderScene& operator=(const PlaceholderScene&) = delete;

    void init(OptixDeviceContext context, CUstream stream);

    OptixTraversableHandle handle() const { return handle_; }

private:
    OptixDeviceContext    context_     = nullptr;
    CUstream              stream_      = nullptr;
    std::vector<float3>   vertices_;
    std::vector<uint3>    indices_;
    CUdeviceptr           d_vertices_  = 0;
    CUdeviceptr           d_indices_   = 0;
    CUdeviceptr           d_gasOutput_ = 0;
    OptixTraversableHandle handle_     = 0;
};
