#include "check_macros.h"
#include "placeholder_scene.h"

#include <optix.h>
#include <optix_stubs.h>

#include <cstdint>

namespace {

struct CubeMesh
{
    std::vector<float3> vertices;
    std::vector<uint3>  indices;
};

CubeMesh makeCube()
{
    float3 verts[8] = {
        make_float3(-1, -1, -1), make_float3( 1, -1, -1),
        make_float3( 1,  1, -1), make_float3(-1,  1, -1),
        make_float3(-1, -1,  1), make_float3( 1, -1,  1),
        make_float3( 1,  1,  1), make_float3(-1,  1,  1),
    };

    uint32_t idx[36] = {
        0, 1, 2,  0, 2, 3,
        4, 6, 5,  4, 7, 6,
        0, 3, 7,  0, 7, 4,
        1, 5, 6,  1, 6, 2,
        3, 2, 6,  3, 6, 7,
        0, 4, 5,  0, 5, 1,
    };

    CubeMesh mesh;
    mesh.vertices.assign(verts, verts + 8);
    for (int i = 0; i < 12; ++i)
        mesh.indices.push_back(make_uint3(idx[i * 3], idx[i * 3 + 1], idx[i * 3 + 2]));
    return mesh;
}

}  // namespace

PlaceholderScene::~PlaceholderScene()
{
    if (d_vertices_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_vertices_)));
    if (d_indices_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_indices_)));
    if (d_gasOutput_)
        CUDA_CHECK_NOEXCEPT(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
}

void PlaceholderScene::init(OptixDeviceContext context, CUstream stream)
{
    context_ = context;
    stream_  = stream;

    CubeMesh cube = makeCube();
    vertices_ = cube.vertices;
    indices_  = cube.indices;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_),
                          vertices_.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_),
                          indices_.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices_), vertices_.data(),
                          vertices_.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices_), indices_.data(),
                          indices_.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    uint32_t input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type                             = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat       = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices        = static_cast<uint32_t>(vertices_.size());
    triangle_input.triangleArray.vertexBuffers      = &d_vertices_;
    triangle_input.triangleArray.flags              = input_flags;
    triangle_input.triangleArray.numSbtRecords      = 1;
    triangle_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    triangle_input.triangleArray.indexBuffer        = d_indices_;
    triangle_input.triangleArray.numIndexTriplets   = static_cast<uint32_t>(indices_.size());

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accel_options, &triangle_input,
                                             1, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_output = 0;
    size_t compactedSizeOffset = (gas_buffer_sizes.outputSizeInBytes + 7) & ~static_cast<size_t>(7);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_output), compactedSizeOffset + 8));

    OptixAccelEmitDesc emit_prop = {};
    emit_prop.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_prop.result = reinterpret_cast<CUdeviceptr>(
        reinterpret_cast<char*>(d_buffer_output) + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(context_, stream_, &accel_options, &triangle_input, 1,
                                d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_output, gas_buffer_sizes.outputSizeInBytes,
                                &handle_, &emit_prop, 1));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_size = 0;
    CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void*>(emit_prop.result),
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    if (compacted_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput_), compacted_size));
        OPTIX_CHECK(optixAccelCompact(context_, stream_, handle_,
                                      d_gasOutput_, compacted_size, &handle_));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_buffer_output)));
    }
    else
    {
        d_gasOutput_ = d_buffer_output;
    }
}
