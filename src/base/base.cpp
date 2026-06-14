#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "base.h"

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData>   MissSbtRecord;

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: "                      \
               << optixGetErrorName(res) << " (" << __FILE__ << ":"            \
               << __LINE__ << ")\n";                                           \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (false)

#define OPTIX_CHECK_LOG(call)                                                  \
    do {                                                                       \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof(LOG);                                         \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: "                      \
               << optixGetErrorName(res) << " (" << __FILE__ << ":"            \
               << __LINE__ << ")\nLog:\n" << LOG << "\n";                      \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (false)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::stringstream ss;                                              \
            ss << "CUDA call '" << #call << "' failed: "                       \
               << cudaGetErrorString(err) << " (" << __FILE__ << ":"           \
               << __LINE__ << ")\n";                                           \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (false)

int main()
{
    const unsigned int width  = 64;
    const unsigned int height = 32;

    try
    {
        OptixDeviceContext context = nullptr;
        {
            CUDA_CHECK(cudaFree(0));
            CUcontext cuCtx = 0;
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = nullptr;
            options.logCallbackLevel          = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }

        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};

            pipeline_compile_options.usesMotionBlur                   = false;
            pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues                 = 0;
            pipeline_compile_options.numAttributeValues               = 0;
            pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

            std::string optixIr;
            {
                const char* path = std::getenv("OPTIXIR_PATH");
                if (!path)
                    path = OPTIXIR_PATH;
                std::ifstream file(path, std::ios::binary);
                if (!file)
                {
                    std::string err = "Cannot open OptiX IR file: ";
                    err += path;
                    throw std::runtime_error(err);
                }
                optixIr.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            }

            OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options,
                                              optixIr.data(), optixIr.size(),
                                              LOG, &LOG_SIZE, &module));
        }

        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group   = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_desc = {};
            raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_desc.raygen.module            = module;
            raygen_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &raygen_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &raygen_prog_group));

            OptixProgramGroupDesc miss_desc = {};
            miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_desc.miss.module            = module;
            miss_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &miss_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &miss_prog_group));
        }

        OptixPipeline pipeline = nullptr;
        {
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group };

            OptixPipelineLinkOptions link_options = {};
            link_options.maxTraceDepth = 1;
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context, &pipeline_compile_options, &link_options,
                program_groups, 2,
                LOG, &LOG_SIZE, &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& pg : program_groups)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline));

            uint32_t css;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &css, &css, &css));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, css, 1));
        }

        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr raygen_record;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayGenSbtRecord)));
            RayGenSbtRecord rg_sbt = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt,
                                  sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

            CUdeviceptr miss_record;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
            MissSbtRecord ms_sbt = {};
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt,
                                  sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

            sbt.raygenRecord            = raygen_record;
            sbt.missRecordBase          = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount         = 1;
        }

        uchar4* d_pixels = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pixels), width * height * sizeof(uchar4)));

        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            Params params;
            params.image        = d_pixels;
            params.image_width  = width;
            params.image_height = height;

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(Params),
                                  cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        {
            std::vector<uchar4> pixels(width * height);
            CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, width * height * sizeof(uchar4),
                                  cudaMemcpyDeviceToHost));

            for (unsigned int y = 0; y < height; ++y)
            {
                for (unsigned int x = 0; x < width; ++x)
                {
                    uchar4 p = pixels[y * width + x];
                    char ch = " .:-=+*#@"[(p.x + p.y + p.z) / 3 * 9 / 255];
                    std::cout << ch << ch;
                }
                std::cout << "\n";
            }
        }

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_pixels)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
        OPTIX_CHECK(optixModuleDestroy(module));
        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
