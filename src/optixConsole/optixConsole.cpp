#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "optixConsole.h"
#include "vec_math.h"

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

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

#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaDeviceSynchronize();                                               \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::stringstream ss;                                              \
            ss << "CUDA sync error: " << cudaGetErrorString(err)               \
               << " (" << __FILE__ << ":" << __LINE__ << ")\n";                \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (false)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void configureCamera(float3 cam_eye, float3 cam_lookat, float3 cam_up,
                     float fovY, float aspectRatio,
                     float3& U, float3& V, float3& W)
{
    float3 w = cam_lookat - cam_eye;
    float wlen = length(w);
    W = w;
    U = normalize(cross(w, cam_up));
    V = normalize(cross(U, w));

    float vlen = wlen * tanf(0.5f * fovY * static_cast<float>(M_PIf) / 180.0f);
    V = V * vlen;
    float ulen = vlen * aspectRatio;
    U = U * ulen;
}

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 96x48\n";
    exit(1);
}

void parseDimensions(const std::string& arg, int& width, int& height)
{
    size_t xpos = arg.find('x');
    if (xpos == std::string::npos || xpos == 0 || xpos == arg.size() - 1)
        throw std::invalid_argument("Failed to parse dimensions: " + arg);
    width  = std::stoi(arg.substr(0, xpos));
    height = std::stoi(arg.substr(xpos + 1));
}

void displaySubframe(size_t width, size_t height, uchar4* pixels, std::ostream& output_stream)
{
    float              minLum = std::numeric_limits<float>::infinity();
    float              maxLum = 0;
    std::vector<float> lums(width * height);
    for (unsigned int y = 0; y < height; ++y)
    {
        uchar4* row = pixels + ((height - y - 1) * width);
        for (unsigned int x = 0; x < width; ++x)
        {
            uchar4 ucolor = row[x];
            float3 color = make_float3(
                static_cast<float>(ucolor.x),
                static_cast<float>(ucolor.y),
                static_cast<float>(ucolor.z));
            float lum = color.x * 0.3f + color.y * 0.6f + color.z * 0.1f;
            minLum = std::min(minLum, lum);
            maxLum = std::max(maxLum, lum);
            lums[y * width + x] = lum;
        }
    }

    std::ostringstream out;
    char lumchar[] = { ' ', '.', ',', ';', '!', 'o', '&', '8', '#', '@' };
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            float normalized = (lums[y * width + x] - minLum) / (maxLum - minLum);
            int idx = static_cast<int>(normalized * 9);
            if (idx < 0) idx = 0;
            if (idx > 9) idx = 9;
            out << lumchar[idx];
        }
        out << "\n";
    }
    output_stream << out.str();
}

int main(int argc, char* argv[])
{
    std::string outfile;
    int         width  = 96;
    int         height = 48;

    std::ostream* out_stream = &std::cout;
    std::ofstream file_stream;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            file_stream.open(argv[++i], std::ofstream::out);
            out_stream = &file_stream;
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            parseDimensions(dims_arg, width, height);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        OptixDeviceContext context = nullptr;
        {
            CUDA_CHECK(cudaFree(0));
            CUcontext cuCtx = 0;
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }

        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            float3 sphereVertex = make_float3(0.f, 0.f, 0.f);
            float  sphereRadius = 1.5f;

            CUdeviceptr d_vertex_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buffer), sizeof(float3)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buffer), &sphereVertex,
                                  sizeof(float3), cudaMemcpyHostToDevice));

            CUdeviceptr d_radius_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buffer), sizeof(float)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buffer), &sphereRadius, sizeof(float),
                                  cudaMemcpyHostToDevice));

            OptixBuildInput sphere_input = {};
            sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            sphere_input.sphereArray.vertexBuffers = &d_vertex_buffer;
            sphere_input.sphereArray.numVertices   = 1;
            sphere_input.sphereArray.radiusBuffers = &d_radius_buffer;

            uint32_t sphere_input_flags[1]         = {OPTIX_GEOMETRY_FLAG_NONE};
            sphere_input.sphereArray.flags         = sphere_input_flags;
            sphere_input.sphereArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t compactedSizeOffset = (gas_buffer_sizes.outputSizeInBytes + 7) & ~7ULL;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                                  compactedSizeOffset + 8));

            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

            OPTIX_CHECK(optixAccelBuild(context,
                                        0,
                                        &accel_options, &sphere_input,
                                        1,
                                        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                        d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes,
                                        &gas_handle,
                                        &emitProperty,
                                        1));

            d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

            CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
            CUDA_CHECK(cudaFree((void*)d_vertex_buffer));
            CUDA_CHECK(cudaFree((void*)d_radius_buffer));

            size_t compacted_gas_size;
            CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
            {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));
                OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));
                CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
            }
            else
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        OptixModule module = nullptr;
        OptixModule sphere_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};

            pipeline_compile_options.usesMotionBlur                   = false;
            pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues                 = 3;
            pipeline_compile_options.numAttributeValues               = 1;
            pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

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

            OptixBuiltinISOptions builtin_is_options = {};
            builtin_is_options.usesMotionBlur      = false;
            builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
            OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options,
                                                    &builtin_is_options, &sphere_module));
        }

        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &raygen_prog_group_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &raygen_prog_group));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &miss_prog_group_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &miss_prog_group));

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
            hitgroup_prog_group_desc.hitgroup.moduleIS            = sphere_module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &hitgroup_prog_group_desc, 1, &program_group_options,
                LOG, &LOG_SIZE, &hitgroup_prog_group));
        }

        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context, &pipeline_compile_options, &pipeline_link_options,
                program_groups, sizeof(program_groups) / sizeof(program_groups[0]),
                LOG, &LOG_SIZE, &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
            {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                                   0, 0,
                                                   &direct_callable_stack_size_from_traversal,
                                                   &direct_callable_stack_size_from_state,
                                                   &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                                  direct_callable_stack_size_from_state, continuation_stack_size,
                                                  1));
        }

        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));

            float3 cam_eye   = make_float3(0.0f, 0.0f, 3.0f);
            float3 cam_lookat = make_float3(0.0f, 0.0f, 0.0f);
            float3 cam_up    = make_float3(0.0f, 1.0f, 0.0f);
            float  fovY      = 60.0f;
            float  aspect    = static_cast<float>(width) / (2.0f * static_cast<float>(height));

            RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            rg_sbt.data.cam_eye = cam_eye;
            configureCamera(cam_eye, cam_lookat, cam_up, fovY, aspect,
                            rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w);
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt,
                                  raygen_record_size, cudaMemcpyHostToDevice));

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt,
                                  miss_record_size, cudaMemcpyHostToDevice));

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt,
                                  hitgroup_record_size, cudaMemcpyHostToDevice));

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount         = 1;
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
            params.origin_x     = width / 2;
            params.origin_y     = height / 2;
            params.handle       = gas_handle;

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params),
                                  cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1));
            CUDA_SYNC_CHECK();

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
        }

        {
            std::vector<uchar4> pixels(width * height);
            CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, width * height * sizeof(uchar4),
                                  cudaMemcpyDeviceToHost));
            displaySubframe(width, height, pixels.data(), *out_stream);
        }

        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_pixels)));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));
            OPTIX_CHECK(optixModuleDestroy(sphere_module));
            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
