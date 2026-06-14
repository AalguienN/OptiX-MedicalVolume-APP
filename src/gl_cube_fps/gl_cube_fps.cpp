#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define GL_GLEXT_PROTOTYPES 1
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "gl_cube_fps.h"
#include "vec_math.h"
#include "text_overlay.h"

template <typename T> struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

#define OPTIX_CHECK(call)                                                      \
  do {                                                                         \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "Optix call '" << #call << "' failed: " << optixGetErrorName(res)  \
         << " (" << __FILE__ << ":" << __LINE__ << ")\n";                      \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (false)

#define OPTIX_CHECK_LOG(call)                                                  \
  do {                                                                         \
    char LOG[2048];                                                            \
    size_t LOG_SIZE = sizeof(LOG);                                             \
    OptixResult res = call;                                                    \
    if (res != OPTIX_SUCCESS) {                                                \
      std::stringstream ss;                                                    \
      ss << "Optix call '" << #call << "' failed: " << optixGetErrorName(res)  \
         << " (" << __FILE__ << ":" << __LINE__ << ")\nLog:\n"                 \
         << LOG << "\n";                                                       \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (false)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::stringstream ss;                                                    \
      ss << "CUDA call '" << #call << "' failed: " << cudaGetErrorString(err)  \
         << " (" << __FILE__ << ":" << __LINE__ << ")\n";                      \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (false)

#define GL_CHECK(call)                                                         \
  do {                                                                         \
    call;                                                                      \
    GLenum err = glGetError();                                                 \
    if (err != GL_NO_ERROR) {                                                  \
      std::stringstream ss;                                                    \
      ss << "GL error 0x" << std::hex << err << " after '" << #call << "' ("   \
         << __FILE__ << ":" << __LINE__ << ")\n";                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (false)

static const char *vertex_shader_src = R"(
#version 330 core
out vec2 uv;
void main() {
    const vec2 verts[6] = vec2[](
        vec2(-1,-1), vec2( 1,-1), vec2( 1, 1),
        vec2(-1,-1), vec2( 1, 1), vec2(-1, 1)
    );
    vec2 p = verts[gl_VertexID];
    uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

static const char *fragment_shader_src = R"(
#version 330 core
in vec2 uv;
uniform sampler2D tex;
out vec4 color;
void main() {
    color = texture(tex, uv);
}
)";

static GLuint compileShader(GLenum type, const char *src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);
  GLint ok;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    glDeleteShader(shader);
    std::string typeName = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
    throw std::runtime_error(typeName + " shader compile error:\n" +
                             std::string(log));
  }
  return shader;
}

static GLuint createProgram() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, vertex_shader_src);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragment_shader_src);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  GLint ok;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glDeleteProgram(prog);
    throw std::runtime_error(std::string("program link error:\n") + log);
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return prog;
}

struct CubeMesh {
  std::vector<float3> vertices;
  std::vector<uint3> indices;
};

static CubeMesh makeCube() {
  float3 verts[8] = {
      make_float3(-1, -1, -1), make_float3(1, -1, -1), make_float3(1, 1, -1),
      make_float3(-1, 1, -1),  make_float3(-1, -1, 1), make_float3(1, -1, 1),
      make_float3(1, 1, 1),    make_float3(-1, 1, 1),
  };

  uint32_t idx[36] = {
      0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4,
      1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6, 7, 0, 4, 5, 0, 5, 1,
  };

  CubeMesh mesh;
  mesh.vertices.assign(verts, verts + 8);
  for (int i = 0; i < 12; i++)
    mesh.indices.push_back(
        make_uint3(idx[i * 3], idx[i * 3 + 1], idx[i * 3 + 2]));
  return mesh;
}

struct GlInteropBuffer {
  GLuint pbo = 0;
  GLuint texture = 0;
  cudaGraphicsResource *cuda_resource = nullptr;
  unsigned int width = 0;
  unsigned int height = 0;

  void init(unsigned int w, unsigned int h) {
    width = w;
    height = h;

    GL_CHECK(glGenBuffers(1, &pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(uchar4),
                          nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    GL_CHECK(glGenTextures(1, &texture));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                          GL_UNSIGNED_BYTE, nullptr));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  }

  uchar4 *map(CUstream stream = 0) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_resource, stream));
    uchar4 *ptr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&ptr), &size, cuda_resource));
    return ptr;
  }

  void unmap(CUstream stream = 0) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_resource, stream));
  }

  void display() {
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));
    GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                             GL_UNSIGNED_BYTE, nullptr));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  }

  void destroy() {
    if (cuda_resource)
      cudaGraphicsUnregisterResource(cuda_resource);
    if (pbo)
      glDeleteBuffers(1, &pbo);
    if (texture)
      glDeleteTextures(1, &texture);
    cuda_resource = nullptr;
    pbo = 0;
    texture = 0;
  }
};

struct OrbitCamera {
  float theta = 0.0f;
  float phi = 0.3f;
  float radius = 4.0f;
  float fovY = 45.0f;
  float aspect = 1.0f;
  double last_x = 0.0;
  double last_y = 0.0;
  bool dragging = false;

  float3 eye() const {
    return make_float3(radius * cosf(phi) * sinf(theta), radius * sinf(phi),
                       radius * cosf(phi) * cosf(theta));
  }

  void updateUVW(float3 &U, float3 &V, float3 &W) const {
    float3 cam_eye = eye();
    float3 cam_lookat = make_float3(0, 0, 0);
    float3 cam_up = make_float3(0, 1, 0);

    float3 w = cam_lookat - cam_eye;
    float wlen = length(w);
    W = w;
    U = normalize(cross(w, cam_up));
    V = normalize(cross(U, w));

    float vlen = wlen * tanf(0.5f * fovY * static_cast<float>(M_PIf) / 180.0f);
    V = V * vlen;
    float ulen = vlen * aspect;
    U = U * ulen;
  }
};

static void errorCallback(int error, const char *desc) {
  std::cerr << "GLFW error " << error << ": " << desc << "\n";
}

int main() {
  unsigned int width = 800;
  unsigned int height = 600;

  glfwSetErrorCallback(errorCallback);
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  GLFWwindow *window =
      glfwCreateWindow(width, height, "OptiX Cube", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window\n";
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));

  GLuint fullscreen_vao;
  glGenVertexArrays(1, &fullscreen_vao);
  glBindVertexArray(fullscreen_vao);

  GLuint shader_program = createProgram();
  GLint tex_uniform = glGetUniformLocation(shader_program, "tex");

  try {
    OptixDeviceContext context = nullptr;
    CUcontext cuCtx = 0;
    CUstream stream = nullptr;
    {
      CUDA_CHECK(cudaFree(0));
      OPTIX_CHECK(optixInit());
      OptixDeviceContextOptions options = {};
      OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
      CUDA_CHECK(cudaStreamCreate(&stream));
    }

    CubeMesh cube = makeCube();

    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices),
                          cube.vertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_indices),
                          cube.indices.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices), cube.vertices.data(),
        cube.vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_indices), cube.indices.data(),
        cube.indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    {
      OptixAccelBuildOptions accel_options = {};
      accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

      uint32_t input_flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
      OptixBuildInput triangle_input = {};
      triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
      triangle_input.triangleArray.numVertices =
          static_cast<uint32_t>(cube.vertices.size());
      triangle_input.triangleArray.vertexBuffers = &d_vertices;
      triangle_input.triangleArray.flags = input_flags;
      triangle_input.triangleArray.numSbtRecords = 1;
      triangle_input.triangleArray.indexFormat =
          OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
      triangle_input.triangleArray.indexBuffer = d_indices;
      triangle_input.triangleArray.numIndexTriplets =
          static_cast<uint32_t>(cube.indices.size());

      OptixAccelBufferSizes gas_buffer_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(
          context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

      CUdeviceptr d_temp_buffer;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer),
                            gas_buffer_sizes.tempSizeInBytes));

      CUdeviceptr d_buffer_output;
      size_t compactedSizeOffset =
          (gas_buffer_sizes.outputSizeInBytes + 7) & ~7ULL;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buffer_output),
                            compactedSizeOffset + 8));

      OptixAccelEmitDesc emit_prop = {};
      emit_prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emit_prop.result =
          (CUdeviceptr)((char *)d_buffer_output + compactedSizeOffset);

      OPTIX_CHECK(optixAccelBuild(
          context, stream, &accel_options, &triangle_input, 1, d_temp_buffer,
          gas_buffer_sizes.tempSizeInBytes, d_buffer_output,
          gas_buffer_sizes.outputSizeInBytes, &gas_handle, &emit_prop, 1));

      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));

      size_t compacted_size;
      CUDA_CHECK(cudaMemcpy(&compacted_size,
                            reinterpret_cast<void *>(emit_prop.result),
                            sizeof(size_t), cudaMemcpyDeviceToHost));
      if (compacted_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                              compacted_size));
        OPTIX_CHECK(optixAccelCompact(context, stream, gas_handle,
                                      d_gas_output_buffer, compacted_size,
                                      &gas_handle));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_buffer_output)));
      } else {
        d_gas_output_buffer = d_buffer_output;
      }
    }

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
      OptixModuleCompileOptions module_compile_options = {};

      pipeline_compile_options.usesMotionBlur = false;
      pipeline_compile_options.traversableGraphFlags =
          OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      pipeline_compile_options.numPayloadValues = 3;
      pipeline_compile_options.numAttributeValues = 2;
      pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
      pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
      pipeline_compile_options.usesPrimitiveTypeFlags =
          OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

      std::string optixIr;
      {
        const char *path = std::getenv("gl_cube_fps_OPTIXIR_PATH");
        if (!path)
          path = gl_cube_fps_OPTIXIR_PATH;
        std::ifstream file(path, std::ios::binary);
        if (!file) {
          std::string err = "Cannot open OptiX IR file: ";
          err += path;
          throw std::runtime_error(err);
        }
        optixIr.assign(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
      }

      OPTIX_CHECK_LOG(optixModuleCreate(
          context, &module_compile_options, &pipeline_compile_options,
          optixIr.data(), optixIr.size(), LOG, &LOG_SIZE, &module));
    }

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    {
      OptixProgramGroupOptions program_group_options = {};

      OptixProgramGroupDesc raygen_desc = {};
      raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_desc.raygen.module = module;
      raygen_desc.raygen.entryFunctionName = "__raygen__rg";
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_desc, 1,
                                              &program_group_options, LOG,
                                              &LOG_SIZE, &raygen_prog_group));

      OptixProgramGroupDesc miss_desc = {};
      miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_desc.miss.module = module;
      miss_desc.miss.entryFunctionName = "__miss__ms";
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_desc, 1,
                                              &program_group_options, LOG,
                                              &LOG_SIZE, &miss_prog_group));

      OptixProgramGroupDesc hitgroup_desc = {};
      hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitgroup_desc.hitgroup.moduleCH = module;
      hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
      OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_desc, 1,
                                              &program_group_options, LOG,
                                              &LOG_SIZE, &hitgroup_prog_group));
    }

    OptixPipeline pipeline = nullptr;
    {
      OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group,
                                            hitgroup_prog_group};

      OptixPipelineLinkOptions link_options = {};
      link_options.maxTraceDepth = 1;
      OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options,
                                          &link_options, program_groups, 3, LOG,
                                          &LOG_SIZE, &pipeline));

      OptixStackSizes stack_sizes = {};
      for (auto &pg : program_groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline));

      uint32_t css;
      OPTIX_CHECK(
          optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &css, &css, &css));
      OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, css, 1));
    }

    OptixShaderBindingTable sbt = {};
    {
      CUdeviceptr raygen_record;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record),
                            sizeof(RayGenSbtRecord)));
      RayGenSbtRecord rg_sbt = {};
      OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                            sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

      CUdeviceptr miss_record;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record),
                            sizeof(MissSbtRecord)));
      MissSbtRecord ms_sbt = {};
      ms_sbt.data = {0.3f, 0.1f, 0.2f};
      OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                            sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

      CUdeviceptr hitgroup_record;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record),
                            sizeof(HitGroupSbtRecord)));
      HitGroupSbtRecord hg_sbt = {};
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
                            sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

      sbt.raygenRecord = raygen_record;
      sbt.missRecordBase = miss_record;
      sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
      sbt.missRecordCount = 1;
      sbt.hitgroupRecordBase = hitgroup_record;
      sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
      sbt.hitgroupRecordCount = 1;
    }

    GlInteropBuffer gl_buffer;
    gl_buffer.init(width, height);

    TextOverlay overlay;
    overlay.init(width, height);

    Params params;
    params.image_width = width;
    params.image_height = height;
    params.handle = gas_handle;

    CUdeviceptr d_params;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));

    OrbitCamera camera;
    camera.aspect = static_cast<float>(width) / static_cast<float>(height);

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      {
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if (static_cast<unsigned int>(fb_w) != width ||
            static_cast<unsigned int>(fb_h) != height) {
          width = static_cast<unsigned int>(fb_w);
          height = static_cast<unsigned int>(fb_h);
          gl_buffer.destroy();
          gl_buffer.init(width, height);
          overlay.resize(static_cast<int>(width), static_cast<int>(height));
          params.image_width = width;
          params.image_height = height;
          camera.aspect = static_cast<float>(width) / static_cast<float>(height);
          glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        }
      }

      double now = glfwGetTime();
      double dt = now - last_time;
      last_time = now;

      double fps = 1.0 / dt;
      static double fps_smoothed = 0.0;
      double alpha = 0.05;
      fps_smoothed = alpha * fps + (1.0 - alpha) * fps_smoothed;

      camera.theta += static_cast<float>(dt * 0.5);

      float3 cam_eye = camera.eye();
      float3 cam_U, cam_V, cam_W;
      camera.updateUVW(cam_U, cam_V, cam_W);

      uchar4 *d_pixels = gl_buffer.map(stream);

      params.image = d_pixels;

      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params,
                                 sizeof(Params), cudaMemcpyHostToDevice,
                                 stream));

      {
        RayGenSbtRecord rg_sbt;
        rg_sbt.data.cam_eye = cam_eye;
        rg_sbt.data.camera_u = cam_U;
        rg_sbt.data.camera_v = cam_V;
        rg_sbt.data.camera_w = cam_W;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(sbt.raygenRecord),
                                   &rg_sbt, sizeof(RayGenSbtRecord),
                                   cudaMemcpyHostToDevice, stream));
      }

      OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt,
                              width, height, 1));

      gl_buffer.unmap(stream);

      gl_buffer.display();

      glUseProgram(shader_program);
      glUniform1i(tex_uniform, 0);
      glBindTexture(GL_TEXTURE_2D, gl_buffer.texture);
      glBindVertexArray(fullscreen_vao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      char fps_buf[64];
      std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps_smoothed);
      overlay.draw(10.0f, 10.0f, fps_buf, 1.0f, 0.7f, 0.7f, 0.7f);

      glfwSwapBuffers(window);
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_indices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
    gl_buffer.destroy();
    overlay.destroy();
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CUDA_CHECK(cudaStreamDestroy(stream));
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
    glfwTerminate();
    return 1;
  }

    glfwTerminate();
    return 0;
}
