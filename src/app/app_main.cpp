#include "camera.h"
#include "check_macros.h"
#include "display.h"
#include "gl_check.h"
#include "gl_window.h"
#include "interop_buffer.h"
#include "optix_context.h"
#include "pipeline_base.h"
#include "placeholder_scene.h"
#include "shared_device.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  unsigned int width = 800;
  unsigned int height = 600;

  try {
    GlWindow window;
    if (!window.init(width, height, "OptiX Volume Renderer"))
      return 1;

    GL_CHECK(glViewport(0, 0, static_cast<GLsizei>(width),
                        static_cast<GLsizei>(height)));

    Display display;
    display.init(width, height);

    OptixContext optixContext;
    optixContext.init();

    PlaceholderScene scene;
    scene.init(optixContext.context(), optixContext.stream());

    const char *irPath = std::getenv("OPTIXIR_PATH");
    PipelineBase pipeline;
    pipeline.init(optixContext.context(), irPath ? irPath : OPTIXIR_PATH,
                  "__raygen__rg", "__miss__ms", "__closesthit__ch");

    InteropBuffer interopBuffer;
    interopBuffer.init(width, height);

    Camera camera;
    camera.aspect = static_cast<float>(width) / static_cast<float>(height);

    Params params = {};
    params.image_width = width;
    params.image_height = height;
    params.handle = scene.handle();

    CUdeviceptr d_params = 0;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));

    double lastTime = glfwGetTime();

    while (!window.shouldClose()) {
      glfwPollEvents();

      int fb_w, fb_h;
      window.framebufferSize(fb_w, fb_h);
      if (static_cast<unsigned int>(fb_w) != width ||
          static_cast<unsigned int>(fb_h) != height) {
        width = static_cast<unsigned int>(fb_w);
        height = static_cast<unsigned int>(fb_h);
        interopBuffer.resize(width, height);
        display.resize(width, height);
        camera.aspect = static_cast<float>(width) / static_cast<float>(height);
        params.image_width = width;
        params.image_height = height;
        glViewport(0, 0, static_cast<GLsizei>(width),
                   static_cast<GLsizei>(height));
      }

      double now = glfwGetTime();
      double dt = now - lastTime;
      lastTime = now;

      camera.theta += static_cast<float>(dt * 0.5);

      float3 cam_U, cam_V, cam_W;
      camera.uvw(cam_U, cam_V, cam_W);

      uchar4 *d_pixels = interopBuffer.map(optixContext.stream());
      params.image = d_pixels;

      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params,
                                 sizeof(Params), cudaMemcpyHostToDevice,
                                 optixContext.stream()));

      RayGenData rgData;
      rgData.cam_eye = camera.eye();
      rgData.camera_u = cam_U;
      rgData.camera_v = cam_V;
      rgData.camera_w = cam_W;
      pipeline.updateRayGenRecord(optixContext.stream(), rgData);

      OPTIX_CHECK(optixLaunch(pipeline.pipeline(), optixContext.stream(),
                              d_params, sizeof(Params), pipeline.sbt(), width,
                              height, 1));

      interopBuffer.unmap(optixContext.stream());

      display.present(interopBuffer.pbo());

      window.swapBuffers();
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
