#include "camera.h"
#include "check_macros.h"
#include "dicom_loader.h"
#include "display.h"
#include "gl_check.h"
#include "gl_window.h"
#include "interop_buffer.h"
#include "optix_context.h"
#include "pipeline_base.h"
#include "placeholder_scene.h"
#include "shared_device.h"
#include "transfer_function.h"
#include "volume.h"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void glfwMouseButtonCallback(GLFWwindow *win, int button, int action, int /*mods*/) {
    auto *cam = static_cast<Camera *>(glfwGetWindowUserPointer(win));
    if (cam) cam->onMouseButton(button, action);
}

void glfwCursorPosCallback(GLFWwindow *win, double xpos, double ypos) {
    auto *cam = static_cast<Camera *>(glfwGetWindowUserPointer(win));
    if (cam) cam->onCursorPos(xpos, ypos);
}

void glfwScrollCallback(GLFWwindow *win, double /*xoffset*/, double yoffset) {
    auto *cam = static_cast<Camera *>(glfwGetWindowUserPointer(win));
    if (cam) cam->onScroll(yoffset);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path-to-dicom-series>\n";
        return 1;
    }

    std::string seriesPath = argv[1];
    unsigned int width = 1920;
    unsigned int height = 1080;

    try
    {
        std::cout << "Loading DICOM series from: " << seriesPath << "\n";
        DicomSeries series = loadDicomSeries(seriesPath);

        Volume volume;
        volume.buildFromSeries(series);

        TransferFunction tf;
        if (volume.modality == "CT")
            tf.buildDefaultCT();
        else
            tf.buildDefaultMR();

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
        {
            float volSpanX = volume.dimX * volume.spacingX;
            float volSpanY = volume.dimY * volume.spacingY;
            float volSpanZ = volume.dimZ * volume.spacingZ;
            camera.radius = fmaxf(volSpanX, fmaxf(volSpanY, volSpanZ)) * 1.5f;
            camera.lookAt = make_float3(
                volume.originX + volSpanX * 0.5f,
                volume.originY + volSpanY * 0.5f,
                volume.originZ + volSpanZ * 0.5f);
            camera.theta = 0.0f;
            camera.phi   = 0.3f;
        }

        float vmin = 1e30f, vmax = -1e30f;
        {
            size_t total = volume.data.size();
            size_t nonzero = 0;
            for (size_t i = 0; i < total; ++i)
            {
                float v = volume.data[i];
                if (v != 0.0f) nonzero++;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            std::cout << "Volume stats: " << nonzero << "/" << total
                      << " nonzero, range=[" << vmin << ", " << vmax << "]\n";

            int cx = volume.dimX / 2, cy = volume.dimY / 2, cz = volume.dimZ / 2;
            std::cout << "Center voxel [" << cx << "," << cy << "," << cz << "] = "
                      << volume.data[cz * volume.dimX * volume.dimY + cy * volume.dimX + cx] << "\n";
            std::cout << "Origin voxel [0,0,0] = " << volume.data[0] << "\n";

            int midZ = volume.dimZ / 2;
            int nnz = 0;
            for (int y = 0; y < volume.dimY; ++y)
                for (int x = 0; x < volume.dimX; ++x)
                    if (volume.data[midZ * volume.dimX * volume.dimY + y * volume.dimX + x] != 0.0f)
                        nnz++;
            std::cout << "Middle slice (z=" << midZ << "): " << nnz << "/" << volume.dimX * volume.dimY << " nonzero\n";
        }

        glfwSetWindowUserPointer(window.handle(), &camera);
        glfwSetMouseButtonCallback(window.handle(), glfwMouseButtonCallback);
        glfwSetCursorPosCallback(window.handle(), glfwCursorPosCallback);
        glfwSetScrollCallback(window.handle(), glfwScrollCallback);

        double xpos, ypos;
        glfwGetCursorPos(window.handle(), &xpos, &ypos);
        camera.lastPos = make_float2(static_cast<float>(xpos),
                                     static_cast<float>(ypos));

        cudaArray* d_volumeArray = nullptr;
        cudaTextureObject_t volumeTex = 0;
        volume.uploadToDevice(&d_volumeArray, &volumeTex);

        float4* d_tfData = nullptr;
        {
            size_t tfBytes = sizeof(float4) * TransferFunction::LUT_SIZE;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tfData), tfBytes));
            CUDA_CHECK(cudaMemcpy(d_tfData, tf.lut, tfBytes, cudaMemcpyHostToDevice));
        }

        Params params = {};
        params.image_width    = width;
        params.image_height   = height;
        params.handle         = scene.handle();
        params.volumeTex      = volumeTex;
        params.tfData          = d_tfData;
        params.volumeDims     = make_int3(volume.dimX, volume.dimY, volume.dimZ);
        params.volumeSpacing  = make_float3(volume.spacingX, volume.spacingY, volume.spacingZ);
        params.volumeOrigin   = make_float3(volume.originX, volume.originY, volume.originZ);
        params.volumeMax      = make_float3(volume.maxX(), volume.maxY(), volume.maxZ());

        if (volume.modality == "CT") {
            params.scalarMin = -1000.0f;
            params.scalarMax =  1000.0f;
        } else {
            params.scalarMin = 0.0f;
            params.scalarMax = 4096.0f;
        }

        CUdeviceptr d_params = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));

        std::cout << "Starting render loop...\n";

        while (!window.shouldClose())
        {
            glfwPollEvents();

            int fb_w, fb_h;
            window.framebufferSize(fb_w, fb_h);
            if (static_cast<unsigned int>(fb_w) != width ||
                static_cast<unsigned int>(fb_h) != height)
            {
                width  = static_cast<unsigned int>(fb_w);
                height = static_cast<unsigned int>(fb_h);
                interopBuffer.resize(width, height);
                display.resize(width, height);
                camera.aspect = static_cast<float>(width) / static_cast<float>(height);
                params.image_width  = width;
                params.image_height = height;
                glViewport(0, 0, static_cast<GLsizei>(width),
                           static_cast<GLsizei>(height));
            }

            float3 cam_U, cam_V, cam_W;
            camera.uvw(cam_U, cam_V, cam_W);

            uchar4 *d_pixels = interopBuffer.map(optixContext.stream());
            params.image = d_pixels;

            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &params,
                                       sizeof(Params), cudaMemcpyHostToDevice,
                                       optixContext.stream()));

            RayGenData rgData;
            rgData.cam_eye   = camera.eye();
            rgData.camera_u  = cam_U;
            rgData.camera_v  = cam_V;
            rgData.camera_w  = cam_W;
            pipeline.updateRayGenRecord(optixContext.stream(), rgData);

            OPTIX_CHECK(optixLaunch(pipeline.pipeline(), optixContext.stream(),
                                    d_params, sizeof(Params), pipeline.sbt(),
                                    width, height, 1));

            interopBuffer.unmap(optixContext.stream());

            display.present(interopBuffer.pbo());

            window.swapBuffers();
        }

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
        volume.destroyDevice(d_volumeArray, volumeTex);
        if (d_tfData) cudaFree(d_tfData);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
