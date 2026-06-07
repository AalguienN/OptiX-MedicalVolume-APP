#include <cstdio>
#include <optix.h>
#include <optix_host.h>

int main() {
  optixInit();

  optixDeviceContext deviceContext;
  optixDeviceContextCreateFromCurrentContext(nullptr, &deviceContext);

  printf("OptiX initialized successfully!\n");
  printf("Checking CUDA context...\n");

  optixDeviceContextDestroy(deviceContext);
  optixShutdown();

  return 0;
}
