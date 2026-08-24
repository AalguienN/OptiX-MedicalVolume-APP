#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

#define CUDA_CHECK_NOEXCEPT(call)                                              \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess)                                                \
            std::cerr << "CUDA call '" << #call << "' failed during cleanup: " \
                      << cudaGetErrorString(err) << "\n";                      \
    } while (false)

#define OPTIX_CHECK_NOEXCEPT(call)                                             \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS)                                              \
            std::cerr << "Optix call '" << #call                               \
                      << "' failed during cleanup: "                           \
                      << optixGetErrorName(res) << "\n";                       \
    } while (false)
