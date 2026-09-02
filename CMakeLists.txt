cmake_minimum_required(VERSION 3.14)
project(optix_clone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    # NanoVDB headers include CCCL (cuda/std), which requires the
    # standard-conforming MSVC preprocessor.
    add_compile_options(/Zc:preprocessor)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(WIN32)
    set(OPTIX_ROOT_DIR_DEFAULT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
else()
    set(OPTIX_ROOT_DIR_DEFAULT "$ENV{HOME}/nvidia/OptiX91")
endif()

set(OPTIX_ROOT_DIR "${OPTIX_ROOT_DIR_DEFAULT}" CACHE PATH "OptiX SDK root directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")

# Header-only NanoVDB library (part of the OpenVDB project). Only the headers
# are needed; the NanoVDB CUDA traversal, GridBuilder and CreateNanoGrid tools
# are all header-only and depend on nothing else.
set(NANOVDB_INCLUDE "$ENV{HOME}/nvidia/nanovdb/openvdb/nanovdb" CACHE PATH "NanoVDB header root directory")
set(NANOVDB_INCLUDE_DIR "${NANOVDB_INCLUDE}")

find_package(CUDA REQUIRED)

if(WIN32 AND NOT glfw3_DIR AND EXISTS "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
    set(glfw3_DIR "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
endif()
find_package(glfw3 REQUIRED)

find_package(OpenGL REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
