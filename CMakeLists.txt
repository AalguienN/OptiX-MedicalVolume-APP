cmake_minimum_required(VERSION 3.14)
project(optixConsoleStandalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPTIX_ROOT_DIR "$ENV{HOME}/nvidia/OptiX91" CACHE PATH "OptiX SDK root directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")

find_package(CUDA REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

add_subdirectory(src)
