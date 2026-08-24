cmake_minimum_required(VERSION 3.14)
project(optix_clone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(OPTIX_ROOT_DIR "$ENV{HOME}/nvidia/OptiX91" CACHE PATH "OptiX SDK root directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")

find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
