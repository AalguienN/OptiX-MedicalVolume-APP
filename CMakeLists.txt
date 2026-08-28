cmake_minimum_required(VERSION 3.14)
project(optix_clone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(WIN32)
    set(OPTIX_ROOT_DIR_DEFAULT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
else()
    set(OPTIX_ROOT_DIR_DEFAULT "$ENV{HOME}/nvidia/Optix91")
endif()

set(OPTIX_ROOT_DIR "${OPTIX_ROOT_DIR_DEFAULT}" CACHE PATH "OptiX SDK root directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")

find_package(CUDA REQUIRED)

if(WIN32 AND NOT glfw3_DIR AND EXISTS "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
    set(glfw3_DIR "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
endif()
find_package(glfw3 REQUIRED)

find_package(OpenGL REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
