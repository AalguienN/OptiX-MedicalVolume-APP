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
#
# NANOVDB_INCLUDE is the include root: the directory that contains the
# 'nanovdb' folder (code includes <nanovdb/NanoVDB.h> and friends).
# Override with -DNANOVDB_INCLUDE=<dir> to use a local checkout instead.
#
# Linux keeps its original local OpenVDB checkout path. Windows has no such
# install, so a copy of the OpenVDB v13.0.0 headers is vendored under
# third_party/nanovdb to make the Windows build self-contained.
if(NOT DEFINED NANOVDB_INCLUDE)
    if(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/nanovdb/NanoVDB.h")
        set(NANOVDB_INCLUDE "${CMAKE_SOURCE_DIR}/third_party")
    elseif(WIN32)
        set(NANOVDB_INCLUDE "$ENV{USERPROFILE}/nvidia/nanovdb/openvdb/nanovdb")
    else()
        set(NANOVDB_INCLUDE "$ENV{HOME}/nvidia/nanovdb/openvdb/nanovdb")
    endif()
endif()
set(NANOVDB_INCLUDE "${NANOVDB_INCLUDE}" CACHE PATH "NanoVDB header root directory")
set(NANOVDB_INCLUDE_DIR "${NANOVDB_INCLUDE}")
if(NOT EXISTS "${NANOVDB_INCLUDE}/nanovdb/NanoVDB.h")
    message(FATAL_ERROR
        "NanoVDB headers not found: expected '${NANOVDB_INCLUDE}/nanovdb/NanoVDB.h'.\n"
        "Pass -DNANOVDB_INCLUDE=<dir> where <dir>/nanovdb/NanoVDB.h exists "
        "(on Windows the vendored copy is at ${CMAKE_SOURCE_DIR}/third_party).")
endif()

find_package(CUDA REQUIRED)

if(WIN32 AND NOT glfw3_DIR AND EXISTS "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
    set(glfw3_DIR "${OPTIX_ROOT_DIR}/lib/cmake/glfw3")
endif()
find_package(glfw3 REQUIRED)

find_package(OpenGL REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
