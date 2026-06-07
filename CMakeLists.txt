cmake_minimum_required(VERSION 3.14)
project(optixConsoleStandalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPTIX_ROOT_DIR "$ENV{HOME}/nvidia/OptiX91" CACHE PATH "OptiX SDK root directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")

find_package(CUDA REQUIRED)

set(CUDA_ARCH "sm_120" CACHE STRING "CUDA architecture for OptiX IR compilation")

set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/optixConsole.cu")
set(OPTIXIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/optixConsole.optixir")

add_custom_command(
    OUTPUT ${OPTIXIR_FILE}
    COMMAND ${CUDA_NVCC_EXECUTABLE}
        --optix-ir
        -arch=${CUDA_ARCH}
        -I${OPTIX_INCLUDE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}/src
        -o ${OPTIXIR_FILE}
        ${DEVICE_SRC}
    DEPENDS ${DEVICE_SRC}
    COMMENT "Compiling optixConsole.cu to OptiX IR"
)

add_executable(optixConsole
    src/optixConsole.cpp
    ${OPTIXIR_FILE}
)

target_include_directories(optixConsole PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(optixConsole PRIVATE
    OPTIX_CONSOLE_OPTIXIR_PATH="${OPTIXIR_FILE}"
)

target_link_libraries(optixConsole PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    dl
)

if(UNIX)
    target_link_libraries(optixConsole PRIVATE m)
endif()
