find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test.cu")
set(OPTIXIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/test.optixir")

add_custom_command(
    OUTPUT ${OPTIXIR_FILE}
    COMMAND ${CUDA_NVCC_EXECUTABLE}
        --optix-ir
        -arch=${CUDA_ARCH}
        -I${OPTIX_INCLUDE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_SOURCE_DIR}/src/common
        -o ${OPTIXIR_FILE}
        ${DEVICE_SRC}
    DEPENDS ${DEVICE_SRC}
    COMMENT "Compiling test.cu to OptiX IR"
)

add_executable(test
    test.cpp
    ${OPTIXIR_FILE}
)

target_include_directories(test PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(test PRIVATE
    OPTIXIR_PATH="${OPTIXIR_FILE}"
)

target_link_libraries(test PRIVATE
    glfw
    OpenGL::GL
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    dl
)

if(UNIX)
    target_link_libraries(test PRIVATE m)
endif()
