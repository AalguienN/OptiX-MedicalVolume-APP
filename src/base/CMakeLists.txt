set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/base.cu")
set(OPTIXIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/base.optixir")

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
    COMMENT "Compiling base.cu to OptiX IR"
)

add_executable(base
    base.cpp
    ${OPTIXIR_FILE}
)

target_include_directories(base PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(base PRIVATE
    OPTIXIR_PATH="${OPTIXIR_FILE}"
)

target_link_libraries(base PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    dl
)

if(UNIX)
    target_link_libraries(base PRIVATE m)
endif()
