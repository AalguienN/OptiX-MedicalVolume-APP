include(OptiXIR)

set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device/device_programs.cu")
add_optix_ir_target(OPTIXIR_FILE "${DEVICE_SRC}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/common"
    "${NANOVDB_INCLUDE_DIR}"
)

add_executable(optix_app
    app_main.cpp
    gl_window.cpp
    optix_context.cpp
    pipeline_base.cpp
    interop_buffer.cpp
    display.cpp
    placeholder_scene.cpp
    volume_scene.cpp
    regions_scene.cpp
    bricked_volume.cpp
    adaptive_grid_marcher.cpp
    octree_volume.cpp
    octree_regions_scene.cpp
    nanovdb_volume.cpp
    dicom_loader.cpp
    volume.cpp
    ../common/metrics.cpp
    ${OPTIXIR_FILE}
)

target_include_directories(optix_app PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${NANOVDB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(optix_app PRIVATE
    OPTIXIR_PATH="${OPTIXIR_FILE}"
)

target_link_libraries(optix_app PRIVATE
    glfw
    OpenGL::GL
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    dl
)

if(UNIX)
    target_link_libraries(optix_app PRIVATE m)
endif()
