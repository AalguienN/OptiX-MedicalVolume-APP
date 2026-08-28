include(OptiXIR)

set(DEVICE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/device/device_programs.cu")
add_optix_ir_target(OPTIXIR_FILE "${DEVICE_SRC}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/common"
)

set(APP_SOURCES
    app_main.cpp
    gl_window.cpp
    optix_context.cpp
    pipeline_base.cpp
    interop_buffer.cpp
    display.cpp
    placeholder_scene.cpp
    volume_scene.cpp
    dicom_loader.cpp
    volume.cpp
    ../common/metrics.cpp
)

if(WIN32)
    # Resolves GL entry points missing from the SDK's opengl32.lib.
    list(APPEND APP_SOURCES gl_shim_loader.cpp)
endif()

add_executable(optix_app
    ${APP_SOURCES}
    ${OPTIXIR_FILE}
)

target_include_directories(optix_app PRIVATE
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(optix_app PRIVATE
    OPTIXIR_PATH="${OPTIXIR_FILE}"
)

if(WIN32)
    # Minimal GL 2.0/3.0 prototypes (include/GL/glext.h) for SDK installs
    # that only ship the legacy GL/GL.h.
    target_include_directories(optix_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

if(WIN32)
    # Windows SDK legacy gl.h only declares GL 1.1; pull in the full GLEXT
    # prototypes (modern core functions) through the GLFW header.
    target_compile_definitions(optix_app PRIVATE
        GL_GLEXT_PROTOTYPES
        GLFW_INCLUDE_GLEXT
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

target_link_libraries(optix_app PRIVATE
    glfw
    OpenGL::GL
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
)

if(UNIX)
    target_link_libraries(optix_app PRIVATE dl m)
endif()

if(WIN32 AND EXISTS "${OPTIX_ROOT_DIR}/bin/glfw3.dll")
    # Ship the prebuilt GLFW DLL from the OptiX SDK next to the executable.
    add_custom_command(TARGET optix_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPTIX_ROOT_DIR}/bin/glfw3.dll" "$<TARGET_FILE_DIR:optix_app>"
        COMMENT "Copying glfw3.dll"
    )
endif()
