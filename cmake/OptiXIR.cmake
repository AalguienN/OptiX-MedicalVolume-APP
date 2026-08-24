# Compiles a CUDA source file to OptiX IR.
#
#   add_optix_ir_target(<output-var> <source.cu> [extra include dir]...)
#
# Sets <output-var> to the path of the generated .optixir file in the
# current binary directory and registers a custom command producing it.
function(add_optix_ir_target output_var source_file)
    get_filename_component(_name_we "${source_file}" NAME_WE)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/${_name_we}.optixir")
    set(${output_var} "${_output}" PARENT_SCOPE)

    set(_extra_includes "")
    foreach(_dir IN LISTS ARGN)
        list(APPEND _extra_includes "-I${_dir}")
    endforeach()

    add_custom_command(
        OUTPUT "${_output}"
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            --optix-ir
            -arch=${CUDA_ARCH}
            -I${OPTIX_INCLUDE_DIR}
            ${_extra_includes}
            -o "${_output}"
            "${source_file}"
        DEPENDS "${source_file}"
        COMMENT "Compiling ${source_file} to OptiX IR"
    )
endfunction()
