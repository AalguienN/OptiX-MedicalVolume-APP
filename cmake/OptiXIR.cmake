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

    # Also rebuild when any sibling .cu, .h or .hpp in the source directory
    # changes. The app's device programs are #included from a single entry .cu
    # (so all strategy programs and their shared headers land in one OptiX IR
    # module), and those included files must trigger a recompile here.
    get_filename_component(_src_dir "${source_file}" DIRECTORY)
    file(GLOB _device_deps "${_src_dir}/*.cu" "${_src_dir}/*.h" "${_src_dir}/*.hpp")

    add_custom_command(
        OUTPUT "${_output}"
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            --optix-ir
            -arch=${CUDA_ARCH}
            ${_std_flag}
            -I${OPTIX_INCLUDE_DIR}
            ${_extra_includes}
            ${_host_flags}
            -o "${_output}"
            "${source_file}"
        DEPENDS "${source_file}" ${_device_deps}
        COMMENT "Compiling ${source_file} to OptiX IR"
    )
endfunction()
