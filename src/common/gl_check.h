#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define GL_GLEXT_PROTOTYPES 1
#include <GLFW/glfw3.h>

#define GL_CHECK(call)                                                         \
    do {                                                                       \
        call;                                                                  \
        GLenum err = glGetError();                                             \
        if (err != GL_NO_ERROR) {                                              \
            std::stringstream ss;                                              \
            ss << "GL error 0x" << std::hex << err << " after '" << #call      \
               << "' (" << __FILE__ << ":" << __LINE__ << ")\n";               \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (false)

#define GL_CHECK_NOEXCEPT(call)                                                \
    do {                                                                       \
        call;                                                                  \
        GLenum err = glGetError();                                             \
        if (err != GL_NO_ERROR)                                                \
            std::cerr << "GL error 0x" << std::hex << err << " after '"        \
                      << #call << "' during cleanup\n";                        \
    } while (false)
