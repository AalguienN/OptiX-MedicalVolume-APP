#include "gl_window.h"

#include <iostream>

#include <GLFW/glfw3.h>

namespace {

void errorCallback(int error, const char* desc)
{
    std::cerr << "GLFW error " << error << ": " << desc << "\n";
}

}  // namespace

GlWindow::~GlWindow()
{
    if (window_)
        glfwDestroyWindow(window_);
    glfwTerminate();
}

bool GlWindow::init(unsigned int width, unsigned int height, const char* title)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title, nullptr, nullptr);
    if (!window_)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
#ifdef _WIN32
    glShimLoadModern();
#endif
    glfwSwapInterval(1);

    return true;
}

bool GlWindow::shouldClose() const
{
    return glfwWindowShouldClose(window_) != 0;
}

void GlWindow::framebufferSize(int& width, int& height) const
{
    glfwGetFramebufferSize(window_, &width, &height);
}

void GlWindow::swapBuffers()
{
    glfwSwapBuffers(window_);
}
