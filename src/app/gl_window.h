#pragma once

struct GLFWwindow;

class GlWindow
{
public:
    GlWindow()      = default;
    ~GlWindow();

    GlWindow(const GlWindow&)            = delete;
    GlWindow& operator=(const GlWindow&) = delete;

    bool init(unsigned int width, unsigned int height, const char* title);

    GLFWwindow* handle() const { return window_; }

    bool shouldClose() const;
    void framebufferSize(int& width, int& height) const;
    void swapBuffers();

private:
    GLFWwindow* window_ = nullptr;
};
