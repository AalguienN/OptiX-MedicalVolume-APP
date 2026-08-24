#pragma once

#include <vector_functions.h>

#include <GLFW/glfw3.h>

class Display
{
public:
    Display()      = default;
    ~Display();

    Display(const Display&)            = delete;
    Display& operator=(const Display&) = delete;

    void init(unsigned int width, unsigned int height);
    void resize(unsigned int width, unsigned int height);

    void present(GLuint pbo);

private:
    void allocTexture();
    void destroy();

    GLuint       texture_  = 0;
    GLuint       program_  = 0;
    GLuint       vao_      = 0;
    GLint        texUniform_ = -1;
    unsigned int width_    = 0;
    unsigned int height_   = 0;
};
