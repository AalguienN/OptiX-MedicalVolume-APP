#pragma once

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <GLFW/glfw3.h>

class InteropBuffer
{
public:
    InteropBuffer()      = default;
    ~InteropBuffer();

    InteropBuffer(const InteropBuffer&)            = delete;
    InteropBuffer& operator=(const InteropBuffer&) = delete;

    void init(unsigned int width, unsigned int height);
    void resize(unsigned int width, unsigned int height);

    uchar4* map(CUstream stream);
    void    unmap(CUstream stream);

    GLuint       pbo() const { return pbo_; }
    unsigned int width() const { return width_; }
    unsigned int height() const { return height_; }

private:
    void alloc();
    void destroy();

    GLuint                pbo_      = 0;
    cudaGraphicsResource* resource_ = nullptr;
    unsigned int          width_    = 0;
    unsigned int          height_   = 0;
};
