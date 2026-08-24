#include "check_macros.h"
#include "gl_check.h"
#include "interop_buffer.h"

InteropBuffer::~InteropBuffer()
{
    destroy();
}

void InteropBuffer::init(unsigned int width, unsigned int height)
{
    width_  = width;
    height_ = height;
    alloc();
}

void InteropBuffer::resize(unsigned int width, unsigned int height)
{
    if (width == width_ && height == height_)
        return;
    destroy();
    width_  = width;
    height_ = height;
    alloc();
}

void InteropBuffer::alloc()
{
    GL_CHECK(glGenBuffers(1, &pbo_));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER,
                          static_cast<GLsizeiptr>(width_) * height_ * sizeof(uchar4),
                          nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource_, pbo_, cudaGraphicsMapFlagsWriteDiscard));
}

uchar4* InteropBuffer::map(CUstream stream)
{
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource_, stream));
    uchar4* ptr = nullptr;
    size_t  size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size, resource_));
    return ptr;
}

void InteropBuffer::unmap(CUstream stream)
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource_, stream));
}

void InteropBuffer::destroy()
{
    if (resource_)
        CUDA_CHECK_NOEXCEPT(cudaGraphicsUnregisterResource(resource_));
    if (pbo_)
        GL_CHECK_NOEXCEPT(glDeleteBuffers(1, &pbo_));
    resource_ = nullptr;
    pbo_      = 0;
    width_    = 0;
    height_   = 0;
}
