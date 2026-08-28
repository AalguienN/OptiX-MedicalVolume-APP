#include <GL/glext.h>

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32

/* Definitions of the function pointers declared extern in <GL/glext.h>
 * (extern "C" in the header, so the definitions match). */
extern "C" {
GLEXTSHIM_PFN_glActiveTexture      glActiveTexture;
GLEXTSHIM_PFN_glAttachShader       glAttachShader;
GLEXTSHIM_PFN_glBindBuffer         glBindBuffer;
GLEXTSHIM_PFN_glBindVertexArray    glBindVertexArray;
GLEXTSHIM_PFN_glBufferData         glBufferData;
GLEXTSHIM_PFN_glCompileShader      glCompileShader;
GLEXTSHIM_PFN_glCreateProgram      glCreateProgram;
GLEXTSHIM_PFN_glCreateShader       glCreateShader;
GLEXTSHIM_PFN_glDeleteBuffers      glDeleteBuffers;
GLEXTSHIM_PFN_glDeleteProgram      glDeleteProgram;
GLEXTSHIM_PFN_glDeleteShader       glDeleteShader;
GLEXTSHIM_PFN_glDeleteVertexArrays glDeleteVertexArrays;
GLEXTSHIM_PFN_glGenBuffers         glGenBuffers;
GLEXTSHIM_PFN_glGenVertexArrays    glGenVertexArrays;
GLEXTSHIM_PFN_glGetProgramInfoLog  glGetProgramInfoLog;
GLEXTSHIM_PFN_glGetProgramiv       glGetProgramiv;
GLEXTSHIM_PFN_glGetShaderInfoLog   glGetShaderInfoLog;
GLEXTSHIM_PFN_glGetShaderiv        glGetShaderiv;
GLEXTSHIM_PFN_glGetUniformLocation glGetUniformLocation;
GLEXTSHIM_PFN_glLinkProgram        glLinkProgram;
GLEXTSHIM_PFN_glShaderSource       glShaderSource;
GLEXTSHIM_PFN_glUniform1i          glUniform1i;
GLEXTSHIM_PFN_glUseProgram         glUseProgram;
}

namespace {

/* wglGetProcAddress() is declared by the SDK (wingdi.h) and resolves entry
 * points that the import library does not cover. */
template <typename T>
bool glShimResolve(T& slot, const char* name)
{
    void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
    if (!p)
    {
        std::fprintf(stderr, "glShimLoadModern: failed to resolve %s\n", name);
        return false;
    }
    slot = reinterpret_cast<T>(p);
    return true;
}

}  // namespace

void glShimLoadModern(void)
{
    bool ok = true;

    ok = ok && glShimResolve(glActiveTexture, "glActiveTexture");
    ok = ok && glShimResolve(glAttachShader, "glAttachShader");
    ok = ok && glShimResolve(glBindBuffer, "glBindBuffer");
    ok = ok && glShimResolve(glBindVertexArray, "glBindVertexArray");
    ok = ok && glShimResolve(glBufferData, "glBufferData");
    ok = ok && glShimResolve(glCompileShader, "glCompileShader");
    ok = ok && glShimResolve(glCreateProgram, "glCreateProgram");
    ok = ok && glShimResolve(glCreateShader, "glCreateShader");
    ok = ok && glShimResolve(glDeleteBuffers, "glDeleteBuffers");
    ok = ok && glShimResolve(glDeleteProgram, "glDeleteProgram");
    ok = ok && glShimResolve(glDeleteShader, "glDeleteShader");
    ok = ok && glShimResolve(glDeleteVertexArrays, "glDeleteVertexArrays");
    ok = ok && glShimResolve(glGenBuffers, "glGenBuffers");
    ok = ok && glShimResolve(glGenVertexArrays, "glGenVertexArrays");
    ok = ok && glShimResolve(glGetProgramInfoLog, "glGetProgramInfoLog");
    ok = ok && glShimResolve(glGetProgramiv, "glGetProgramiv");
    ok = ok && glShimResolve(glGetShaderInfoLog, "glGetShaderInfoLog");
    ok = ok && glShimResolve(glGetShaderiv, "glGetShaderiv");
    ok = ok && glShimResolve(glGetUniformLocation, "glGetUniformLocation");
    ok = ok && glShimResolve(glLinkProgram, "glLinkProgram");
    ok = ok && glShimResolve(glShaderSource, "glShaderSource");
    ok = ok && glShimResolve(glUniform1i, "glUniform1i");
    ok = ok && glShimResolve(glUseProgram, "glUseProgram");

    if (!ok)
        std::exit(1);
}

#endif  // _WIN32
