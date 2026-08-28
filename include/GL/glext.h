/*
 * Minimal OpenGL core (GL 2.0/3.0) prototypes for Windows builds.
 *
 * Some Windows SDK installations only ship the legacy GL/GL.h header
 * (GL 1.1 types and functions) without GL/glext.h, and the SDK's
 * opengl32.lib import library only covers the legacy entry points, so
 * glfw3.h (which includes <GL/glext.h> when GLFW_INCLUDE_GLEXT is set)
 * cannot resolve the modern core function prototypes, constants and
 * import symbols.
 *
 * This header addresses both problems:
 *   - it declares the constants and the prototypes the project uses;
 *     the WINGDIAPI/APIENTRY decoration matches the legacy GL/GL.h so
 *     overlapping declarations stay consistent;
 *   - the entry points missing from opengl32.lib are declared as
 *     function pointers and resolved at runtime with
 *     wglGetProcAddress by glShimLoadModern(), which must be called
 *     once after the GL context has been made current.
 *
 * It is only placed on the include path on Windows. On Linux the system
 * GLVND headers are used instead.
 */
#ifndef _GLEXT_SHIM_H_
#define _GLEXT_SHIM_H_

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#define GLEXT_SHIM_IMPORT WINGDIAPI
#define GLEXT_SHIM_CC     APIENTRY
#else
#define GLEXT_SHIM_IMPORT
#define GLEXT_SHIM_CC
#endif

/* Base GL types; on Windows this is the legacy GL 1.1 header, which also
 * declares the entry points that the SDK's opengl32.lib covers. */
#include <GL/gl.h>

#ifndef _GLEXT_SHIM_TYPES_
#define _GLEXT_SHIM_TYPES_
typedef ptrdiff_t GLsizeiptr;
typedef char      GLchar;
#endif

#define GL_ARRAY_BUFFER          0x8892
#define GL_PIXEL_UNPACK_BUFFER   0x88EC
#define GL_STREAM_DRAW           0x88E0
#define GL_CLAMP_TO_EDGE         0x812F
#define GL_TEXTURE0              0x84C0
#define GL_FRAGMENT_SHADER       0x8B30
#define GL_VERTEX_SHADER         0x8B31
#define GL_COMPILE_STATUS        0x8B81
#define GL_LINK_STATUS           0x8B82
#define GL_RGBA8                 0x8058

#ifdef __cplusplus
extern "C" {
#endif

/* Entry points provided by the SDK's opengl32.lib. */

GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glBindTexture(GLenum target, GLuint texture);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glBlendFunc(GLenum sfactor, GLenum dfactor);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glDeleteTextures(GLsizei n, const GLuint* textures);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glDisable(GLenum cap);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glDrawArrays(GLenum mode, GLint first, GLsizei count);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glEnable(GLenum cap);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glEnableVertexAttribArray(GLuint index);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glGenTextures(GLsizei n, GLuint* textures);
GLEXT_SHIM_IMPORT GLenum GLEXT_SHIM_CC glGetError(void);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width,
                                                    GLsizei height, GLint border, GLenum format, GLenum type, const void* pixels);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glTexParameteri(GLenum target, GLenum pname, GLint params);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width,
                                                       GLsizei height, GLenum format, GLenum type, const void* pixels);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized,
                                                             GLsizei stride, const void* pointer);
GLEXT_SHIM_IMPORT void   GLEXT_SHIM_CC glViewport(GLint x, GLint y, GLsizei width, GLsizei height);

#ifdef _WIN32

/* Entry points missing from the SDK's opengl32.lib. Declared as function
 * pointers; glShimLoadModern() resolves them with wglGetProcAddress. */

typedef void   (APIENTRY *GLEXTSHIM_PFN_glActiveTexture)(GLenum texture);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glAttachShader)(GLuint program, GLuint shader);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glBindBuffer)(GLenum target, GLuint buffer);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glBindVertexArray)(GLuint array);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glBufferData)(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glCompileShader)(GLuint shader);
typedef GLuint (APIENTRY *GLEXTSHIM_PFN_glCreateProgram)(void);
typedef GLuint (APIENTRY *GLEXTSHIM_PFN_glCreateShader)(GLenum type);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glDeleteBuffers)(GLsizei n, const GLuint* buffers);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glDeleteProgram)(GLuint program);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glDeleteShader)(GLuint shader);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glDeleteVertexArrays)(GLsizei n, const GLuint* arrays);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGenBuffers)(GLsizei n, GLuint* buffers);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGenVertexArrays)(GLsizei n, GLuint* arrays);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGetProgramInfoLog)(GLuint program, GLsizei bufsize, GLsizei* length, GLchar* infoLog);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGetProgramiv)(GLuint program, GLenum pname, GLint* params);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGetShaderInfoLog)(GLuint shader, GLsizei bufsize, GLsizei* length, GLchar* infoLog);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glGetShaderiv)(GLuint shader, GLenum pname, GLint* params);
typedef GLint  (APIENTRY *GLEXTSHIM_PFN_glGetUniformLocation)(GLuint program, const GLchar* name);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glLinkProgram)(GLuint program);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glShaderSource)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glUniform1i)(GLint location, GLint v0);
typedef void   (APIENTRY *GLEXTSHIM_PFN_glUseProgram)(GLuint program);

extern GLEXTSHIM_PFN_glActiveTexture      glActiveTexture;
extern GLEXTSHIM_PFN_glAttachShader       glAttachShader;
extern GLEXTSHIM_PFN_glBindBuffer         glBindBuffer;
extern GLEXTSHIM_PFN_glBindVertexArray    glBindVertexArray;
extern GLEXTSHIM_PFN_glBufferData         glBufferData;
extern GLEXTSHIM_PFN_glCompileShader      glCompileShader;
extern GLEXTSHIM_PFN_glCreateProgram      glCreateProgram;
extern GLEXTSHIM_PFN_glCreateShader       glCreateShader;
extern GLEXTSHIM_PFN_glDeleteBuffers      glDeleteBuffers;
extern GLEXTSHIM_PFN_glDeleteProgram      glDeleteProgram;
extern GLEXTSHIM_PFN_glDeleteShader       glDeleteShader;
extern GLEXTSHIM_PFN_glDeleteVertexArrays glDeleteVertexArrays;
extern GLEXTSHIM_PFN_glGenBuffers         glGenBuffers;
extern GLEXTSHIM_PFN_glGenVertexArrays    glGenVertexArrays;
extern GLEXTSHIM_PFN_glGetProgramInfoLog  glGetProgramInfoLog;
extern GLEXTSHIM_PFN_glGetProgramiv       glGetProgramiv;
extern GLEXTSHIM_PFN_glGetShaderInfoLog   glGetShaderInfoLog;
extern GLEXTSHIM_PFN_glGetShaderiv        glGetShaderiv;
extern GLEXTSHIM_PFN_glGetUniformLocation glGetUniformLocation;
extern GLEXTSHIM_PFN_glLinkProgram        glLinkProgram;
extern GLEXTSHIM_PFN_glShaderSource       glShaderSource;
extern GLEXTSHIM_PFN_glUniform1i          glUniform1i;
extern GLEXTSHIM_PFN_glUseProgram         glUseProgram;

/* Resolves the function pointers above. Call once after the GL context
 * has been made current; exits the process if a lookup fails. */
void glShimLoadModern(void);

#else

/* Non-Windows fallback (this header is only used on Windows, but keep it
 * valid elsewhere). */

void  glActiveTexture(GLenum texture);
void  glAttachShader(GLuint program, GLuint shader);
void  glBindBuffer(GLenum target, GLuint buffer);
void  glBindVertexArray(GLuint array);
void  glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
void  glCompileShader(GLuint shader);
GLuint glCreateProgram(void);
GLuint glCreateShader(GLenum type);
void  glDeleteBuffers(GLsizei n, const GLuint* buffers);
void  glDeleteProgram(GLuint program);
void  glDeleteShader(GLuint shader);
void  glDeleteVertexArrays(GLsizei n, const GLuint* arrays);
void  glGenBuffers(GLsizei n, GLuint* buffers);
void  glGenVertexArrays(GLsizei n, GLuint* arrays);
void  glGetProgramInfoLog(GLuint program, GLsizei bufsize, GLsizei* length, GLchar* infoLog);
void  glGetProgramiv(GLuint program, GLenum pname, GLint* params);
void  glGetShaderInfoLog(GLuint shader, GLsizei bufsize, GLsizei* length, GLchar* infoLog);
void  glGetShaderiv(GLuint shader, GLenum pname, GLint* params);
GLint glGetUniformLocation(GLuint program, const GLchar* name);
void  glLinkProgram(GLuint program);
void  glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
void  glUniform1i(GLint location, GLint v0);
void  glUseProgram(GLuint program);

inline void glShimLoadModern(void)
{
}

#endif  // _WIN32

#ifdef __cplusplus
}
#endif

#endif /* _GLEXT_SHIM_H_ */
