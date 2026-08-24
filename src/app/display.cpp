#include "gl_check.h"

#include "display.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace {

const char *vertex_shader_src = R"(
#version 330 core
out vec2 uv;
void main() {
    const vec2 verts[6] = vec2[](
        vec2(-1,-1), vec2( 1,-1), vec2( 1, 1),
        vec2(-1,-1), vec2( 1, 1), vec2(-1, 1)
    );
    vec2 p = verts[gl_VertexID];
    uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

const char *fragment_shader_src = R"(
#version 330 core
in vec2 uv;
uniform sampler2D tex;
out vec4 color;
void main() {
    color = texture(tex, uv);
}
)";

GLuint compileShader(GLenum type, const char *src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);
  GLint ok;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    glDeleteShader(shader);
    std::string typeName = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
    throw std::runtime_error(typeName + " shader compile error:\n" +
                             std::string(log));
  }
  return shader;
}

} // namespace

Display::~Display() { destroy(); }

void Display::init(unsigned int width, unsigned int height) {
  width_ = width;
  height_ = height;

  GLuint vs = compileShader(GL_VERTEX_SHADER, vertex_shader_src);
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragment_shader_src);

  program_ = glCreateProgram();
  glAttachShader(program_, vs);
  glAttachShader(program_, fs);
  glLinkProgram(program_);
  GLint ok;
  glGetProgramiv(program_, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
    glDeleteShader(vs);
    glDeleteShader(fs);
    destroy();
    throw std::runtime_error(std::string("program link error:\n") + log);
  }
  glDeleteShader(vs);
  glDeleteShader(fs);

  texUniform_ = glGetUniformLocation(program_, "tex");

  GL_CHECK(glGenVertexArrays(1, &vao_));

  allocTexture();
}

void Display::resize(unsigned int width, unsigned int height) {
  if (width == width_ && height == height_)
    return;
  width_ = width;
  height_ = height;
  allocTexture();
}

void Display::allocTexture() {
  if (texture_)
    GL_CHECK(glDeleteTextures(1, &texture_));

  GL_CHECK(glGenTextures(1, &texture_));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  GL_CHECK(glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(width_),
      static_cast<GLsizei>(height_), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
}

void Display::present(GLuint pbo) {
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
  GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width_),
                           static_cast<GLsizei>(height_), GL_RGBA,
                           GL_UNSIGNED_BYTE, nullptr));
  GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

  GL_CHECK(glUseProgram(program_));
  GL_CHECK(glUniform1i(texUniform_, 0));
  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_));
  GL_CHECK(glBindVertexArray(vao_));
  GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));

  GL_CHECK(glBindVertexArray(0));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  GL_CHECK(glUseProgram(0));
}

void Display::destroy() {
  if (program_)
    GL_CHECK_NOEXCEPT(glDeleteProgram(program_));
  if (vao_)
    GL_CHECK_NOEXCEPT(glDeleteVertexArrays(1, &vao_));
  if (texture_)
    GL_CHECK_NOEXCEPT(glDeleteTextures(1, &texture_));
  program_ = 0;
  vao_ = 0;
  texture_ = 0;
  texUniform_ = -1;
}
