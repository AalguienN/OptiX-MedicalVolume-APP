#pragma once

#include "vec_math.h"
#include <vector_functions.h>

class Camera {
public:
  float theta = 0.0f;
  float phi = 0.3f;
  float radius = 4.0f;
  float fovY = 45.0f;
  float aspect = 1.0f;

  float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);

  float3 eye() const {
    return lookAt + make_float3(radius * cosf(phi) * sinf(theta),
                                -radius * cosf(phi) * cosf(theta),
                                radius * sinf(phi));
  }

  void uvw(float3 &U, float3 &V, float3 &W) const {
    float3 cam_eye = eye();
    float3 cam_up = make_float3(0.0f, 0.0f, 1.0f);

    float3 w = lookAt - cam_eye;
    float wlen = length(w);
    W = w;
    U = normalize(cross(w, cam_up));
    V = normalize(cross(U, w));

    float vlen = wlen * tanf(0.5f * fovY * static_cast<float>(M_PIf) / 180.0f);
    V = V * vlen;
    float ulen = vlen * aspect;
    U = U * ulen;
  }

  void handleOrbit(float dx, float dy) {
    constexpr float kSensitivity = 0.005f;
    theta -= dx * kSensitivity;
    phi += dy * kSensitivity;

    constexpr float kMaxPhi = 1.5533f;
    if (phi > kMaxPhi)
      phi = kMaxPhi;
    if (phi < -kMaxPhi)
      phi = -kMaxPhi;
  }

  void handleZoom(float dy) {
    constexpr float kZoomFactor = 0.1f;
    radius *= (1.0f - dy * kZoomFactor);
    constexpr float kMinRadius = 0.1f;
    if (radius < kMinRadius)
      radius = kMinRadius;
  }

  void handlePan(float dx, float dy) {
    constexpr float kPanSpeed = 0.001f;
    float3 cam_eye = eye();
    float3 forward = normalize(lookAt - cam_eye);
    float3 right = normalize(cross(forward, make_float3(0.0f, 0.0f, 1.0f)));
    float3 up = normalize(cross(right, forward));

    lookAt = lookAt - right * (dx * kPanSpeed * radius) -
             up * (dy * kPanSpeed * radius);
  }

  float2 lastPos = make_float2(0.0f, 0.0f);
  bool leftMouseDragging = false;
  bool rightMouseDragging = false;

  void onMouseButton(int button, int action) {
    if (button == 0)
      leftMouseDragging = (action == 1);
    else if (button == 1)
      rightMouseDragging = (action == 1);
  }

  void onCursorPos(double xpos, double ypos) {
    float2 current =
        make_float2(static_cast<float>(xpos), static_cast<float>(ypos));
    float2 delta = current - lastPos;
    lastPos = current;

    if (leftMouseDragging)
      handleOrbit(delta.x, delta.y);
    else if (rightMouseDragging)
      handlePan(delta.x, -delta.y);
  }

  void onScroll(double yoffset) { handleZoom(static_cast<float>(yoffset)); }
};
