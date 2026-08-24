#pragma once

#include "vec_math.h"
#include <vector_functions.h>

class Camera
{
public:
    float theta  = 0.0f;
    float phi    = 0.3f;
    float radius = 4.0f;
    float fovY   = 45.0f;
    float aspect = 1.0f;

    float3 eye() const
    {
        return make_float3(
            radius * cosf(phi) * sinf(theta),
            radius * sinf(phi),
            radius * cosf(phi) * cosf(theta));
    }

    void uvw(float3& U, float3& V, float3& W) const
    {
        float3 cam_eye    = eye();
        float3 cam_lookat = make_float3(0, 0, 0);
        float3 cam_up     = make_float3(0, 1, 0);

        float3 w    = cam_lookat - cam_eye;
        float  wlen = length(w);
        W = w;
        U = normalize(cross(w, cam_up));
        V = normalize(cross(U, w));

        float vlen = wlen * tanf(0.5f * fovY * static_cast<float>(M_PIf) / 180.0f);
        V = V * vlen;
        float ulen = vlen * aspect;
        U = U * ulen;
    }
};
