#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#define SUTIL_HOSTDEVICE __host__ __device__
#define SUTIL_INLINE __forceinline__
#else
#define SUTIL_HOSTDEVICE
#define SUTIL_INLINE inline
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

SUTIL_INLINE SUTIL_HOSTDEVICE float clamp(const float f, const float a, const float b)
{
    return fmaxf(a, fminf(f, b));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 make_float3(const float s)
{
    return make_float3(s, s, s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 make_float3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 make_float2(const float s)
{
    return make_float2(s, s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float4 make_float4(const float s)
{
    return make_float4(s, s, s, s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float4 make_float4(const float3& a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator+(const float3& a, const float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator+(const float a, const float3& b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator*(const float3& a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator*(const float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator/(const float3& a, const float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator*(const float s, const float2& a)
{
    return make_float2(a.x * s, a.y * s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator*(const float2& a, const float s)
{
    return make_float2(a.x * s, a.y * s);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator-(const float2& a, const float b)
{
    return make_float2(a.x - b, a.y - b);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float2 operator/(const float2& a, const float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 clamp(const float3& v, const float a, const float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

SUTIL_INLINE SUTIL_HOSTDEVICE float length(const float3& v)
{
    return sqrtf(dot(v, v));
}

SUTIL_INLINE SUTIL_HOSTDEVICE float3 normalize(const float3& v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}
