#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>

struct TransferFunctionEntry
{
    float value;
    float r, g, b;
    float a;
};

struct TransferFunction
{
    static constexpr int LUT_SIZE = 2048;
    float4 lut[LUT_SIZE];

    void buildDefaultCT()
    {
        for (int i = 0; i < LUT_SIZE; ++i)
        {
            float hu = static_cast<float>(i) * (2000.0f / LUT_SIZE) - 1000.0f;

            float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

            if (hu < -900.0f)
            {
                a = 0.0f;
            }
            else if (hu < -700.0f)
            {
                float t = (hu + 900.0f) / 200.0f;
                r = 0.2f * t;
                g = 0.2f * t;
                b = 0.3f * t;
                a = 0.002f * t;
            }
            else if (hu < -200.0f)
            {
                float t = (hu + 700.0f) / 500.0f;
                r = 0.2f + 0.3f * t;
                g = 0.2f + 0.1f * t;
                b = 0.3f + 0.2f * t;
                a = 0.002f + 0.008f * t;
            }
            else if (hu < 50.0f)
            {
                float t = (hu + 200.0f) / 250.0f;
                r = 0.5f + 0.3f * t;
                g = 0.3f + 0.2f * t;
                b = 0.5f + 0.1f * t;
                a = 0.01f + 0.04f * t;
            }
            else if (hu < 200.0f)
            {
                float t = (hu - 50.0f) / 150.0f;
                r = 0.8f + 0.2f * t;
                g = 0.5f + 0.2f * t;
                b = 0.6f - 0.1f * t;
                a = 0.05f + 0.1f * t;
            }
            else if (hu < 500.0f)
            {
                float t = (hu - 200.0f) / 300.0f;
                r = 1.0f;
                g = 0.7f + 0.3f * t;
                b = 0.5f + 0.5f * t;
                a = 0.15f + 0.3f * t;
            }
            else
            {
                r = 1.0f;
                g = 1.0f;
                b = 1.0f;
                a = 0.8f;
            }

            lut[i] = make_float4(r, g, b, a);
        }
    }

    void buildDefaultMR()
    {
        for (int i = 0; i < LUT_SIZE; ++i)
        {
            float val = static_cast<float>(i) * 4096.0f / LUT_SIZE;

            float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

            if (val < 100.0f)
            {
                a = 0.0f;
            }
            else if (val < 500.0f)
            {
                float t = (val - 100.0f) / 400.0f;
                r = 0.3f * t;
                g = 0.2f * t;
                b = 0.4f * t;
                a = 0.005f + 0.025f * t;
            }
            else if (val < 1500.0f)
            {
                float t = (val - 500.0f) / 1000.0f;
                r = 0.3f + 0.5f * t;
                g = 0.2f + 0.4f * t;
                b = 0.4f + 0.1f * t;
                a = 0.03f + 0.05f * t;
            }
            else
            {
                r = 0.8f;
                g = 0.7f;
                b = 0.5f;
                a = 0.15f;
            }

            lut[i] = make_float4(r, g, b, a);
        }
    }

    __host__ __device__ float4 sample(float value) const
    {
        float t = (value + 1000.0f) * (LUT_SIZE - 1) / 2000.0f;
        int idx = static_cast<int>(t);
        if (idx < 0) idx = 0;
        if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
        return lut[idx];
    }
};
