#include "volume.h"

#include "check_macros.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

void Volume::buildFromSeries(const DicomSeries& series)
{
    if (series.slices.empty())
        throw std::runtime_error("Cannot build volume from empty series");

    const auto& ref = series.slices[0];
    dimX = ref.columns;
    dimY = ref.rows;
    modality = series.modality;

    float normal[3];
    {
        float rX = ref.imageOrientation[0];
        float rY = ref.imageOrientation[1];
        float rZ = ref.imageOrientation[2];
        float cX = ref.imageOrientation[3];
        float cY = ref.imageOrientation[4];
        float cZ = ref.imageOrientation[5];
        normal[0] = rY * cZ - rZ * cY;
        normal[1] = rZ * cX - rX * cZ;
        normal[2] = rX * cY - rY * cX;
    }

    std::vector<size_t> sortedIdx(series.slices.size());
    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [&](size_t a, size_t b)
              {
                  float da = normal[0] * series.slices[a].imagePosition[0] +
                             normal[1] * series.slices[a].imagePosition[1] +
                             normal[2] * series.slices[a].imagePosition[2];
                  float db = normal[0] * series.slices[b].imagePosition[0] +
                             normal[1] * series.slices[b].imagePosition[1] +
                             normal[2] * series.slices[b].imagePosition[2];
                  return da < db;
              });

    dimZ = static_cast<int>(series.slices.size());

    spacingX = ref.pixelSpacing[1];
    spacingY = ref.pixelSpacing[0];

    {
        float pos0[3], pos1[3];
        std::memcpy(pos0, series.slices[sortedIdx[0]].imagePosition, sizeof(float) * 3);
        std::memcpy(pos1, series.slices[sortedIdx[1]].imagePosition, sizeof(float) * 3);
        float dz = normal[0] * (pos1[0] - pos0[0]) +
                   normal[1] * (pos1[1] - pos0[1]) +
                   normal[2] * (pos1[2] - pos0[2]);
        spacingZ = std::fabs(dz);
    }

    originX = ref.imagePosition[0];
    originY = ref.imagePosition[1];
    originZ = ref.imagePosition[2];

    data.resize(static_cast<size_t>(dimX) * dimY * dimZ);

    float slope     = ref.rescaleSlope;
    float intercept = ref.rescaleIntercept;
    bool  isCT      = (modality == "CT");

    for (int z = 0; z < dimZ; ++z)
    {
        const auto& slice = series.slices[sortedIdx[z]];
        size_t sliceSize = static_cast<size_t>(dimX) * dimY;

        float* dst = data.data() + static_cast<size_t>(z) * dimX * dimY;

        if (slice.pixelData.empty() || slice.pixelData.size() < sliceSize)
        {
            std::cerr << "Warning: slice " << z << " has empty/incomplete pixel data, filling with zeros\n";
            std::memset(dst, 0, sizeof(float) * sliceSize);
            continue;
        }

        const uint16_t* pixels = slice.pixelData.data();

        if (slice.pixelRepresentation == 1 && slice.bitsAllocated <= 16)
        {
            for (size_t i = 0; i < sliceSize; ++i)
            {
                int16_t raw = static_cast<int16_t>(pixels[i]);
                float val = static_cast<float>(raw);
                if (isCT)
                    dst[i] = val * slope + intercept;
                else
                    dst[i] = val;
            }
        }
        else
        {
            for (size_t i = 0; i < sliceSize; ++i)
            {
                float val = static_cast<float>(pixels[i]);
                if (isCT)
                    dst[i] = val * slope + intercept;
                else
                    dst[i] = val;
            }
        }
    }

    std::cout << "Volume built: " << dimX << "x" << dimY << "x" << dimZ
              << " spacing=(" << spacingX << "," << spacingY << "," << spacingZ << ")"
              << " origin=(" << originX << "," << originY << "," << originZ << ")"
              << " modality=" << modality << "\n";
}

void Volume::uploadToDevice(cudaArray** d_array, cudaTextureObject_t* texObj) const
{
    cudaExtent extent = make_cudaExtent(dimX, dimY, dimZ);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray(d_array, &channelDesc, extent));

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr(
        const_cast<float*>(data.data()),
        dimX * sizeof(float),
        dimX,
        dimY);
    copyParams.dstArray = *d_array;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = *d_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModeLinear;
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(texObj, &resDesc, &texDesc, nullptr));

    std::cout << "Volume uploaded to GPU: " << dimX << "x" << dimY << "x" << dimZ
              << " (" << data.size() * sizeof(float) / (1024.0 * 1024.0) << " MB)\n";
}

void Volume::destroyDevice(cudaArray* d_array, cudaTextureObject_t texObj) const
{
    if (texObj)
        cudaDestroyTextureObject(texObj);
    if (d_array)
        cudaFreeArray(d_array);
}
