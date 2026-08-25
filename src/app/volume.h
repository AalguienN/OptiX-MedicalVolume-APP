#pragma once

#include "dicom_loader.h"

#include <cuda_runtime.h>
#include <vector>

struct Volume
{
    std::vector<float> data;
    int dimX = 0;
    int dimY = 0;
    int dimZ = 0;
    float spacingX = 0.0f;
    float spacingY = 0.0f;
    float spacingZ = 0.0f;
    float originX  = 0.0f;
    float originY  = 0.0f;
    float originZ  = 0.0f;
    std::string modality;

    float maxX() const { return originX + dimX * spacingX; }
    float maxY() const { return originY + dimY * spacingY; }
    float maxZ() const { return originZ + dimZ * spacingZ; }

    void buildFromSeries(const DicomSeries& series);
    void uploadToDevice(cudaArray** d_array, cudaTextureObject_t* texObj) const;
    void destroyDevice(cudaArray* d_array, cudaTextureObject_t texObj) const;
};
