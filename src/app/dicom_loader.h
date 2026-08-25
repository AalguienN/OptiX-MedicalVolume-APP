#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct DicomSlice
{
    std::vector<uint16_t> pixelData;
    int rows               = 0;
    int columns            = 0;
    float pixelSpacing[2]  = {0.0f, 0.0f};
    float imagePosition[3] = {0.0f, 0.0f, 0.0f};
    float imageOrientation[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float rescaleSlope     = 1.0f;
    float rescaleIntercept = 0.0f;
    int bitsAllocated      = 16;
    int bitsStored         = 16;
    int pixelRepresentation = 0;
    int instanceNumber     = 0;
    std::string modality;
};

struct DicomSeries
{
    std::vector<DicomSlice> slices;
    std::string modality;
};

DicomSeries loadDicomSeries(const std::string& seriesPath);
