#include "dicom_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

void skipElement(std::ifstream& f, uint16_t endElem, bool explicitVR)
{
    while (true)
    {
        char hdr[4];
        f.read(hdr, 4);
        if (f.gcount() < 4) return;

        uint16_t g, e;
        std::memcpy(&g, hdr, 2);
        std::memcpy(&e, hdr + 2, 2);

        if (g == 0xFFFE)
        {
            uint32_t len = 0;
            f.read(reinterpret_cast<char*>(&len), 4);
            if (e == endElem) return;
            if (e == 0xE000)
            {
                if (len == 0xFFFFFFFF)
                    skipElement(f, 0xE00D, explicitVR);
                else if (len > 0)
                    f.seekg(len, std::ios::cur);
            }
        }
        else
        {
            uint32_t len = 0;
            if (explicitVR)
            {
                char vrRaw[2];
                f.read(vrRaw, 2);
                if (std::memchr("OBOWOFSQUCUNURUT", vrRaw[0], 16) &&
                    std::memchr("BFWQCNRT", vrRaw[1], 8) &&
                    ((vrRaw[0] == 'O' && (vrRaw[1] == 'B' || vrRaw[1] == 'W' || vrRaw[1] == 'F')) ||
                     (vrRaw[0] == 'S' && vrRaw[1] == 'Q') ||
                     (vrRaw[0] == 'U' && (vrRaw[1] == 'C' || vrRaw[1] == 'N')) ||
                     (vrRaw[0] == 'U' && vrRaw[1] == 'R') ||
                     (vrRaw[0] == 'U' && vrRaw[1] == 'T')))
                {
                    f.seekg(2, std::ios::cur);
                    f.read(reinterpret_cast<char*>(&len), 4);
                }
                else
                {
                    uint16_t len16 = 0;
                    f.read(reinterpret_cast<char*>(&len16), 2);
                    len = len16;
                }
            }
            else
            {
                // Implicit VR Little Endian: every element has a 32-bit length.
                f.read(reinterpret_cast<char*>(&len), 4);
            }

            if (len == 0xFFFFFFFF)
                skipElement(f, 0xE0DD, explicitVR);
            else if (len > 0 && len < 0x7FFFFFFF)
                f.seekg(len, std::ios::cur);
        }
    }
}

bool isSpecialVR(const char* vr)
{
    return (vr[0] == 'O' && (vr[1] == 'B' || vr[1] == 'W' || vr[1] == 'F')) ||
           (vr[0] == 'S' && vr[1] == 'Q') ||
           (vr[0] == 'U' && (vr[1] == 'C' || vr[1] == 'N')) ||
           (vr[0] == 'U' && vr[1] == 'R') ||
           (vr[0] == 'U' && vr[1] == 'T');
}

float parseDS(const std::string& s)
{
    if (s.empty()) return 0.0f;
    return static_cast<float>(std::strtod(s.c_str(), nullptr));
}

int parseInstanceNumber(const std::string& s)
{
    try
    {
        return std::stoi(s);
    }
    catch (const std::exception&)
    {
        return 0;
    }
}

}  // namespace

DicomSeries loadDicomSeries(const std::string& seriesPath)
{
    std::vector<std::string> dcmFiles;

    namespace fs = std::filesystem;
    fs::path dirPath(seriesPath);
    if (!fs::is_directory(dirPath))
        throw std::runtime_error("Not a directory: " + seriesPath);

    for (const auto& entry : fs::directory_iterator(dirPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".dcm")
            dcmFiles.push_back(entry.path().string());
    }

    if (dcmFiles.empty())
        throw std::runtime_error("No .dcm files found in: " + seriesPath);

    std::sort(dcmFiles.begin(), dcmFiles.end());

    DicomSeries series;

    for (const auto& filePath : dcmFiles)
    {
        std::ifstream f(filePath, std::ios::binary);
        if (!f) continue;

        char preamble[128];
        f.read(preamble, 128);

        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "DICM", 4) != 0) continue;

        DicomSlice slice;

        bool foundPixelData = false;

        // The File Meta Group (group 0x0002) is always Explicit VR even in
        // Implicit VR Little Endian datasets. Scan it first to learn the
        // Transfer Syntax UID (0002,0010) and decide how to parse the dataset.
        bool explicitVR = true;
        while (f.good())
        {
            char tagBuf[4];
            f.read(tagBuf, 4);
            if (f.gcount() < 4) break;

            uint16_t group, elem;
            std::memcpy(&group, tagBuf, 2);
            std::memcpy(&elem, tagBuf + 2, 2);

            if (group != 0x0002)
            {
                // Reached the dataset proper; restore the tag for dataset loop.
                f.seekg(-4, std::ios::cur);
                break;
            }

            char vrRaw[2];
            f.read(vrRaw, 2);
            char vr[3] = {vrRaw[0], vrRaw[1], '\0'};

            uint32_t dataLen = 0;
            if (isSpecialVR(vr))
            {
                char reserved[2];
                f.read(reserved, 2);
                f.read(reinterpret_cast<char*>(&dataLen), 4);
            }
            else
            {
                uint16_t len16 = 0;
                f.read(reinterpret_cast<char*>(&len16), 2);
                dataLen = len16;
            }

            if (elem == 0x0010 && dataLen > 0 && dataLen < 0x7FFFFFFF)
            {
                std::vector<char> buf(dataLen);
                f.read(buf.data(), dataLen);
                std::string uid(buf.data(), dataLen);
                while (!uid.empty() && (uid.back() == '\0' || uid.back() == ' '))
                    uid.pop_back();
                // Implicit VR Little Endian transfer syntax.
                explicitVR = (uid != "1.2.840.10008.1.2");
            }
            else if (dataLen > 0 && dataLen < 0x7FFFFFFF)
            {
                f.seekg(dataLen, std::ios::cur);
            }
        }

        while (f.good())
        {
            char tagBuf[4];
            f.read(tagBuf, 4);
            if (f.gcount() < 4) break;

            uint16_t group, elem;
            std::memcpy(&group, tagBuf, 2);
            std::memcpy(&elem, tagBuf + 2, 2);

            if (group == 0x7FE0 && elem == 0x0010)
            {
                uint32_t pixelLen = 0;
                if (explicitVR)
                {
                    char vrCheck[2];
                    f.read(vrCheck, 2);
                    char reserved[2];
                    f.read(reserved, 2);
                    f.read(reinterpret_cast<char*>(&pixelLen), 4);
                }
                else
                {
                    f.read(reinterpret_cast<char*>(&pixelLen), 4);
                }

                if (pixelLen > 0 && pixelLen < 0x7FFFFFFF)
                {
                    size_t numPixels = pixelLen / sizeof(uint16_t);
                    size_t expectedPixels = static_cast<size_t>(slice.rows) * slice.columns;
                    if (expectedPixels > 0 && numPixels != expectedPixels)
                    {
                        std::cerr << "Warning: pixel count mismatch in " << filePath
                                  << " (got " << numPixels << ", expected " << expectedPixels
                                  << " = " << slice.rows << "x" << slice.columns << ")\n";
                    }
                    slice.pixelData.resize(numPixels);
                    f.read(reinterpret_cast<char*>(slice.pixelData.data()),
                           static_cast<std::streamsize>(pixelLen));
                }
                foundPixelData = true;
                break;
            }

            if (group == 0xFFFE)
            {
                uint32_t len = 0;
                f.read(reinterpret_cast<char*>(&len), 4);
                if (len > 0 && len < 0xFFFFFFFF)
                    f.seekg(len, std::ios::cur);
                continue;
            }

            uint32_t dataLen = 0;
            if (explicitVR)
            {
                char vrRaw[2];
                f.read(vrRaw, 2);
                char vr[3] = {vrRaw[0], vrRaw[1], '\0'};
                if (isSpecialVR(vr))
                {
                    char reserved[2];
                    f.read(reserved, 2);
                    f.read(reinterpret_cast<char*>(&dataLen), 4);
                }
                else
                {
                    uint16_t len16 = 0;
                    f.read(reinterpret_cast<char*>(&len16), 2);
                    dataLen = len16;
                }
            }
            else
            {
                // Implicit VR Little Endian: every element has a 32-bit length.
                f.read(reinterpret_cast<char*>(&dataLen), 4);
            }

            if (dataLen == 0xFFFFFFFF)
            {
                skipElement(f, 0xE0DD, explicitVR);
                continue;
            }

            if (dataLen == 0) continue;

            if (dataLen >= 0x7FFFFFFF) continue;

            auto readString = [&]() -> std::string
            {
                std::vector<char> buf(dataLen);
                f.read(buf.data(), dataLen);
                std::string s(buf.data(), dataLen);
                while (!s.empty() && (s.back() == '\0' || s.back() == ' '))
                    s.pop_back();
                return s;
            };

            auto readUint16 = [&]() -> uint16_t
            {
                uint16_t v = 0;
                f.read(reinterpret_cast<char*>(&v), 2);
                if (dataLen > 2)
                    f.seekg(dataLen - 2, std::ios::cur);
                return v;
            };

            if (group == 0x0028)
            {
                if (elem == 0x0010) { slice.rows = static_cast<int>(readUint16()); }
                else if (elem == 0x0011) { slice.columns = static_cast<int>(readUint16()); }
                else if (elem == 0x0100) { slice.bitsAllocated = static_cast<int>(readUint16()); }
                else if (elem == 0x0101) { slice.bitsStored = static_cast<int>(readUint16()); }
                else if (elem == 0x0102)
                {
                    uint16_t hb = readUint16();
                    (void)hb;
                }
                else if (elem == 0x0103) { slice.pixelRepresentation = static_cast<int>(readUint16()); }
                else if (elem == 0x0030)
                {
                    std::string s = readString();
                    size_t pos = s.find('\\');
                    if (pos != std::string::npos)
                    {
                        slice.pixelSpacing[0] = parseDS(s.substr(0, pos));
                        slice.pixelSpacing[1] = parseDS(s.substr(pos + 1));
                    }
                }
                else if (elem == 0x1052)
                    slice.rescaleIntercept = parseDS(readString());
                else if (elem == 0x1053)
                    slice.rescaleSlope = parseDS(readString());
                else
                    f.seekg(dataLen, std::ios::cur);
            }
            else if (group == 0x0020)
            {
                if (elem == 0x0032)
                {
                    std::string s = readString();
                    size_t p1 = s.find('\\');
                    size_t p2 = (p1 != std::string::npos) ? s.find('\\', p1 + 1) : std::string::npos;
                    if (p1 != std::string::npos && p2 != std::string::npos)
                    {
                        slice.imagePosition[0] = parseDS(s.substr(0, p1));
                        slice.imagePosition[1] = parseDS(s.substr(p1 + 1, p2 - p1 - 1));
                        slice.imagePosition[2] = parseDS(s.substr(p2 + 1));
                    }
                }
                else if (elem == 0x0037)
                {
                    std::string s = readString();
                    float vals[6] = {};
                    size_t pos = 0;
                    for (int i = 0; i < 6; ++i)
                    {
                        size_t next = s.find('\\', pos);
                        std::string token = (next != std::string::npos) ? s.substr(pos, next - pos) : s.substr(pos);
                        vals[i] = parseDS(token);
                        if (next == std::string::npos) break;
                        pos = next + 1;
                    }
                    std::memcpy(slice.imageOrientation, vals, sizeof(vals));
                }
                else if (elem == 0x0013)
                    slice.instanceNumber = parseInstanceNumber(readString());
                else
                    f.seekg(dataLen, std::ios::cur);
            }
            else if (group == 0x0008)
            {
                if (elem == 0x0060)
                    slice.modality = readString();
                else
                    f.seekg(dataLen, std::ios::cur);
            }
            else
            {
                f.seekg(dataLen, std::ios::cur);
            }
        }

        if (foundPixelData)
            series.slices.push_back(std::move(slice));
    }

    if (series.slices.empty())
        throw std::runtime_error("No valid DICOM slices loaded from: " + seriesPath);

    series.modality = series.slices[0].modality;

    const auto& ref = series.slices[0];
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

    std::sort(series.slices.begin(), series.slices.end(),
              [&](const DicomSlice& a, const DicomSlice& b)
              {
                  float da = normal[0] * a.imagePosition[0] +
                             normal[1] * a.imagePosition[1] +
                             normal[2] * a.imagePosition[2];
                  float db = normal[0] * b.imagePosition[0] +
                             normal[1] * b.imagePosition[1] +
                             normal[2] * b.imagePosition[2];
                  if (da != db) return da < db;
                  return a.instanceNumber < b.instanceNumber;
              });

    std::cout << "Loaded " << series.slices.size() << " slices from " << seriesPath
              << " (modality=" << series.modality << ", "
              << series.slices[0].rows << "x" << series.slices[0].columns << ")\n";

    return series;
}
