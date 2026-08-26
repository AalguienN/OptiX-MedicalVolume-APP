# DICOM Loader Improvements — Robustness, Portability and Correctness

Documents the four changes applied to `dicom_loader.cpp` to address reliability, cross-platform
support, and geometric correctness of the DICOM data pipeline (Section 5.4).

## Motivation

The original parser was built as a minimal, self-contained DICOM reader for a controlled input
set (uncompressed, little-endian, 16-bit TCIA data). While functional for that scope, a review
identified issues that could cause silent data corruption or platform lock-in:

- Slice ordering relied on `InstanceNumber`, which is not a reliable proxy for physical position.
- `std::stoi()` could throw on malformed metadata, aborting the entire series load.
- Pixel data size was not validated against declared image dimensions.
- File discovery used `popen("ls ...")`, restricting the loader to POSIX systems.

## Changes

### 1. Spatial slice ordering (#7)

**Before:** Slices were sorted by `InstanceNumber` (tag `0020,0013`).

**After:** Slices are sorted by projecting `ImagePositionPatient` (tag `0020,0032`) onto the
slice normal, computed as the cross product of the row and column direction cosines from
`ImageOrientationPatient` (tag `0020,0037`). `InstanceNumber` is used as a tiebreaker for
slices at the same spatial position.

```cpp
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
```

**Rationale:** `InstanceNumber` is assigned by the scanner and does not necessarily reflect
physical slice position. Re-scans, retractions, or non-sequential numbering can produce
incorrect volume geometry. Projecting onto the slice normal guarantees spatially correct
ordering regardless of DICOM metadata conventions.

**Files changed:** `dicom_loader.cpp`

### 2. Safe InstanceNumber parsing (#11)

**Before:** `std::stoi(readString())` was called directly, with no exception handling.

**After:** A `parseInstanceNumber()` helper wraps the conversion in a try-catch block,
returning 0 on failure instead of aborting the series load.

```cpp
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
```

**Rationale:** A single malformed DICOM value should not prevent loading of the entire series.
Returning 0 allows the slice to be included; spatial sorting will place it correctly relative
to other slices.

**Files changed:** `dicom_loader.cpp`

### 3. Pixel count validation (#12)

**Before:** Pixel data was read into a buffer sized purely from the declared pixel data length,
with no check against the declared image dimensions.

**After:** After reading pixel data, the loader compares the actual pixel count against the
expected value (`Rows x Columns`). A warning is emitted on mismatch, but loading continues.

```cpp
size_t numPixels = pixelLen / sizeof(uint16_t);
size_t expectedPixels = static_cast<size_t>(slice.rows) * slice.columns;
if (expectedPixels > 0 && numPixels != expectedPixels)
{
    std::cerr << "Warning: pixel count mismatch in " << filePath
              << " (got " << numPixels << ", expected " << expectedPixels
              << " = " << slice.rows << "x" << slice.columns << ")\n";
}
```

**Rationale:** A mismatch between declared dimensions and actual pixel data length indicates
corruption or an unsupported pixel format. Emitting a warning makes the problem visible
without silently producing garbage data in the volume.

**Files changed:** `dicom_loader.cpp`

### 4. Cross-platform file discovery

**Before:** File listing was performed via a POSIX shell command:

```cpp
std::string cmd = "ls \"" + seriesPath + "\"/*.dcm 2>/dev/null";
FILE* pipe = popen(cmd.c_str(), "r");
```

**After:** Uses `std::filesystem::directory_iterator` from C++17:

```cpp
namespace fs = std::filesystem;
fs::path dirPath(seriesPath);
if (!fs::is_directory(dirPath))
    throw std::runtime_error("Not a directory: " + seriesPath);

for (const auto& entry : fs::directory_iterator(dirPath))
{
    if (entry.is_regular_file() && entry.path().extension() == ".dcm")
        dcmFiles.push_back(entry.path().string());
}
```

**Rationale:** The `popen("ls ...")` approach is POSIX-only and unavailable on Windows.
`std::filesystem` is part of the C++17 standard and works on Linux, macOS, and Windows
without platform-specific `#ifdef` blocks. This also eliminates shell injection risks from
paths containing special characters.

**Files changed:** `dicom_loader.cpp`, `CMakeLists.txt` (C++ standard upgraded from 11 to 17)

## Build configuration change

`CMAKE_CXX_STANDARD` was upgraded from `11` to `17` in `CMakeLists.txt` to enable
`std::filesystem`. This is compatible with the existing CUDA 13.1 / GCC 13.3 toolchain.

## Verification

- Full rebuild through `mkbuild.sh` (CMake Release, `sm_120`) completes without errors or
  compiler warnings.
- All previously validated CT and MR datasets continue to load and render correctly.
- The spatial sort in `volume.cpp` (lines 34–46) is now redundant with the loader's sort but
  is retained as defense-in-depth.

## Files changed

| File | Change |
|---|---|
| `CMakeLists.txt` | `CMAKE_CXX_STANDARD` 11 → 17 |
| `src/app/dicom_loader.cpp` | Spatial sort, safe InstanceNumber parsing, pixel validation, `std::filesystem` |
