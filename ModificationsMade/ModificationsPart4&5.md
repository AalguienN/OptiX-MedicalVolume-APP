# Data Pipeline & Dense Baseline Render — Implementation Notes

Source material for the *Development → Data Pipeline Implementation* (`tfm.tex:950`) and
*Dense Baseline Render Implementation* (`tfm.tex:956`) sections of the thesis. Covers the commit
range that introduced DICOM loading, volume reconstruction, GPU upload, transfer functions and the
volume ray-casting raygen program.

## Objective

Implement two deliverables:

1. **Data Pipeline (Section 5.4):** A self-contained DICOM parser that reads the Explicit VR Little
   Endian files in the TCIA manifests, reconstructs a regular 3D scalar grid from the slice stack,
   converts raw pixel values to Hounsfield Units (CT) or preserves native intensities (MR), and
   uploads the result to GPU as a 3D CUDA texture.
2. **Dense Baseline Renderer (Section 5.5):** A fixed-step volume ray-casting program implemented
   entirely within the OptiX raygen stage, using a manual ray–AABB intersection against the volume
   bounding box, trilinear texture sampling, and front-to-back compositing with early ray
   termination.

Both deliverables are validated with real DICOM series from the TCIA dataset collection.

## Starting point

After Section 5.3 (`ApplicationInfrastructure.md`), the application rendered a shaded cube via
`PlaceholderScene`, with a fixed orbit camera and no volume data. The `Params` struct carried a
single `OptixTraversableHandle`, a raygen SBT record with camera basis vectors, and a `uchar4*`
interop buffer. There was no DICOM parsing, no volume data path, and the raygen program computed a
camera ray and returned a miss-gradient or a magenta closest-hit colour.

## Implementation

### New modules

| Module | Responsibility |
|---|---|
| `dicom_loader.{h,cpp}` | Self-contained DICOM parser. Reads all required metadata tags (Rows, Columns, BitsAllocated, PixelSpacing, ImagePositionPatient, ImageOrientationPatient, RescaleSlope, RescaleIntercept, Modality, InstanceNumber) and the raw pixel data (`7FE0,0010`) from Explicit VR Little Endian files. Handles undefined-length sequences, encapsulated pixel data VRs (OB, OW, UN), and skips item delimiters without external library dependencies. Returns a `DicomSeries` of sorted slices. |
| `volume.{h,cpp}` | Reconstructs a regular 3D scalar grid from a `DicomSeries`. Sorts slices by projecting `ImagePositionPatient` onto the slice-normal (computed from `ImageOrientationPatient`). Converts raw pixel values to Hounsfield Units for CT via `RescaleSlope`/`RescaleIntercept`, preserves native values for MR. Uploads the grid to GPU as a 3D `cudaArray` with trilinear `cudaTextureObject_t`. |
| `transfer_function.h` | Default transfer-function LUT (2048 entries). Provides `buildDefaultCT()` (air-transparent → soft-tissue semi-opaque → bone opaque white, HU range −1000 to +1000) and `buildDefaultMR()` (intensity-based mapping, range 0–4096). The LUT is uploaded as a raw `float4*` device pointer (indexed directly, not through a texture object). Alpha values tuned for visibility of internal structure. |
| `placeholder_scene.{h,cpp}` | Retained from Section 5.3 as a dummy GAS for the `Params.handle` field. The raygen program does not call `optixTrace`; it performs all volume traversal internally, so the GAS is never traversed. |

### Modified modules

| Module | Changes |
|---|---|
| `shared_device.h` | Extended `Params` with `cudaTextureObject_t volumeTex`, `float4* tfData`, `int3 volumeDims`, `float3 volumeSpacing`, `float3 volumeOrigin`, `float3 volumeMax`, `float scalarMin`, `float scalarMax`. These carry the volume geometry, textures, and scalar-to-TF mapping range to the device. The TF data uses a raw device pointer rather than a texture object (see *Bug fixes*). |
| `device/device_programs.cu` | Rewritten for volume ray casting. The raygen program computes a ray from camera UVW, performs manual ray–AABB intersection against the volume bounding box, then marches in fixed steps (half the minimum voxel spacing), sampling the 3D texture with trilinear interpolation, looking up colour/opacity in the 1D TF texture, and accumulating via front-to-back compositing. Miss and closest-hit are retained as stubs (background gradient / magenta) for pipeline validity but are not invoked in the volume path. |
| `app_main.cpp` | Now accepts a CLI argument `<path-to-dicom-series>`. Loads the DICOM series, builds the volume, creates the transfer function (CT or MR), uploads textures to GPU, constructs the `Params` struct with all volume metadata, and runs the render loop with the updated raygen SBT record. Camera radius is derived from the volume bounding-box extent. |
| `CMakeLists.txt` | Replaced `placeholder_scene.cpp` target references with `dicom_loader.cpp`, `volume.cpp`, and `volume_scene.cpp` (the latter was later reverted back to `placeholder_scene.cpp` after the AABB-based GAS approach did not work as expected at runtime; `volume_scene.{h,cpp}` remain in the source tree but are not compiled). |

### DICOM parser design

The parser is entirely self-contained — no dependency on DCMTK or GDCM — because the development
environment lacked `sudo` access for `apt install`. It handles:

- **Explicit VR Little Endian** transfer syntax (UID `1.2.840.10008.1.2.1`)
- **Undefined-length sequences** (`0xFFFFFFFF`) and item delimiters in group `0xFFFE`
- **Special VRs** (OB, OW, SQ, UC, UN, UR, UT): 2-byte reserved + 4-byte length
- **Standard VRs** (CS, DS, IS, LO, SH, etc.): 2-byte length
- **Pixel data** (`7FE0,0010`): Explicit VR detection, reads VR header correctly before length
- **Slice sorting**: by instance number (group `0020`, element `0013`)

The critical subtlety discovered during implementation: the pixel data tag `(7FE0,0010)` uses
Explicit VR, so after the 4-byte tag come the VR bytes (`"OW"`), 2 reserved bytes, and then the
4-byte length. Earlier versions that skipped the VR header read garbage as the pixel length, causing
empty pixel data and a segfault during volume reconstruction.

### Volume reconstruction

Slice stacking order is determined by the slice normal, computed as the cross-product of
`ImageOrientationPatient` row and column direction cosines. Each slice's position along the normal
is projected, and slices are sorted ascending. The slice-to-slice spacing is computed from the
signed distance between the first two sorted slices along the normal.

Voxel coordinates in the raygen program use the same origin and spacing, so the 3D texture's
normalized coordinates map directly to patient space.

### GPU data structures

| Resource | Format | Filter | Normalized coords |
|---|---|---|---|
| Volume | `cudaArray` (3D, float) | Trilinear (`cudaFilterModeLinear`) | Yes (0–1 range) |
| Transfer function | `float4*` device pointer | Point (integer index) | N/A |

The 3D texture enables the hardware trilinear interpolator to perform 8-tap interpolation in a
single `tex3D` call, which is critical for visual quality in the baseline fixed-step approach.

### Raygen program — volume ray casting

The algorithm in `device_programs.cu`:

1. Compute primary ray origin and direction from the camera UVW basis and pixel coordinates.
2. Perform slab-test ray–AABB intersection against the volume bounding box (`volumeOrigin` to
   `volumeMax`).
3. If intersected, march from `tmin` in fixed steps of `min(spacingX, spacingY, spacingZ) × 0.5`,
   capped at 4096 steps.
4. At each step, convert the world-space sample position to normalized texture coordinates and
   sample the 3D volume texture via `tex3D<float>`.
5. Map the scalar value to a TF LUT index using a generic linear mapping
   `(scalar − scalarMin) / (scalarMax − scalarMin) × 2047` (modality-aware: CT uses −1000…+1000,
   MR uses 0…4096), and look up the colour/opacity via direct device-pointer access (`tfData[tfIdx]`).
6. Apply front-to-back compositing: `opacityFactor = (1 − α_accum) × α_sample`, accumulate
   premultiplied colour.
7. Terminate early when `α_accum ≥ 0.99`.
8. Write the composited colour to the interop buffer via `make_color`.

The step size is half the minimum voxel spacing to satisfy the Nyquist criterion for the dense
baseline. The max-steps cap prevents pathological rays (very large volumes or very small voxels) from
stalling the GPU.

### Compliance with design constraints

- **Common rendering pipeline** (principle 1): raygen, miss and compositing are identical to
  Section 5.3; the volume-specific logic is entirely within the same raygen program, with the
  intersection and sampling stages parameterized through `Params`.
- **Interchangeable representations** (principle 2): the volume textures and metadata are passed
  through `Params`; swapping in a sparse representation will replace the raygen marching logic while
  keeping the camera, compositing, display, and loop unchanged.
- **GPU-oriented execution** (principle 3): the volume resides in GPU memory as a 3D texture; the
  only host→device transfer per frame is the small `Params` struct and the raygen SBT record.

## Verification

- Full rebuild through `mkbuild.sh` (CMake Release, `sm_120`) completes without errors or
  compiler warnings on the CUDA 13.1 / OptiX 9.1 / Ubuntu 24.04 stack.
- Tested with TCIA manifest `manifest-1787683033169_DICOM/series_5` (MR, 112 slices, 512×512):
  the parser loads all slices, the volume is reconstructed as 512×512×112 with spacing
  (0.430, 0.430, 0.400), uploaded to GPU (112 MB), and the render loop starts without CUDA errors.
- Tested with TCIA manifest `manifest-1787683033169_DICOM/series_3` (CT, 176 slices): HU
  conversion applies `RescaleIntercept = −1024` and the bone-window transfer function is exercised.
- After all five bug fixes, CT centre pixel confirms `accumA = 0.9925` with accumulated colour
  `(0.979, 0.695, 0.527)` — the volume renders correctly with visible internal structure.
- MR dataset (640×640×104, range [0, 697]) now renders with correct per-intensity colouring
  through the full LUT range, instead of a uniform block.

## Explicitly out of scope at this stage

- Sparse rendering strategies (Sections 5.6+), metrics instrumentation (Section 5.7)
- Interactive transfer-function editing (fixed defaults only)
- Shading (no normals, no lighting — pure emission-absorption compositing)
- Multi-volume or multi-modality compositing

## Bug fixes applied during implementation

### 1. `normalizedCoords = 0` in volume texture

The `cudaTextureDesc` for the 3D volume texture had `normalizedCoords = 0`, which caused the
texture coordinates to be interpreted as raw texel indices (0–dim range) instead of normalised
[0,1] values. Since the raygen computes normalised coordinates, the volume appeared empty. Fixed by
setting `normalizedCoords = 1`.

### 2. Camera looking at world origin instead of volume centre

The initial camera `lookAt` target was `(0, 0, 0)`, but the reconstructed volume was centred at
approximately `(19, −58, −26)` in patient space. This caused the camera to orbit around empty
space. Fixed by deriving the camera target from the volume's bounding-box centre.

### 3. Transfer function texture object returning zeros

The 1D transfer function was uploaded as a `cudaArray` via `cudaMemcpy2DToArray` and bound as a
`cudaTextureObject_t` with `cudaResourceTypeArray`. Despite the host-side LUT containing valid
values, `tex1D<float4>` in the raygen kernel returned `(0,0,0,0)` for all lookups. This was
confirmed via GPU printf: scalar values from the 3D volume texture were non-zero, but the TF
lookup produced zero alpha, resulting in a fully transparent (black) image after compositing.

The issue was reproduced with both `cudaArray` + `cudaMemcpy2DToArray` and `cudaMalloc` +
`cudaResourceTypeLinear` + `cudaFilterModeLinear`. Switching to a raw device pointer
(`float4*` uploaded via `cudaMalloc` + `cudaMemcpy`) with integer-index access in the kernel
immediately resolved the problem. The root cause is likely a CUDA texture-object interoperability
issue specific to the OptiX launch path or the `__constant__` parameter passing mechanism. The raw
pointer bypasses this entirely and has negligible performance difference for a 2048-entry LUT.

### 4. Solid opaque cube (TF alpha values too high)

After fixing Bug 3, the volume rendered as a solid opaque cube with no visible internal
structure. The default CT transfer function had alpha density values of 0.1–0.6 per mm for soft
tissue, which with a step size of ~0.34 mm produced per-step opacity of 0.03–0.20, reaching full
accumulation (α ≥ 0.99) in ~5–10 steps. Only the outermost few millimetres of tissue were
visible. Fixed by reducing alpha values across both CT and MR transfer functions by approximately
one order of magnitude, allowing rays to penetrate deep into the volume.

### 5. MR dataset renders as uniform block (scalar-to-TF mapping hardcoded for CT)

The kernel's TF index formula `(scalar + 1000) × 2047 / 2000` was designed for CT Hounsfield
Units (range −1000 to +1000). MR scalar values (range 0–697) were mapped to LUT indices
1023–1737, all falling in the uniform "else" branch of the MR TF. Fixed by adding
`scalarMin`/`scalarMax` to `Params`, set per-modality on the host (CT: −1000…+1000, MR:
0…4096), and using a generic linear mapping in the kernel.
