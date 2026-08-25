# Rendering Bug Fixes — Volume Raycaster Debugging

Documents the three critical bugs discovered and fixed during validation of the volume raycaster
(Sections 5.4–5.5) with real DICOM datasets.

## Bug 1: Transfer function texture object returning zeros

**Symptom:** Volume renders as completely black. GPU printf confirmed non-zero scalar values
sampled from the 3D volume texture, but `tex1D<float4>` on the TF texture returned
`(0,0,0,0)` for all lookups, producing zero accumulated colour and zero alpha.

**Root cause:** The 1D TF LUT was uploaded as a `cudaTextureObject_t` using `cudaArray` +
`cudaMemcpy2DToArray` (and later `cudaMalloc` + `cudaResourceTypeLinear`). In both cases,
`tex1D<float4>` in the raygen kernel returned zeros despite the host-side LUT containing valid
non-zero values. The 3D volume texture (also a `cudaTextureObject_t`) worked correctly, so the
issue was specific to the 1D texture or the TF binding path through `__constant__ Params`.

**Fix:** Replaced the `cudaTextureObject_t` with a raw `float4*` device pointer. The TF data is
uploaded via `cudaMalloc` + `cudaMemcpy` and indexed directly in the kernel with integer
arithmetic:

```cuda
int tfIdx = __float2int_rn(tf_t);
tfIdx = max(0, min(2047, tfIdx));
float4 tfVal = params.tfData[tfIdx];
```

**Files changed:**
- `shared_device.h`: `cudaTextureObject_t tfTex` → `float4* tfData`
- `device_programs.cu`: `tex1D<float4>(params.tfTex, ...)` → `params.tfData[tfIdx]`
- `app_main.cpp`: Removed `cudaArray`/`cudaCreateTextureObject` for TF; replaced with
  `cudaMalloc` + `cudaMemcpy`

**Verification:** After fix, GPU printf confirmed `tfVal=(0.9703,0.6703,0.5148,0.5703)` and
`accumA=0.9925` for the CT centre pixel.

## Bug 2: Solid opaque cube (TF alpha values too high)

**Symptom:** After Bug 1 was fixed, the volume rendered as a solid opaque cube with no visible
internal structure. The front face was uniformly coloured with the TF colour for soft tissue,
and early ray termination fired after only ~5 steps.

**Root cause:** The default CT transfer function alpha values were too high for the step size.
With soft tissue alpha density of 0.1–0.4 per mm and step size ~0.34 mm, the per-step opacity
was 0.03–0.14, reaching full accumulation (α ≥ 0.99) in ~10 steps (~3.4 mm). Only the outermost
3 mm of tissue was visible, appearing as a solid block.

**Fix:** Reduced alpha values across both CT and MR transfer functions by approximately one order
of magnitude:

| Region | Old alpha (per mm) | New alpha (per mm) |
|---|---|---|
| CT air (-900..-700 HU) | 0–0.02 | 0–0.002 |
| CT lung (-700..-200 HU) | 0.02–0.10 | 0.002–0.010 |
| CT soft tissue (-200..+50 HU) | 0.10–0.40 | 0.01–0.05 |
| CT muscle (50..200 HU) | 0.40–0.60 | 0.05–0.15 |
| CT bone (200..500 HU) | 0.60–0.90 | 0.15–0.45 |
| CT dense bone (>500 HU) | 0.90 | 0.80 |
| MR low (100..500) | 0.05–0.30 | 0.005–0.030 |
| MR mid (500..1500) | 0.30–0.60 | 0.03–0.08 |
| MR high (>1500) | 0.60 | 0.15 |

**Files changed:** `transfer_function.h` — both `buildDefaultCT()` and `buildDefaultMR()`.

**Result:** CT dataset renders with visible internal structure (soft tissue semi-transparent,
bone visible through the tissue).

## Bug 3: MR dataset renders incorrectly (scalar-to-TF mapping hardcoded for CT)

**Symptom:** The MR dataset (`manifest-1787683033169_DICOM/series_1`, 640×640×104, range
[0, 697]) rendered as a uniform opaque mass with no structural detail, while the CT dataset
rendered correctly.

**Root cause:** The TF index mapping formula in the kernel was hardcoded for CT Hounsfield
Units:

```cuda
float tf_t = (scalar + 1000.0f) * 2047.0f / 2000.0f;
```

This maps scalar −1000→+1000 to LUT index 0→2047, which is correct for CT. For MR data
(range 0–697), this formula maps all voxels to LUT indices 1023–1737, which all fall in the
"else" branch of the MR TF (uniform colour `(0.8, 0.7, 0.5)`, uniform alpha 0.15).

**Fix:** Added `float scalarMin` and `float scalarMax` to the `Params` struct. These are set
on the host based on modality:
- CT: `scalarMin = −1000`, `scalarMax = +1000` (matches `buildDefaultCT()` LUT range)
- MR: `scalarMin = 0`, `scalarMax = 4096` (matches `buildDefaultMR()` LUT range)

The kernel now uses a generic linear mapping:

```cuda
float tf_t = (scalar - params.scalarMin) / (params.scalarMax - params.scalarMin) * 2047.0f;
```

**Files changed:**
- `shared_device.h`: Added `float scalarMin, scalarMax` to `Params`
- `device_programs.cu`: Replaced hardcoded formula with generic linear mapping
- `app_main.cpp`: Set `scalarMin`/`scalarMax` based on `volume.modality`

**Result:** MR scalar values now map correctly across the full LUT range, producing distinct
colours for different tissue intensities instead of a uniform block.

## Summary of final state

After all three fixes, both CT and MR datasets render correctly:

| Dataset | Modality | Dimensions | Scalar range | TF range | Status |
|---|---|---|---|---|---|
| `manifest-1787681973389_DICOM/series_1` | CT | 512×512×90 | [−1024, 1665] | [−1000, 1000] | Rendering with visible structure |
| `manifest-1787683033169_DICOM/series_1` | MR | 640×640×104 | [0, 697] | [0, 4096] | Rendering with correct TF mapping |
