# Implicit VR Little Endian Support in the DICOM Loader

Documents the addition of **Implicit VR Little Endian** transfer-syntax support to the
hand-written DICOM parser in `src/app/dicom_loader.cpp`. This extends the data pipeline
(Section 5.4) beyond the *only* Explicit VR Little Endian files the original parser could read.

## Motivation

The TCIA manifests do not all use the same transfer syntax. A batch test of all 47 series
(`./build/bin/optix_app <series_dir> --frames 5`) showed **42 loaded and 5 failed**:

- 42 working series use **Explicit VR Little Endian** (`1.2.840.10008.1.2.1`).
- 5 failing series use **Implicit VR Little Endian** (`1.2.840.10008.1.2`):
  - `manifest-1787681973389_DICOM/series_4`
  - `manifest-1788373825860_DICOM/series_1`
  - `manifest-1788373825860_DICOM/series_2`
  - `manifest-1788373825860_DICOM/series_6`
  - `manifest-1788373825860_DICOM/series_7`

All five threw `No valid DICOM slices loaded from: <path>` because the parser never found the
pixel-data tag `(7FE0,0010)`.

### Why the original parser failed on implicit VR

The original parser was written exclusively for **explicit VR**: after every `(group,element)`
tag it read a 2-byte VR field, then a 2- or 4-byte length depending on the VR. In **implicit VR**
there is **no VR field at all** — the bytes that the parser read as a VR were actually the start
of the value. It immediately lost byte alignment and could never reach pixel data.

A secondary, more subtle issue: in implicit VR Little Endian every element carries a **32-bit
length** (not a 2-byte length). Any implicit-VR branch therefore had to read four bytes per
element length, not two.

## Changes

All changes are confined to `src/app/dicom_loader.cpp`.

### 1. Transfer-syntax detection from the File Meta Group

Per the DICOM standard the File Meta Group (group `0002`) is always **explicit VR**, even inside
an implicit-VR dataset. The loader now scans the meta group first, reading the **Transfer Syntax
UID** element `(0002,0010)`, and only then parses the rest of the dataset in the detected mode.

```cpp
// The File Meta Group (group 0x0002) is always Explicit VR even in
// Implicit VR Little Endian datasets. Scan it first to learn the
// Transfer Syntax UID (0002,0010) and decide how to parse the dataset.
bool explicitVR = true;
while (f.good()) {
    // read tag ...
    if (group != 0x0002) { f.seekg(-4, std::ios::cur); break; }  // dataset begins
    // read VR + length (explicit VR form) ...
    if (elem == 0x0010) {
        // read UID ...
        explicitVR = (uid != "1.2.840.10008.1.2");  // implicit => false
    }
    // seek past value ...
}
```

The UID is trimmed of numeric padding/NULs before comparison, so `1.2.840.10008.1.2` is matched
exactly. Any other transfer syntax keeps the original explicit-VR path (so the 42 working series
are byte-for-byte unaffected).

### 2. Implicit-VR element parsing (32-bit length, no VR)

Inside the dataset loop, when `explicitVR == false` the element length is read as a **32-bit
uint** directly after the tag (no VR field, no reserved bytes):

```cpp
else {
    // Implicit VR Little Endian: every element has a 32-bit length.
    f.read(reinterpret_cast<char*>(&dataLen), 4);
}
```

The existing metadata extraction (rows/columns, pixel spacing, rescale slope/intercept, image
position/orientation, modality, instance number) is unchanged and now operates correctly on the
implicit layout because it is driven purely by tag + length.

### 3. Implicit-VR pixel data length

The pixel-data element `(7FE0,0010)` is read with a 4-byte length in both modes. In implicit VR
the `explicitVR == false` branch reads the 32-bit length directly:

```cpp
if (explicitVR) {
    char vrCheck[2]; f.read(vrCheck, 2);
    char reserved[2]; f.read(reserved, 2);
    f.read(reinterpret_cast<char*>(&pixelLen), 4);
} else {
    f.read(reinterpret_cast<char*>(&pixelLen), 4);
}
```

### 4. `skipElement` made mode-aware for undefined-length sequences

Sequences with undefined length (`0xFFFFFFFF`) are skipped with `skipElement`, which previously
was hard-coded to explicit VR. It now takes a `bool explicitVR` argument and, in implicit mode,
reads each nested element's length as 32 bits:

```cpp
void skipElement(std::ifstream& f, uint16_t endElem, bool explicitVR) {
    // ...
    if (explicitVR) {
        // read VR, then 2- or 4-byte length ...
    } else {
        // Implicit VR Little Endian: every element has a 32-bit length.
        f.read(reinterpret_cast<char*>(&len), 4);
    }
    if (len == 0xFFFFFFFF)
        skipElement(f, 0xE0DD, explicitVR);
    else if (len > 0 && len < 0x7FFFFFFF)
        f.seekg(len, std::ios::cur);
}
```

The `FFFE` item / sequence-delimiter handling is unchanged (those always carry a 4-byte length),
and the mode is threaded through the recursive calls.

## Verification

Rebuilt the application with CMake/Make (Release, `sm_120`) and re-ran the batch test over all 47
series:

- **Before the fix:** 42 loaded, 5 failed.
- **After the fix:** **47 loaded, 0 failed.**

All five previously-failing series now report e.g. `Loaded 110 slices ... (modality=CT, 512x512)`
and reach the render loop / metrics summary with exit code 0. The 42 explicit-VR series were
re-confirmed to load identically (no regression).

Cross-check: the same five series are decoded correctly by an independent DICOM library
(pydicom) — `TransferSyntaxUID = 1.2.840.10008.1.2`, `Rows = 512`, `Cols = 512`, valid pixel
data — confirming the datasets are conformant and the limitation was in the hand-written parser.

## Scope notes

- Only **Implicit VR Little Endian** (`1.2.840.10008.1.2`) is added, matching the five TCIA
  datasets. **Explicit VR Big Endian** (`1.2.840.10008.1.2.2`) and other transfer syntaxes remain
  out of scope; any UID other than the implicit one keeps the explicit-VR path.
- Deflated/encapsulated transfer syntaxes (e.g. `1.2.840.10008.1.2.1.99`) are not handled.
