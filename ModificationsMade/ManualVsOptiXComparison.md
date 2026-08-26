# Manual vs. OptiX Trace — Implementation Comparison

Side-by-side comparison of the two traversal strategies for the volume raycaster,
showing how the same visual result is achieved through different hardware paths.

## Conceptual difference

| | Manual mode (`--mode manual`) | OptiX mode (`--mode optix`) |
|---|---|---|
| **Who finds the ray-AABB hit?** | The raygen program, in scalar ALU code | The RT-core BVH hardware, then confirmed by the intersection program |
| **Who does the volume march?** | The raygen program | The closest-hit program (`__closesthit__ch_vol`) |
| **RT-core usage** | Idle | Active (BVH traversal) |
| **Role of the GAS** | Placeholder — required by `optixLaunch` but never traversed | Functional — the AABB leaf is the volume bounding box |
| **Traversable handle source** | `PlaceholderScene` (triangle mesh) | `VolumeScene` (custom-primitive AABB) |

## Code path comparison

### Ray generation

Both modes compute the camera ray identically:

```cuda
const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
float2 d = 2.0f * make_float2(
    static_cast<float>(idx.x) / static_cast<float>(dim.x),
    static_cast<float>(idx.y) / static_cast<float>(dim.y)) - 1.0f;
float3 origin    = rtData->cam_eye;
float3 direction = normalize(d.x * rtData->camera_u
                           + d.y * rtData->camera_v
                           + rtData->camera_w);
```

The divergence begins immediately after.

### Ray-AABB intersection

**Manual** — pure software slab test in the raygen program:

```cuda
float tmin, tmax;
if (intersectAABB(origin, direction,
                  params.volumeOrigin, params.volumeMax, tmin, tmax))
{
    if (tmin < 0.0f) tmin = 0.0f;
    color = volumeMarch(origin, direction, tmin, tmax);
}
```

The raygen does everything: slab test, then volume march. No GPU hardware
specialised in ray traversal is involved. The `intersectAABB` function is a
branchless slab test operating on three float3 axes:

```cuda
float3 inv_dir = make_float3(1.0f/direction.x, 1.0f/direction.y, 1.0f/direction.z);
float3 t0 = (bmin - origin) * inv_dir;
float3 t1 = (bmax - origin) * inv_dir;
float3 tmin3 = fminf(t0, t1);
float3 tmax3 = fmaxf(t0, t1);
tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
tmax = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);
return tmin <= tmax && tmax > 0.0f;
```

**OptiX** — `optixTrace` delegates traversal to RT-core hardware:

```cuda
unsigned int p0 = __float_as_uint(0.0f);
unsigned int p1 = __float_as_uint(0.0f);
unsigned int p2 = __float_as_uint(0.0f);

optixTrace(params.handle,
           origin, direction,
           0.0f, 1e30f,        // tmin, tmax
           0.0f,               // ray time
           OptixVisibilityMask(0xFF),
           OPTIX_RAY_FLAG_NONE,
           0, 1, 0,            // SBT offset, stride, miss index
           p0, p1, p2);

float3 color = make_float3(
    __uint_as_float(p0),
    __uint_as_float(p1),
    __uint_as_float(p2));
```

The raygen does **not** call `intersectAABB`. Instead it hands the ray to the
RT-core hardware via `optixTrace`. The hardware traverses the BVH (built from
the AABB in `VolumeScene`), finds the leaf, and invokes `__intersection__is`.

### Intersection program (OptiX mode only)

When the RT-core finds the AABB leaf, it invokes `__intersection__is`:

```cuda
extern "C" __global__ void __intersection__is()
{
    float3 origin    = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float  tmin_ray  = optixGetRayTmin();
    float  tmax_ray  = optixGetRayTmax();

    float tmin, tmax;
    if (!intersectAABB(origin, direction,
                       params.volumeOrigin, params.volumeMax, tmin, tmax))
        return;

    tmin = fmaxf(tmin, tmin_ray);
    tmax = fminf(tmax, tmax_ray);

    if (tmin <= tmax)
    {
        optixReportIntersection(tmin, 0,
            __float_as_uint(tmin),   // attribute 0: entry t
            __float_as_uint(tmax));  // attribute 1: exit t
    }
}
```

The same `intersectAABB` function is used, but now the ray origin/direction
come from `optixGetWorldRayOrigin/Direction` (set by the RT-core during
traversal) rather than being locally computed. The entry/exit t-values are
packed into two 32-bit OptiX attributes and passed to the closest-hit program.

This program does **not exist** in manual mode. The hitgroup is created with
`moduleIS = nullptr`.

### Closest-hit program (OptiX mode only)

```cuda
extern "C" __global__ void __closesthit__ch_vol()
{
    float tmin = __uint_as_float(optixGetAttribute_0());
    float tmax = __uint_as_float(optixGetAttribute_1());
    float3 color = volumeMarch(optixGetWorldRayOrigin(),
                               optixGetWorldRayDirection(), tmin, tmax);
    setPayload(color);
}
```

Reads the entry/exit t-values from attributes, then calls the same
`volumeMarch` helper as manual mode. The result is packed into payload slots
via `optixSetPayload_0/1/2` and read back by the raygen through the
`optixTrace` output parameters `p0/p1/p2`.

### Volume march (shared)

Both modes call the same function:

```cuda
static __forceinline__ __device__ float3 volumeMarch(
    float3 origin, float3 direction, float tmin, float tmax)
{
    float stepSize = fminf(params.volumeSpacing.x,
                   fminf(params.volumeSpacing.y, params.volumeSpacing.z)) * 0.5f;

    float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f, accumA = 0.0f;
    int maxSteps = static_cast<int>((tmax - tmin) / stepSize) + 1;
    if (maxSteps > 4096) maxSteps = 4096;

    for (int i = 0; i < maxSteps && accumA < 0.99f; ++i)
    {
        float t = tmin + i * stepSize;
        float3 samplePos = origin + direction * t;
        float3 texCoord = (samplePos - params.volumeOrigin) / volumeSize;
        // clamp, tex3D, TF lookup, front-to-back compositing...
    }
    return make_float3(accumR, accumG, accumB);
}
```

The function takes `origin` and `direction` as explicit parameters rather than
calling `optixGetWorldRayOrigin/Direction` internally. This is necessary because
the manual raygen has no active OptiX ray context — those functions are only
valid during or after `optixTrace`.

## GAS construction comparison

### PlaceholderScene (manual mode)

```
Type:           Triangle mesh (OPTIX_BUILD_INPUT_TYPE_TRIANGLE_ARRAY)
Contents:       Geometric primitives (cube/plane) — irrelevant to rendering
Purpose:        Satisfy optixLaunch's requirement for a traversable handle
RT-core usage:  None (optixTrace is never called)
```

### VolumeScene (optix mode)

```
Type:           Custom primitives (OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES)
Contents:       Single OptixAabb = volume bounding box
Purpose:        The RT-core traverses this BVH; the AABB leaf triggers __intersection__is
RT-core usage:  Active (broad-phase traversal of BVH hierarchy)
```

The AABB is constructed from `volumeOrigin` and `volumeMax`:

```cpp
OptixAabb aabb = {
    bmin.x, bmin.y, bmin.z,   // volume origin corner
    bmax.x, bmax.y, bmax.z    // volume max corner
};
```

Build flags include `OPTIX_BUILD_FLAG_ALLOW_COMPACTION` to reduce the
output buffer size when the compacted GAS is smaller.

## Pipeline configuration

| Parameter | Value | Notes |
|---|---|---|
| `numPayloadValues` | 3 | RGB colour packed as 3 × uint32 |
| `numAttributeValues` | 2 | Entry and exit t-values (OptiX mode) |
| `traversableGraphFlags` | `ALLOW_SINGLE_GAS` | One-level BVH (no instances) |
| `usesPrimitiveTypeFlags` | 0 | Enables both CUSTOM and TRIANGLE (single OptiX IR for both modes) |
| `maxTraceDepth` | 1 | Single `optixTrace` call per ray |
| `pipelineLaunchParamsVariableName` | `"params"` | `__constant__ Params params` |

## SBT (Shader Binding Table) layout

| Record | Data | Notes |
|---|---|---|
| Raygen | `RayGenData { cam_eye, camera_u, camera_v, camera_w }` | Updated every frame |
| Miss | `MissData { 0.3, 0.1, 0.2 }` | Background gradient (unused — rays always hit AABB) |
| Hitgroup | `HitGroupData {}` (empty) | Intersection data comes from the GAS AABB, not SBT |

## Program groups created

| Program group | MANUAL | OPTIX |
|---|---|---|
| `raygenPG_` | `__raygen__rg` | `__raygen__rg_optix` |
| `missPG_` | `__miss__ms` | `__miss__ms` |
| `hitgroupPG_` | `__closesthit__ch` (stub) | `__closesthit__ch_vol` + `__intersection__is` |

The intersection program is not a separate program group in OptiX 9.1 — it is
embedded in the hitgroup via `hitgroup_desc.hitgroup.moduleIS` and
`entryFunctionNameIS`.

## Data flow summary

```
MANUAL MODE:

  Camera ──► __raygen__rg
                │
                ├── intersectAABB(origin, direction, bmin, bmax)
                │     └── returns tmin, tmax  (pure ALU slab test)
                │
                └── volumeMarch(origin, direction, tmin, tmax)
                      ├── tex3D(volumeTex, ...)          — 3D texture fetch
                      ├── params.tfData[idx]             — TF lookup
                      └── front-to-back compositing
                            └── result written to params.image


OPTIX MODE:

  Camera ──► __raygen__rg_optix
                │
                └── optixTrace(handle, origin, direction, ...)
                      │
                      ├── RT-core BVH traversal  (hardware)
                      │     └── finds AABB leaf
                      │
                      ├── __intersection__is
                      │     ├── optixGetWorldRayOrigin/Direction()
                      │     ├── intersectAABB(...)
                      │     └── optixReportIntersection(tmin, 0, tmin, tmax)
                      │
                      ├── __closesthit__ch_vol
                      │     ├── optixGetAttribute_0/1()  — read tmin, tmax
                      │     ├── volumeMarch(origin, direction, tmin, tmax)
                      │     │     ├── tex3D(volumeTex, ...)
                      │     │     ├── params.tfData[idx]
                      │     │     └── front-to-back compositing
                      │     └── setPayload(color)
                      │           └── writes p0, p1, p2
                      │
                      └── raygen reads p0, p1, p2
                            └── result written to params.image
```

## What differs at runtime

| Aspect | Manual | OptiX |
|---|---|---|
| Slab test execution | Scalar ALU in raygen warp | RT-core hardware (broad) + IS in ALU (narrow) |
| Volume march execution | In raygen warp | In closest-hit warp |
| Ray-AABB hit detection | Software | Hardware BVH + software confirmation |
| Branching in raygen | Single if-block (intersectAABB result) | Single optixTrace call, then payload read |
| GPU unit utilisation | Shader cores only | Shader cores + RT cores |
| Scalability for sparsity | Cannot skip empty regions — raygen iterates all voxels | Could skip empty BVH subtrees (future work) |

## Thesis relevance

The manual mode is the **control condition** (dense baseline). It demonstrates
what volume ray-casting looks like when every ray visits every voxel along its
path, with no hardware acceleration for the spatial query.

The optix mode is the **experimental condition**. It delegates spatial queries to
RT-core hardware, enabling future sparsity strategies (sparse BVH, early-out
AABBs, per-voxel AABBs) where the hardware can skip empty regions without
shader-core involvement. The performance difference between the two modes
measures the cost/benefit of RT-core acceleration for volume rendering.
