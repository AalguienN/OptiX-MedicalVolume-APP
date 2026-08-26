# User Interaction Implementation — Implementation Notes

Source material for the *Development → User Interaction Implementation* section of the thesis
(`tfm.tex:928`). Covers the changes that replace the placeholder auto-orbit camera with the
interactive orbital camera described in Section *User Interaction Design* (`tfm.tex:773`).

## Objective

Implement interactive camera controls so the user can rotate, zoom, and pan the view of the
volume in real time, matching the design specified in `tfm.tex:773-774`:

- **Orbit rotation**: left-mouse drag rotates the camera around the look-at target.
- **Zoom**: scroll wheel adjusts the distance (radius) to the target.
- **Pan**: right-mouse drag translates the look-at point in the camera's local right/up plane.
- Every input event updates the camera state and the next frame reflects the most recent input.

The camera and input handling remain part of the **shared front end**; no strategy-specific
interaction logic is introduced (design principle: no strategy-specific interaction, `tfm.tex:774`).

## Coordinate system

The camera was reoriented from the default Y-up convention to a **Z-up, Y-forward** convention.
The original spherical parameterisation placed the eye at:

```
eye = lookAt + (r·cos(φ)·sin(θ),  r·sin(φ),  r·cos(φ)·cos(θ))   // Y-up, looks along −Z
```

This was replaced with:

```
eye = lookAt + (r·cos(φ)·sin(θ), −r·cos(φ)·cos(θ),  r·sin(φ))   // Z-up, looks along −Y
```

Effectively the Y and Z components are swapped and the new Y component is negated, rotating
the entire viewing frame 90° so that the former "up" axis (Y) now serves as the forward
direction.  The world-up vector used for the UVW basis and pan computation is correspondingly
changed from `(0, 1, 0)` to `(0, 0, 1)`.

This reorientation was chosen because the DICOM volume data is most naturally inspected along
its slice-stacking axis; placing the camera along −Y provides a direct view down that axis
without needing to rotate the volume data itself.

## Starting point

Before this change, `camera.h` was a header-only orbital camera with spherical coordinates
(`theta`, `phi`, `radius`, `fovY`, `aspect`) producing an eye point and UVW basis via `eye()`
and `uvw()`. It used a Y-up convention: world up was `(0, 1, 0)` and the default view looked
along −Z. It contained no input handling. In `app_main.cpp`, the camera was driven by
auto-increment (`camera.theta += dt * 0.5`), producing a continuously rotating view with no
user control. The look-at target was hardcoded to `(0, 0, 0)` inside `uvw()`.

## Changes

### `src/app/camera.h` — full rewrite

| Aspect | Before | After |
|---|---|---|
| Coordinate convention | Y-up, default view along −Z | Z-up, default view along −Y (90° rotation of viewing frame) |
| World-up vector | `(0, 1, 0)` | `(0, 0, 1)` |
| `eye()` | `(r·cos(φ)·sin(θ),  r·sin(φ),  r·cos(φ)·cos(θ))` | `(r·cos(φ)·sin(θ), −r·cos(φ)·cos(θ),  r·sin(φ))` |
| Look-at target | Hardcoded `(0,0,0)` inside `uvw()` | `float3 lookAt` member, defaults to `(0,0,0)`, used by `eye()` and `uvw()` |
| `uvw()` | Looks at hardcoded origin, Y-up | Looks at `lookAt` member, Z-up |
| Input state | None | `lastPos` (float2), `leftMouseDragging`, `rightMouseDragging` |
| Orbit | N/A | `handleOrbit(dx, dy)` — adjusts `theta -= dx·kSensitivity`, `phi += dy·kSensitivity`; `phi` clamped to ±89° to prevent gimbal lock.  Negated dx maps screen-right to the correct orbital direction under the Z-up axis mapping. |
| Zoom | N/A | `handleZoom(dy)` — scales `radius` by `(1 - dy * 0.1)`, clamped to minimum 0.1 |
| Pan | N/A | `handlePan(dx, dy)` — translates `lookAt` along camera-local right and up vectors, both **negated** (`- right · dx - up · dy`), scaled by `radius` and a reduced speed factor (`0.001f` vs the original `0.005f`). The negation makes pan follow the cursor in an intuitive screen-space direction under the Z-up frame. |
| GLFW callbacks | N/A | `onMouseButton(button, action)`, `onCursorPos(x, y)`, `onScroll(yoffset)` — thin wrappers that dispatch to the handlers above |

**Design rationale**: all input logic lives in `Camera` so it can be tested and reused
independently of the GLFW wiring. The `Camera` class remains a plain data + logic object with
no GLFW dependency; the GLFW-to-Camera bridge is in `app_main.cpp`.

### `src/app/app_main.cpp` — callback registration + auto-orbit removal

Changes to `app_main.cpp`:

1. **Added `#include <GLFW/glfw3.h>`** — required for `glfwSetMouseButtonCallback` and friends,
   since `gl_window.h` only forward-declares `GLFWwindow`.

2. **Added static GLFW callback functions** in an anonymous namespace:
   - `glfwMouseButtonCallback` — retrieves `Camera*` via `glfwGetWindowUserPointer`, calls
     `cam->onMouseButton(button, action)`.
   - `glfwCursorPosCallback` — calls `cam->onCursorPos(xpos, ypos)`.
   - `glfwScrollCallback` — calls `cam->onScroll(yoffset)`.

   These are static free functions because GLFW callbacks cannot be non-static member functions.
   The `Camera*` is passed through GLFW's per-window user pointer, avoiding global state.

3. **Registered callbacks** after `Camera` construction:
   ```
   glfwSetWindowUserPointer(window.handle(), &camera);
   glfwSetMouseButtonCallback(window.handle(), glfwMouseButtonCallback);
   glfwSetCursorPosCallback(window.handle(), glfwCursorPosCallback);
   glfwSetScrollCallback(window.handle(), glfwScrollCallback);
   ```

4. **Initialized `camera.lastPos`** from the current cursor position via
   `glfwGetCursorPos`, preventing a large first-frame delta when the cursor is
   already inside the window.

5. **Removed auto-orbit** (`camera.theta += dt * 0.5`) and the associated `lastTime`
   variable. Camera is now entirely user-driven.

### Files NOT changed

- `gl_window.{h,cpp}` — no changes; `handle()` already exposes the `GLFWwindow*`.
- `shared_device.h` — no changes; camera data flows through the existing `RayGenData` SBT record.
- `pipeline_base.{h,cpp}` — no changes.
- `display.{h,cpp}` — no changes.
- `device/device_programs.cu` — no changes; ray generation already consumes camera UVW from SBT.

## Compliance with design constraints

- **Common rendering pipeline (principle 1)**: camera input modifies only `Camera` state.
  The ray generation program, transfer function slot, compositing order, and display path are
  all unchanged. Any rendering strategy will receive the same camera data through the same
  `RayGenData` SBT record.

- **Interchangeable representations (principle 2)**: the camera module has no knowledge of
  which volume representation is active. It produces an eye point and UVW basis consumed by
  the shared ray-generation program; strategy-specific intersection programs are unaffected.

- **GPU-oriented execution (principle 3)**: no new host-device data transfer is introduced.
  Only the existing `RayGenData` record (3 × float3 + float3 = 48 bytes) is updated per frame.

- **No strategy-specific interaction logic** (`tfm.tex:774`): all input handling is in the
  shared front end (`Camera` + GLFW callbacks in `app_main.cpp`). Future strategies (dense
  baseline, sparse octree, etc.) inherit the same camera behaviour without modification.

- **Static volumes** (`tfm.tex:488`): camera interaction triggers a new `optixLaunch` per
  frame, but the volume data itself is never modified — consistent with the static-volume scope.

## Verification

- Full rebuild via `mkbuild.sh` (CMake Release, `sm_120`) completes without errors or warnings.
- The binary opens a window and renders the shaded cube. Left-drag orbits, right-drag pans,
  scroll zooms. The cube remains stationary at the origin while the camera moves around it.
  No CUDA, OptiX, or GL errors reported.
