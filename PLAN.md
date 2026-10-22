# Implementation plan and acceptance gates

## Goal and scope

Estimate a stationary object's marked-front-surface distance from two phone images. One phone moves along a fixed horizontal guide, initially by 20 cm. Target validation range: 0.5–3 m. CPU-only Python desktop application; no training, live video, full depth map, or web backend.

Whiteboard implementation ownership: Codex writes the small modules and interface; the user assembles the physical support, captures calibration and stereo photographs, and measures real distances. Core concepts are reviewed through README explanations and worked examples.

## Physical setup

- Rigid phone holder and straight horizontal guide with two measured stops.
- Fixed object with matte, flat, 15 cm ArUco marker ID 0 from DICT_4X4_50.
- Stable diffuse lighting, complete marker visible in both shots.
- Same lens, focus, zoom, orientation, and resolution as calibration. Original JPEGs; no independent crop, resize, or artificial image shift.
- Depth assumes parallel camera positions and positive rightward translation; no rotation or vertical/depth movement.

## Components and gates

| Component | Inputs | Outputs | Dependencies | Owner | Acceptance |
|---|---|---|---|---|---|
| Printable targets | Marker/board defaults | Sized PNGs and print guide | OpenCV, Pillow | Codex | Printed dimensions verifiable; synthetic target detection passes |
| Calibration | Varied board photos, measured square size | K, distortion, image size, RMS and diagnostics | OpenCV, NumPy | Codex code; user photos | >=15 usable views and RMS <=1 px |
| Marker detection | Corrected image | Four canonical corresponding corners | OpenCV ArUco | Codex | Exactly one target marker in each image |
| Geometry | Corners, K, measured baseline | Four 3D corners, centre, depth and range | Calibration + detection | Codex; joint concept review | Analytic recovery passes; reject disparity <10 px or vertical mismatch >2 px |
| Desktop/reporting | Images, calibration, baseline, optional tape distance | Annotated views and durable run records | Core modules, Tkinter, Pillow | Codex | End-to-end synthetic flow and GUI state checks pass |
| Physical evaluation | 18 independent captures with tape measurements | Error/repeatability/readout and coverage | Real setup | User captures; Codex reporting | >=3 attempts and >=2 accepted at every target; <=10% MAPE in each range |

## Order

1. Generate and verify printable targets.
2. Implement intrinsic calibration and gate loading by usable views/RMS.
3. Detect and refine corresponding marker corners.
4. Reconstruct corners using Z=fx*B/d, X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy; average 3D positions and compute straight-line distance.
5. Connect the desktop window; maintain full-resolution geometry, clear stale results, and save originals, JSON, annotations and rejection reasons.
6. Validate software with synthetic analytic/image fixtures and GUI smoke checks.
7. Capture real photographs and evaluate 0.5, 0.75, 1, 1.5, 2 and 3 m, three fresh pairs per distance.

## Interfaces and coordinate conventions

Calibration JSON stores K, distortion, dimensions, usable-view count, quality diagnostics and metadata. estimate_pair takes image paths, a Calibration, baseline in metres and optional measured range in metres. Results contain centre coordinates relative to the left camera, depth, straight-line range, disparities, vertical residuals, calibration identity and source information.

Images are undistorted with unchanged K and full frame. Only previews are resized. Marker physical size does not determine scale. The guide's measured baseline determines scale. Main displayed distance is left camera to marker centre; depth is its forward component.

## Validation and evidence rules

Automated cases: known positions, off-axis/tilted points, unequal focal lengths, bad numbers, reversed/zero/small disparities, vertical mismatch, missing/duplicate markers, inconsistent resolutions, calibration fitting, printable targets, persistence and evaluation accounting.

Real acceptance requires ruler measurements and independent physical captures. Synthetic calibration/images are labelled and excluded from physical summaries. Duplicate original-image pairs count once. Report near/far MAPE separately, equal-weighted across target distances, and show rejected captures and repeatability. Low calibration RMS and vertical agreement do not independently prove stereo accuracy.

The README contains setup, commands, physical capture procedure, important functions and concept explanations. Physical calibration and range validation remain pending until actual photographs are supplied.
