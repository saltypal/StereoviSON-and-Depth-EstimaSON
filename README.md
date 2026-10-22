# Stereo Vision: Distance from Two Phone Images

A small CPU-only desktop assignment. Keep an object still, move one phone **20 cm to the right without rotating it**, and estimate the distance from horizontal displacement of an attached ArUco marker. Planned validation range: **0.5–3 m**.

**Software tests are not physical accuracy results.** The supplied synthetic example validates the software only. You must calibrate your phone and complete ruler measurements before claiming real accuracy.

## Start the application

Open PowerShell in the project folder:

```powershell
Set-Location 'D:\Bunker\BaseCamp\ComputerVision\StereoviSON and Depth EstimaSON'
& '.\.venv\Scripts\python.exe' app.py
```

The isolated environment is provided on this machine. To recreate it on another Windows computer, install Python 3.14 with Tkinter, then run:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

Tkinter comes with the Windows Python installer; it is not a pip package. Install only one OpenCV wheel. Current `opencv-python` already includes the ArUco functionality used here; a second `opencv-contrib-python` install is unnecessary and can conflict.

## Part 1: Prepare and print the targets

```powershell
& '.\.venv\Scripts\python.exe' app.py --prepare-targets
```

Outputs in `data/targets/`:

- `aruco_id0_A4_150mm.png`: A4 portrait; black marker square 150 × 150 mm. Keep the white surround.
- `checkerboard_A4_9x6_25mm.png`: A4 landscape; **10 × 7 squares**, therefore **9 × 6 internal corners**. Each square is nominally 25 mm.
- `PRINT_INSTRUCTIONS.txt`: sizing instructions.

Print at **actual size / 100%**, with scaling disabled. PNGs carry 254 DPI metadata, but print software may ignore it. Measure the printed dimensions. Measure checkerboard square width and height; they should match. Enter the actual square size during calibration. Attach the paper to flat rigid backing and avoid shiny lamination.

Attach the marker to the front of a stationary box or similar object. We measure the centre of this marked surface, not the centre of the object's volume. The marker's physical size is used for visibility, **not** to supply depth scale.

## Part 2: Calibrate your phone once

Calibration finds the focal lengths in pixels, the principal point, and lens distortion. It needs photographs **in addition to** the two photographs used for each stereo measurement.

1. Choose the main camera at 1× zoom and lock focus. Keep lens, focus, resolution, and orientation consistent throughout the project. Disable automatic macro/lens switching and stabilization where practical. Use JPEG rather than HEIC.
2. Photograph the entire checkerboard at least 20 times. Move/tilt the board between photographs so it covers different image regions and viewing angles. This movement is for calibration only; the stereo object stays still during its pair.
3. Keep the board flat, sharp, evenly lit, and fully visible. Include views near the centre and edges. Avoid a set of almost identical views.
4. Transfer original photographs into `data/calibration/`. Do not send through services that resize or compress them. Keep all images the same dimensions/orientation.
5. Run the command below, replacing `25` if your measured squares differ:

```powershell
& '.\.venv\Scripts\python.exe' app.py --calibrate data/calibration --square-mm 25
```

The command creates `data/camera_calibration.json`, including used/rejected images, per-view errors, and overall RMS. It requires at least **15 successfully detected boards**. RMS above **1 pixel** fails the quality gate: the diagnostic file is saved, but the desktop app refuses it.

Low reprojection RMS means the fitted camera model explains those checkerboard images. It does **not** prove physical distance accuracy or eliminate the need for varied views. If focus, zoom, lens, image resolution, or cropping changes, recalibrate.

## Part 3: Capture a stereo pair

Use a rigid phone holder that slides on a straight horizontal guide with two stops **20 cm apart**. A sliding cradle against a straight edge is sufficient if it constrains rotation and height. Moving the same holder by a measured amount moves the lens by that amount.

```text
                  FIXED OBJECT + MARKER
                       [ ID 0 ]

                    viewing direction
                    ↑               ↑
                 [phone L] -----> [phone R]
                          20 cm
             Same height and orientation at both stops
```

- Keep the object on a stable platform. Place the marker approximately at camera height and make it clearly visible in both positions.
- Use steady diffuse illumination. Avoid glare, dark shadows across the border, blur, and changes in lighting between shots.
- At the left stop, take the left image. Slide the holder right; take the right image. **Do not turn the phone to centre the marker again.**
- Keep the original image settings. Do not digitally shift, crop, resize, rotate, or independently stabilize the two images.
- At 3 m, check that the marker is still clear; aim for at least roughly 60 pixels across the black square. A larger print can improve detection without changing the calculation.
- Store the originals in `data/pairs/`, with descriptive names such as `1m_trial1_left.jpg` and `1m_trial1_right.jpg`.

## Part 4: Estimate and save

Open the desktop app, choose `data/camera_calibration.json`, and choose both photographs. Enter the measured camera separation in **centimetres**. Optional ground truth is the ruler/tape distance from the **left camera lens position to the marker centre**, in **metres**.

Click **Estimate**. The application:

1. Checks that both images match the calibration dimensions.
2. Undistorts both full-resolution images while retaining the same camera matrix and frame size.
3. Finds marker ID 0 and refines its four corners.
4. Checks positive disparities, minimum disparity of 10 pixels, and maximum vertical mismatch of 2 pixels.
5. Reconstructs all four corners and averages their 3D positions.
6. Displays distance, depth, corner disparities, and ruler error if supplied.

Preview images are resized for the window only. The calculations use original-resolution corrected pixels. A failed estimate clears the previous estimate and disables saving. Accepted captures are saved when you click **Save Result**. Rejected image-processing attempts are recorded automatically after valid inputs were submitted. Invalid form entries are not physical capture attempts.

Each accepted run folder under `results/` includes:

- `comparison.png`: corrected views, corner numbering, and numerical measurements.
- `result.json`: coordinates, baseline, calibration snapshot and identity, image hashes, environment versions, ground truth, and errors.
- Original left/right photographs, so the result remains reviewable even if source folders move.

Rejected records include their reason and input/calibration identity, with no invented distance. Save a capture once. Reprocessing the same originals counts as the same capture; the summary uses its latest record. A fresh physical trial requires two new photographs.

## The three concepts you should be able to explain

**Calibration:** Camera focal length in millimetres is not what this formula needs. The calibrated matrix gives `fx` and `fy` in pixels, and `cx`, `cy` locate the principal point. Lens distortion changes where features appear, so we correct it before measuring disparities. Keeping the same matrix during undistortion prevents an unnoticed coordinate-system change.

**Correspondence:** Stereo needs the *same physical point* in two images. The marker's ID and its canonical ordered corners give four reliable correspondences. Never sort corners simply by their image x/y position; that can change their identities. Each number refers to the same marker corner in both views.

**Triangulation:** A closer point shifts farther across the image when the camera moves sideways. For parallel camera positions with the same intrinsics:

```text
d = u_left - u_right        [pixels]
Z = fx × baseline / d      [metres]
X = (u_left - cx) × Z / fx
Y = (v_left - cy) × Z / fy
```

For each corner, reconstruct `(X, Y, Z)` relative to the left camera. Average the four reconstructed positions to get the marker centre, then compute:

```text
Depth = centre_Z
Distance = sqrt(centre_X² + centre_Y² + centre_Z²)
```

Depth is the forward component; distance is the straight-line length. They differ when the marker is off-axis. Averaging 3D corners also handles a tilted, flat marker better than treating its 2D bounding-box centre as a physical point.

For an illustrative focal length of 1500 pixels, a 0.20 m baseline, and a 100-pixel disparity, depth is 3 m. Focal length must come from your calibration; do not copy this example value.

## Part 5: Validate both distance ranges

Collect **three independent pairs each** at approximately **0.5, 0.75, 1, 1.5, 2, and 3 m**: 18 pairs total. Keep the object fixed within each pair. Record the tape measurement, lens reference, ruler precision, baseline, and lighting. Enter actual measured distance rather than the nominal label.

Click **Summarize Physical Trials**, or run:

```powershell
& '.\.venv\Scripts\python.exe' app.py --summarize
```

This creates `results/evaluation_readout.md`, `evaluation_summary.json`, and `evaluation_runs.csv`. The readable report includes per-distance errors, rejection counts, and repeatability (sample standard deviation). Target bins accept ruler readings within 1 cm of their nominal distances.

The **near range** is 0.5, 0.75, and 1 m; the **far range** is 1.5, 2, and 3 m. Each range's mean percentage error gives its three distances equal weight. The initial target requires:

- At least three independent attempts at every test distance.
- At least two accepted captures at every distance.
- Mean absolute percentage error at most 10% in **each** range.

Synthetic runs, duplicates, and runs without verifiable original-image identity do not satisfy coverage. Captures without ground truth cannot produce accuracy scores. If the target fails, report the observed errors and investigate the setup; do not claim full-range accuracy.

For capture notes, use this format in your assignment log:

| Nominal distance | Trial | Actual tape distance | Baseline | Lens reference / ruler precision | Lighting / outcome |
|---|---|---|---|---|---|
| 0.5 m | 1 | | 20 cm | | |

## Tests and synthetic demonstration

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Tests cover analytic 3D recovery, off-axis range, unequal focal lengths, a tilted marker, invalid inputs, threshold boundaries, real ArUco detection, duplicate/missing markers, resolution mismatch, calibration fit using synthetic projections, print dimensions, saved results, and evaluation accounting.

The checkerboard detector is tested on the printable checkerboard. Calibration fitting is separately tested using synthetic projected corners; that test does not claim real-phone calibration. The application is also smoke-tested with a hidden Tkinter window and asynchronous estimation.

`data/synthetic_demo/` contains clearly labelled invented images and an invented calibration. You can load those files to check the interface: baseline **20 cm**, reference distance **2 m**. Synthetic labels propagate into result files, and these runs are excluded from physical evaluation. **Never use this calibration for phone photographs.**

## Architecture and limitations

| File | What it owns |
|---|---|
| `app.py` | Desktop interface, input fields, asynchronous estimation, preparation/calibration commands. |
| `src/calibration.py` | Image reading, intrinsic calibration, JSON format and quality gate. |
| `src/markers.py` | Printable targets and marker correspondence. |
| `src/geometry.py` | Disparity checks and 3D reconstruction, independent of the GUI. |
| `src/reporting.py` | Annotations, saved evidence, and physical trial summaries. |

The simple model assumes translation along X with no camera rotation or vertical/depth movement. **Vertical agreement alone cannot detect all violations:** a small yaw can still bias depth. Do not attempt to hide capture errors by shifting images into alignment. Use a better guide and recapture. Full stereo pose calibration/rectification would be a separate extension if the physical setup cannot maintain alignment.

Other limitations include changing phone intrinsics, blur, uneven illumination, a bent marker/board, small disparities at greater distances, uncertain baseline, and tape measurements referenced to the external lens surface rather than the exact optical centre. The marker must remain fully visible. This implementation estimates one marked surface; it does not estimate an entire scene or an unmarked object's volume centre.

## References mapped to this project

- [OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html): checkerboard corners, camera matrix, distortion, and reprojection error; maps to `calibration.py`.
- [OpenCV ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html): dictionary, marker ID, ordered corners, and corner refinement; maps to `markers.py`.
- [OpenCV stereo depth](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html): disparity and the `Bf/Z` relationship; maps to `geometry.py`. We use sparse marker correspondences instead of its dense depth-map example.
