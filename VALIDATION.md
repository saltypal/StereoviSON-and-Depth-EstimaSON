# Validation record — 13 October 2026

## Verified software

- Windows Python 3.14 isolated environment in `.venv`.
- Pinned OpenCV 4.13.0.92, NumPy 2.4.3, and Pillow 12.1.1 installed successfully.
- Dependency consistency check passed.
- Final automated suite: **21 tests passed** in the isolated environment.
- Real Tkinter widgets were instantiated in hidden windows; background image processing, invalid form handling, and clearing previous results after failed estimation were verified.
- Both printable PNG targets generated successfully; marker detection and 9 × 6 checkerboard-corner detection passed. PNG dimensions and 254 DPI metadata were checked, and the generated checkerboard was visually inspected.
- Synthetic two-image pipeline recovered **2.000 m**, with four disparities of **100 pixels**, a **0.20 m** baseline, and zero vertical mismatch. Corrected annotations and JSON were saved.
- Synthetic evidence is labelled in calibration, interface and run records, and excluded from the physical summary.
- Physical summary currently has **0 physical captures**, and correctly reports that the accuracy target is not demonstrated.
- Calibration command correctly rejects the empty physical-photo folder without creating a fabricated camera calibration.

Reproduce the suite from the project folder:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m pip check
```

## Evidence limits

Calibration fitting was exercised with synthetic projected checkerboard corners; detector validation was separately performed on the printable checkerboard. No real phone calibration images or stereo captures were available. No printer, physical guide, lighting arrangement, or ruler measurement was tested. GUI smoke tests used hidden windows; they verify widget creation and behaviour rather than visual appearance on every display.

The synthetic demonstration verifies the software's controlled ideal-geometry path. It does not establish 0.5–3 m field accuracy. The real 18-pair protocol, marker/board print measurements, phone calibration, and mechanical alignment checks remain to be performed using the README capture guide.
