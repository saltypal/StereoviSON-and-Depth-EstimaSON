"""Generate printable targets and identify corresponding ArUco corners."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


MARKER_ID = 0
DICTIONARY_NAME = "DICT_4X4_50"


def detect_marker(image: np.ndarray, marker_id: int = MARKER_ID) -> np.ndarray:
    """Return canonical TL, TR, BR, BL corners; never sort by image location."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        raise ValueError(f"Marker ID {marker_id} not detected. Show the whole marker, retain its white margin, and improve lighting/sharpness.")
    indices = np.flatnonzero(ids.reshape(-1) == marker_id)
    if len(indices) != 1:
        raise ValueError(f"Need exactly one marker ID {marker_id}; found {len(indices)}.")
    return np.asarray(corners[int(indices[0])], dtype=np.float64).reshape(4, 2)


def prepare_targets(output_directory: str | Path) -> list[Path]:
    """Save exact-size PNGs at 254 DPI (10 pixels/mm). Verify after printing."""
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 1500)
    marker_page = np.full((2970, 2100), 255, dtype=np.uint8)
    marker_page[735:2235, 300:1800] = marker
    marker_path = directory / "aruco_id0_A4_150mm.png"
    Image.fromarray(marker_page).save(marker_path, dpi=(254, 254))

    # 10 x 7 SQUARES produce 9 x 6 INTERNAL corners; each square is 25 mm.
    checkerboard = np.full((2100, 2970), 255, dtype=np.uint8)
    for row in range(7):
        for column in range(10):
            if (row + column) % 2 == 0:
                x, y = 235 + column * 250, 175 + row * 250
                checkerboard[y:y + 250, x:x + 250] = 0
    checkerboard_path = directory / "checkerboard_A4_9x6_25mm.png"
    Image.fromarray(checkerboard).save(checkerboard_path, dpi=(254, 254))
    instructions = directory / "PRINT_INSTRUCTIONS.txt"
    instructions.write_text(
        "PRINT AT ACTUAL SIZE / 100%, not fit-to-page. These PNGs carry 254 DPI metadata.\n"
        "Marker: A4 portrait, black square 150 x 150 mm; keep surrounding white margin.\n"
        "Checkerboard: A4 landscape, 10 x 7 squares = 9 x 6 internal corners.\n"
        "Each checkerboard square should be 25 x 25 mm. Board is 250 x 175 mm.\n"
        "Measure the actual printed square width AND height with a ruler. Enter the measured square size.\n"
        "Printer/software may ignore DPI. Adjust print scale until dimensions match.\n"
        "Use flat matte paper on rigid backing. Do not bend or stretch either target.\n",
        encoding="utf-8",
    )
    return [marker_path, checkerboard_path, instructions]
