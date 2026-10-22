"""Explicitly synthetic fixtures. These are not physical camera measurements."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.calibration import Calibration, save_calibration


def synthetic_calibration() -> Calibration:
    matrix = np.array([[1000., 0., 499.5], [0., 1000., 349.5], [0., 0., 1.]])
    return Calibration(matrix, np.zeros(5), (1000, 700), 0.0, 15, {
        "evidence_kind": "synthetic",
        "description": "Invented ideal camera intrinsics for software validation; never use for physical photographs.",
    })


def create_synthetic_pair(directory: str | Path) -> tuple[Path, Path, Calibration]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 100)
    paths = []
    for name, x in (("left", 450), ("right", 350)):
        image = np.full((700, 1000), 255, dtype=np.uint8)
        image[300:400, x:x + 100] = marker
        path = directory / f"{name}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
    calibration = synthetic_calibration()
    save_calibration(calibration, directory / "SYNTHETIC_calibration.json")
    return paths[0], paths[1], calibration
