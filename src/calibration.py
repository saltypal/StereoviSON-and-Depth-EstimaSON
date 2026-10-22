"""Calibrate one camera and retain the exact pixel coordinate system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import hashlib
import json

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    """Read an original image, including Windows paths with non-ASCII characters."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        raise ValueError(f"Cannot read image: {path}") from error
    if image is None:
        raise ValueError(f"Unreadable image: {path}. Use an original JPEG or PNG, not HEIC.")
    return image


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_px: float
    usable_views: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            raise ValueError("Calibration metadata must be a JSON object.")
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        self.image_size = tuple(self.image_size)
        if self.camera_matrix.shape != (3, 3) or not np.isfinite(self.camera_matrix).all():
            raise ValueError("Calibration must have a finite 3 by 3 camera matrix.")
        matrix = self.camera_matrix
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("Calibration focal lengths must be positive.")
        if not np.allclose(matrix[2], [0, 0, 1]) or abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8:
            raise ValueError("Expected the standard OpenCV camera matrix with zero skew.")
        if len(self.distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("Invalid OpenCV distortion coefficients.")
        if len(self.image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in self.image_size):
            raise ValueError("Calibration image dimensions must be positive integers.")
        if not np.isfinite(self.rms_px) or self.rms_px < 0:
            raise ValueError("Calibration RMS must be finite and nonnegative.")
        if not isinstance(self.usable_views, int) or self.usable_views < 0:
            raise ValueError("Calibration view count must be a nonnegative integer.")

    @property
    def quality_passed(self) -> bool:
        return self.usable_views >= 15 and self.rms_px <= 1.0

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion.tolist(),
            "image_size_wh": list(self.image_size),
            "rms_reprojection_error_px": float(self.rms_px),
            "usable_views": self.usable_views,
            "quality_passed": self.quality_passed,
            "metadata": self.metadata,
        }

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def save_calibration(calibration: Calibration, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> Calibration:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Calibration file must contain a JSON object.")
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported calibration format.")
        calibration = Calibration(
            data["camera_matrix"], data["distortion_coefficients"],
            tuple(data["image_size_wh"]), data["rms_reprojection_error_px"],
            data["usable_views"], data.get("metadata", {}),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load calibration: {error}") from error
    if not calibration.quality_passed:
        raise ValueError("Calibration fails the gate: need at least 15 usable views and RMS <= 1 pixel. Recapture and recalibrate.")
    return calibration


def calibrate_camera(
    image_paths: list[Path],
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    progress: Callable[[str], None] | None = None,
) -> Calibration:
    """Estimate intrinsics from varied checkerboard views, not the stereo pair."""
    if not np.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("Measured checkerboard square size must be positive.")
    if len(pattern_size) != 2 or any(not isinstance(v, int) or v < 2 for v in pattern_size):
        raise ValueError("Checkerboard internal-corner counts must be integers >= 2.")
    if len(image_paths) < 15:
        raise ValueError("Provide at least 15 calibration images; 20 varied photographs are recommended.")
    columns, rows = pattern_size
    object_grid = np.zeros((columns * rows, 3), dtype=np.float32)
    object_grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    object_points, image_points, used, rejected = [], [], [], []
    image_size = None
    for index, path in enumerate(image_paths):
        if progress:
            progress(f"Checking calibration photo {index + 1}/{len(image_paths)}: {Path(path).name}")
        image = read_image(path)
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"Calibration images have mixed resolutions/orientations: {path}. Use identical camera settings.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # SB already provides subpixel corner positions.
            object_points.append(object_grid.copy())
            image_points.append(corners)
            used.append(str(Path(path).resolve()))
        else:
            rejected.append(str(Path(path).resolve()))
    if len(used) < 15:
        raise ValueError(f"Checkerboard found in only {len(used)} images; need at least 15. Check lighting, sharpness, and 9 x 6 INTERNAL corners.")
    if progress:
        progress(f"Fitting camera parameters using {len(used)} usable views...")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
    )
    errors = []
    for objects, observed, rotation, translation in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
        differences = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(differences ** 2, axis=1)))))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pattern_internal_corners": list(pattern_size),
        "square_size_m": square_size_m,
        "used_images": used,
        "rejected_images": rejected,
        "per_view_rms_px": errors,
        "opencv_version": cv2.__version__,
        "reminder": "Same lens, focus, zoom, orientation and resolution are required. Low RMS alone does not prove depth accuracy.",
    }
    return Calibration(matrix, distortion, image_size, float(rms), len(used), metadata)
