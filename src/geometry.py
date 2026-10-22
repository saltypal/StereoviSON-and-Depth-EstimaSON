"""Parallel-camera stereo geometry. Marker size never supplies metric scale."""

from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

from .calibration import Calibration, read_image
from .markers import detect_marker, MARKER_ID


@dataclass
class Estimate:
    baseline_m: float
    left_corners: np.ndarray
    right_corners: np.ndarray
    disparities_px: np.ndarray
    vertical_mismatch_px: np.ndarray
    corners_xyz_m: np.ndarray
    centre_xyz_m: np.ndarray
    depth_m: float
    distance_m: float
    measured_distance_m: float | None = None
    left_path: str = ""
    right_path: str = ""
    calibration_id: str = ""
    calibration_snapshot: dict = field(default_factory=dict)
    left_image: np.ndarray | None = field(default=None, repr=False)
    right_image: np.ndarray | None = field(default=None, repr=False)
    evidence_kind: str = "physical_capture"

    def to_dict(self) -> dict:
        error = None
        if self.measured_distance_m is not None:
            absolute = abs(self.distance_m - self.measured_distance_m)
            error = {"absolute_m": absolute, "percentage": 100 * absolute / self.measured_distance_m}
        return {
            "schema_version": 1, "outcome": "accepted", "evidence_kind": self.evidence_kind,
            "reference": "Left camera optical centre to centre of marked front surface",
            "baseline_m": self.baseline_m, "marker_id": MARKER_ID, "dictionary": "DICT_4X4_50",
            "left_corners_px": self.left_corners.tolist(), "right_corners_px": self.right_corners.tolist(),
            "disparities_px": self.disparities_px.tolist(),
            "vertical_mismatch_px": self.vertical_mismatch_px.tolist(),
            "corners_xyz_m": self.corners_xyz_m.tolist(), "centre_xyz_m": self.centre_xyz_m.tolist(),
            "depth_m": self.depth_m, "distance_m": self.distance_m,
            "measured_distance_m": self.measured_distance_m, "measurement_error": error,
            "left_path": self.left_path, "right_path": self.right_path,
            "calibration_id": self.calibration_id, "calibration": self.calibration_snapshot,
            "thresholds": {"maximum_vertical_mismatch_px": 2.0, "minimum_disparity_px": 10.0},
            "within_target_depth_range": 0.5 <= self.depth_m <= 3.0,
            "within_target_distance_range": 0.5 <= self.distance_m <= 3.0,
            "assumption": "Camera translated only along +X, with identical orientation and intrinsics. Vertical agreement does not prove this.",
        }


def reconstruct_corners(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    calibration: Calibration,
    baseline_m: float,
    measured_distance_m: float | None = None,
) -> Estimate:
    """Use undistorted, full-resolution pixel positions and baseline in metres."""
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("Camera separation must be a finite positive number.")
    if measured_distance_m is not None and (not np.isfinite(measured_distance_m) or measured_distance_m <= 0):
        raise ValueError("Measured distance must be a finite positive number, or left blank.")
    left = np.asarray(left_corners, dtype=np.float64)
    right = np.asarray(right_corners, dtype=np.float64)
    if left.shape != (4, 2) or right.shape != (4, 2) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Need four finite, corresponding (x, y) corners in each image.")
    disparities = left[:, 0] - right[:, 0]
    vertical = np.abs(left[:, 1] - right[:, 1])
    if np.any(disparities <= 0):
        raise ValueError("Nonpositive disparity: images may be reversed. Left photo must be from the left camera position; do not use absolute disparity.")
    if np.any(disparities < 10):
        raise ValueError("Disparity below 10 pixels: the pair is too uncertain. Check baseline, distance, and original image resolution.")
    if np.any(vertical > 2):
        raise ValueError(f"Vertical mismatch {vertical.max():.2f} px exceeds 2 px. Check the guide, camera rotation, height, and calibration; recapture.")
    matrix = calibration.camera_matrix
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    depth = fx * baseline_m / disparities
    x = (left[:, 0] - cx) * depth / fx
    y = (left[:, 1] - cy) * depth / fy
    xyz = np.column_stack((x, y, depth))
    if not np.isfinite(xyz).all():
        raise ValueError("Computed coordinates overflowed; check calibration and baseline.")
    centre = np.mean(xyz, axis=0)
    return Estimate(
        baseline_m=float(baseline_m), left_corners=left.copy(), right_corners=right.copy(),
        disparities_px=disparities, vertical_mismatch_px=vertical,
        corners_xyz_m=xyz, centre_xyz_m=centre, depth_m=float(centre[2]),
        distance_m=float(np.linalg.norm(centre)), measured_distance_m=measured_distance_m,
        calibration_id=calibration.identity, calibration_snapshot=calibration.to_dict(),
    )


def estimate_pair(
    left_path: str | Path,
    right_path: str | Path,
    calibration: Calibration,
    baseline_m: float,
    measured_distance_m: float | None = None,
) -> Estimate:
    if not calibration.quality_passed:
        raise ValueError("Calibration fails the 15-view / 1-pixel RMS quality gate.")
    if Path(left_path).resolve() == Path(right_path).resolve():
        raise ValueError("Select two different photographs.")
    images = []
    for label, path in (("Left", left_path), ("Right", right_path)):
        image = read_image(path)
        if (image.shape[1], image.shape[0]) != calibration.image_size:
            raise ValueError(f"{label} image resolution differs from calibration {calibration.image_size}. Use original images with matching orientation/settings.")
        # Preserve K and the entire frame. Cropping/resizing would change pixel geometry.
        corrected = cv2.undistort(image, calibration.camera_matrix, calibration.distortion, None, calibration.camera_matrix)
        images.append(corrected)
    try:
        left_corners = detect_marker(images[0])
    except ValueError as error:
        raise ValueError(f"Left image: {error}") from error
    try:
        right_corners = detect_marker(images[1])
    except ValueError as error:
        raise ValueError(f"Right image: {error}") from error
    result = reconstruct_corners(left_corners, right_corners, calibration, baseline_m, measured_distance_m)
    result.left_path = str(Path(left_path).resolve())
    result.right_path = str(Path(right_path).resolve())
    result.left_image, result.right_image = images
    if calibration.metadata.get("evidence_kind") == "synthetic":
        result.evidence_kind = "synthetic"
    return result
