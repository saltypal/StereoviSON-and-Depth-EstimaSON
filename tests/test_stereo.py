import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from src.calibration import Calibration, calibrate_camera, load_calibration, save_calibration
from src.geometry import estimate_pair, reconstruct_corners
from src.markers import detect_marker, prepare_targets
from src.reporting import save_run, save_rejection, summarize_runs, TARGET_DISTANCES_M
from tests.fixtures import create_synthetic_pair, synthetic_calibration


def project(points, calibration, camera_x=0.0):
    relative = points - [camera_x, 0, 0]
    matrix = calibration.camera_matrix
    x = matrix[0, 0] * relative[:, 0] / relative[:, 2] + matrix[0, 2]
    y = matrix[1, 1] * relative[:, 1] / relative[:, 2] + matrix[1, 2]
    return np.column_stack((x, y))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.calibration = synthetic_calibration()
        self.points = np.array([[-.1, -.1, 2], [.1, -.1, 2], [.1, .1, 2], [-.1, .1, 2]])
        self.left = project(self.points, self.calibration)
        self.right = project(self.points, self.calibration, .2)

    def test_known_depth(self):
        result = reconstruct_corners(self.left, self.right, self.calibration, .2, 2.0)
        np.testing.assert_allclose(result.corners_xyz_m, self.points, atol=1e-12)
        self.assertAlmostEqual(result.depth_m, 2.0)
        self.assertAlmostEqual(result.distance_m, 2.0)
        self.assertEqual(result.to_dict()["measurement_error"]["percentage"], 0)

    def test_off_axis_range_and_tilted_marker(self):
        # Tilt a square around Y: corresponding corners have unequal depths.
        angle = .35
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        local = self.points - [0, 0, 2]
        points = local @ rotation.T + [.4, .3, 2.5]
        result = reconstruct_corners(project(points, self.calibration), project(points, self.calibration, .2), self.calibration, .2)
        np.testing.assert_allclose(result.centre_xyz_m, [.4, .3, 2.5], atol=1e-12)
        self.assertAlmostEqual(result.distance_m, np.sqrt(.4 ** 2 + .3 ** 2 + 2.5 ** 2))
        self.assertGreater(result.distance_m, result.depth_m)

    def test_different_horizontal_vertical_focal_lengths(self):
        self.calibration.camera_matrix[1, 1] = 1200
        result = reconstruct_corners(project(self.points, self.calibration), project(self.points, self.calibration, .2), self.calibration, .2)
        np.testing.assert_allclose(result.corners_xyz_m, self.points, atol=1e-12)

    def test_invalid_baselines_and_ground_truth(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(baseline=value), self.assertRaises(ValueError):
                reconstruct_corners(self.left, self.right, self.calibration, value)
            with self.subTest(ground_truth=value), self.assertRaises(ValueError):
                reconstruct_corners(self.left, self.right, self.calibration, .2, value)

    def test_reversed_zero_and_small_disparity(self):
        for right in (self.left, self.left + [100, 0], self.left - [9.99, 0]):
            with self.subTest(), self.assertRaises(ValueError):
                reconstruct_corners(self.left, right, self.calibration, .2)

    def test_vertical_mismatch_and_threshold_boundary(self):
        right = self.right.copy()
        right[0, 1] += 2.001
        with self.assertRaisesRegex(ValueError, "Vertical mismatch"):
            reconstruct_corners(self.left, right, self.calibration, .2)
        reconstruct_corners(self.left, self.left - [10, 2], self.calibration, .2)

    def test_bad_corner_shape_and_nonfinite_values(self):
        for corners in (np.zeros((3, 2)), np.full((4, 2), np.nan)):
            with self.subTest(), self.assertRaises(ValueError):
                reconstruct_corners(corners, self.right, self.calibration, .2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.left, self.right, self.calibration = create_synthetic_pair(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_detection_estimation_and_save(self):
        calibration = load_calibration(self.directory / "SYNTHETIC_calibration.json")
        result = estimate_pair(self.left, self.right, calibration, .2, 2.)
        self.assertAlmostEqual(result.distance_m, 2., places=3)
        np.testing.assert_allclose(result.disparities_px, [100] * 4, atol=.01)
        self.assertEqual(result.evidence_kind, "synthetic")
        saved = save_run(result, self.directory / "results")
        record = json.loads((saved / "result.json").read_text())
        self.assertEqual(record["calibration_id"], calibration.identity)
        self.assertEqual(record["evidence_kind"], "synthetic")
        self.assertTrue(record["input_sha256"]["left"])
        with Image.open(saved / "comparison.png") as image:
            self.assertEqual(image.width, 2000)
        self.assertTrue((saved / "left_original.png").is_file())
        self.assertEqual(summarize_runs(self.directory / "results")["physical_runs"], 0)

    def test_realistic_pipeline_reversed_images(self):
        with self.assertRaisesRegex(ValueError, "Nonpositive disparity"):
            estimate_pair(self.right, self.left, self.calibration, .2)

    def test_missing_marker_and_same_image(self):
        blank = self.directory / "blank.png"
        Image.fromarray(np.full((700, 1000), 255, np.uint8)).save(blank)
        with self.assertRaisesRegex(ValueError, "Left image"):
            estimate_pair(blank, self.right, self.calibration, .2)
        with self.assertRaisesRegex(ValueError, "different photographs"):
            estimate_pair(self.left, self.left, self.calibration, .2)

    def test_mismatched_size_and_unreadable_file(self):
        small = self.directory / "small.png"
        Image.new("RGB", (800, 600)).save(small)
        with self.assertRaisesRegex(ValueError, "resolution"):
            estimate_pair(small, self.right, self.calibration, .2)
        invalid = self.directory / "broken.jpg"
        invalid.write_text("Not an image")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            estimate_pair(invalid, self.right, self.calibration, .2)

    def test_duplicate_marker_rejected(self):
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 100)
        image = np.full((400, 600), 255, np.uint8)
        image[100:200, 100:200] = marker
        image[100:200, 350:450] = marker
        with self.assertRaisesRegex(ValueError, "found 2"):
            detect_marker(image)

    def test_calibration_roundtrip_and_quality_gate(self):
        save_calibration(self.calibration, self.directory / "camera.json")
        self.assertEqual(load_calibration(self.directory / "camera.json").identity, self.calibration.identity)
        self.calibration.rms_px = 1.01
        save_calibration(self.calibration, self.directory / "bad.json")
        with self.assertRaisesRegex(ValueError, "fails the gate"):
            load_calibration(self.directory / "bad.json")
        with self.assertRaises(ValueError):
            calibrate_camera([self.left] * 14)

    def test_malformed_calibration_json(self):
        for payload in ([], {"schema_version": 42}, {"schema_version": 1}):
            path = self.directory / "invalid.json"
            path.write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                load_calibration(path)

    def test_printable_target_dimensions_and_detection(self):
        paths = prepare_targets(self.directory / "targets")
        with Image.open(paths[0]) as marker:
            self.assertEqual(marker.size, (2100, 2970))
            self.assertAlmostEqual(marker.info["dpi"][0], 254, places=1)
            self.assertEqual(detect_marker(np.asarray(marker)).shape, (4, 2))
        with Image.open(paths[1]) as board:
            self.assertEqual(board.size, (2970, 2100))
            gray = np.asarray(board)
            found, corners = cv2.findChessboardCornersSB(gray, (9, 6))
            self.assertTrue(found)
            self.assertEqual(len(corners), 54)

    def test_calibration_fit_from_synthetic_projected_views(self):
        # Exercise real calibrateCamera using known projections. Detection is isolated here;
        # the printable checkerboard test separately exercises the actual detector.
        grid = np.zeros((54, 3), np.float32)
        grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * .025
        corners = []
        for i in range(18):
            rotation = np.array([.05 + .08 * (i % 4), -.22 + .10 * (i % 5), -.1 + .04 * (i % 6)])
            translation = np.array([-.10 + .015 * (i % 5), -.07 + .015 * (i % 4), .65 + .055 * i])
            projection, _ = cv2.projectPoints(grid, rotation, translation, self.calibration.camera_matrix, np.zeros(5))
            corners.append((True, projection.astype(np.float32)))
        with patch("src.calibration.cv2.findChessboardCornersSB", side_effect=corners):
            fitted = calibrate_camera([self.left] * 18)
        self.assertTrue(fitted.quality_passed)
        np.testing.assert_allclose(fitted.camera_matrix, self.calibration.camera_matrix, atol=.1)


class EvaluationTests(unittest.TestCase):
    def test_complete_physical_protocol_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Invented records test accounting ONLY. They never leave this temporary directory.
            for i, target in enumerate(TARGET_DISTANCES_M):
                for repeat in range(3):
                    run = directory / f"{i}_{repeat}"
                    run.mkdir()
                    data = {"outcome": "accepted", "evidence_kind": "physical_capture", "measured_distance_m": target,
                            "distance_m": target * 1.05, "input_sha256": {"left": f"left{i}{repeat}", "right": f"right{i}{repeat}"}}
                    (run / "result.json").write_text(json.dumps(data))
            summary = summarize_runs(directory)
            self.assertTrue(summary["full_range_target_met"])
            self.assertAlmostEqual(summary["ranges"]["near"]["mean_percentage_error"], 5)
            duplicate = directory / "z_duplicate"
            duplicate.mkdir()
            (duplicate / "result.json").write_text((directory / "0_0" / "result.json").read_text())
            summary = summarize_runs(directory)
            self.assertEqual(summary["physical_runs"], 18)
            self.assertEqual(len(summary["superseded_duplicate_runs"]), 1)
            record = json.loads((directory / "5_2" / "result.json").read_text())
            record["outcome"] = "rejected"
            record.pop("distance_m")
            record["reason"] = "Vertical mismatch"
            (directory / "5_2" / "result.json").write_text(json.dumps(record))
            summary = summarize_runs(directory)
            self.assertEqual(summary["rejected"], 1)
            self.assertTrue(summary["full_range_target_met"])
            (directory / "5_1" / "result.json").unlink()
            self.assertFalse(summarize_runs(directory)["full_range_target_met"])

    def test_rejection_log_and_empty_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            saved = save_rejection(directory, "bad alignment", "", "", .2, 1.)
            self.assertTrue((saved / "result.json").exists())
            summary = summarize_runs(directory)
            self.assertEqual(summary["rejected"], 1)
            self.assertFalse(summary["full_range_target_met"])
            self.assertIsNone(summary["ranges"]["near"]["mean_percentage_error"])


if __name__ == "__main__":
    unittest.main()
