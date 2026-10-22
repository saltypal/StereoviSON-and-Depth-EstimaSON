"""Exercise real Tk widgets and the asynchronous pipeline without opening a window."""

from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from app import StereoWindow
from tests.fixtures import create_synthetic_pair


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.left, self.right, calibration = create_synthetic_pair(self.directory)
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = StereoWindow(self.root)
        self.window.calibration = calibration
        self.window.paths["left"].set(str(self.left))
        self.window.paths["right"].set(str(self.right))
        self.window.measured.set("2")

    def tearDown(self):
        self.window.close()
        self.temp.cleanup()

    def wait_for_processing(self):
        deadline = time.monotonic() + 20
        while self.window.estimate_button.instate(["disabled"]) and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.window.estimate_button.instate(["disabled"]), "Asynchronous desktop estimate timed out")

    def test_success_then_input_change_clears_previous_result(self):
        self.window.start_estimate()
        self.wait_for_processing()
        self.assertIsNotNone(self.window.result)
        self.assertAlmostEqual(self.window.result.distance_m, 2, places=3)
        self.assertIn("SYNTHETIC", self.window.status.get())
        self.assertFalse(self.window.save_button.instate(["disabled"]))
        self.assertEqual(len(self.window.preview_images), 2)
        self.window.baseline.set("21")
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.metrics.get(), "No estimate yet")
        self.assertTrue(self.window.save_button.instate(["disabled"]))

    def test_failed_estimate_clears_result_and_records_rejection(self):
        self.window.start_estimate()
        self.wait_for_processing()
        self.window.paths["left"].set(str(self.right))
        self.window.paths["right"].set(str(self.left))
        with patch("app.messagebox.showerror") as dialog, patch("app.save_rejection", return_value=self.directory / "rejected") as rejected:
            self.window.start_estimate()
            self.wait_for_processing()
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.save_button.instate(["disabled"]))
        self.assertEqual(self.window.preview_images, [])
        dialog.assert_called_once()
        rejected.assert_called_once()

    def test_invalid_form_does_not_start_a_capture(self):
        self.window.baseline.set("nan")
        with patch("app.messagebox.showerror") as dialog:
            self.window.start_estimate()
        self.assertIsNone(self.window.future)
        self.assertIsNone(self.window.result)
        dialog.assert_called_once()


if __name__ == "__main__":
    unittest.main()
