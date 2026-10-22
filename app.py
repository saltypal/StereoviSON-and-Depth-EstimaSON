"""Desktop demonstration and small preparation/calibration commands."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.calibration import IMAGE_EXTENSIONS, calibrate_camera, load_calibration, save_calibration
from src.geometry import estimate_pair
from src.markers import prepare_targets
from src.reporting import annotated_views, save_run, save_rejection, summarize_runs


PROJECT_DIRECTORY = Path(__file__).resolve().parent


def positive_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return number


class StereoWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stereo Vision · Object Distance")
        self.root.geometry("1160x850")
        self.root.minsize(960, 740)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.result = None
        self.calibration = None
        self.preview_images = []
        self.closed = False
        self.paths = {name: tk.StringVar() for name in ("calibration", "left", "right")}
        self.baseline = tk.StringVar(value="20")
        self.measured = tk.StringVar()
        self.status = tk.StringVar(value="Load a real camera calibration and two original photographs to begin.")
        self.metrics = tk.StringVar(value="No estimate yet")
        self.disparities = tk.StringVar()
        self.settings = ttk.Frame(root, padding=(22, 14))
        self.settings.pack(fill="x")
        ttk.Label(self.settings, text="Stereo Vision", font=("Segoe UI", 22, "bold")).grid(row=0, column=0, sticky="w", columnspan=3)
        ttk.Label(self.settings, text="One fixed object. Two horizontal camera positions. Target range: 0.5–3 m.").grid(row=1, column=0, sticky="w", columnspan=3, pady=(2, 14))
        self.select_buttons = []
        for row, (name, label) in enumerate((("calibration", "Camera calibration"), ("left", "Left photograph"), ("right", "Right photograph")), start=2):
            ttk.Label(self.settings, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            ttk.Entry(self.settings, textvariable=self.paths[name], state="readonly").grid(row=row, column=1, sticky="ew", pady=4)
            button = ttk.Button(self.settings, text="Choose…", command=lambda selected=name: self.select_file(selected))
            button.grid(row=row, column=2, padx=(8, 0), pady=4)
            self.select_buttons.append(button)
        self.settings.columnconfigure(1, weight=1)
        fields = ttk.Frame(self.settings)
        fields.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(fields, text="Camera separation (cm)").pack(side="left")
        self.baseline_entry = ttk.Entry(fields, textvariable=self.baseline, width=10)
        self.baseline_entry.pack(side="left", padx=(8, 24))
        ttk.Label(fields, text="Measured distance to marker centre (m), optional").pack(side="left")
        self.measured_entry = ttk.Entry(fields, textvariable=self.measured, width=10)
        self.measured_entry.pack(side="left", padx=8)
        self.estimate_button = ttk.Button(fields, text="Estimate", command=self.start_estimate)
        self.estimate_button.pack(side="right")
        previews = ttk.Frame(root, padding=(22, 0))
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="LEFT POSITION · undistorted view", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, pady=6)
        ttk.Label(previews, text="RIGHT POSITION · undistorted view", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, pady=6)
        self.canvases = []
        for column in range(2):
            canvas = tk.Canvas(previews, background="#edf1f5", highlightthickness=0, width=520, height=300)
            canvas.grid(row=1, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            canvas.bind("<Configure>", self.refresh_previews)
            self.canvases.append(canvas)
        footer = ttk.Frame(root, padding=(22, 14))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.metrics, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(footer, textvariable=self.disparities).pack(anchor="w", pady=(4, 8))
        buttons = ttk.Frame(footer)
        buttons.pack(fill="x")
        self.save_button = ttk.Button(buttons, text="Save Result", state="disabled", command=self.save_current)
        self.save_button.pack(side="left")
        self.summary_button = ttk.Button(buttons, text="Summarize Physical Trials", command=self.summarize)
        self.summary_button.pack(side="left", padx=8)
        ttk.Label(footer, textvariable=self.status, wraplength=1080).pack(anchor="w", pady=(10, 0))
        ttk.Label(footer, text="Keep the phone orientation fixed on a straight guide. Vertical agreement alone cannot prove correct alignment.", wraplength=1080).pack(anchor="w", pady=(6, 0))
        self.baseline.trace_add("write", self.invalidate)
        self.measured.trace_add("write", self.invalidate)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def invalidate(self, *args):
        self.result = None
        self.preview_images = []
        self.metrics.set("No estimate yet")
        self.disparities.set("")
        self.save_button.configure(state="disabled")
        for canvas in self.canvases:
            canvas.delete("all")

    def select_file(self, name):
        types = [("Calibration JSON", "*.json")] if name == "calibration" else [("Photographs", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        filename = filedialog.askopenfilename(title=f"Choose {name}", initialdir=PROJECT_DIRECTORY / "data", filetypes=types)
        if not filename:
            return
        self.invalidate()
        if name == "calibration":
            try:
                self.calibration = load_calibration(filename)
            except (ValueError, cv2.error) as error:
                self.calibration = None
                self.paths[name].set("")
                self.status.set(str(error))
                messagebox.showerror("Calibration not accepted", str(error))
                return
            self.status.set(f"Calibration accepted: {self.calibration.usable_views} views, RMS {self.calibration.rms_px:.3f} px, resolution {self.calibration.image_size}.")
            if self.calibration.metadata.get("evidence_kind") == "synthetic":
                self.status.set("SYNTHETIC calibration selected: use only the supplied demo images, never your phone photographs.")
        else:
            self.status.set("Image selected. Check that both views use the same camera settings.")
        self.paths[name].set(filename)

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        for widget in self.select_buttons + [self.baseline_entry, self.measured_entry, self.estimate_button, self.summary_button]:
            widget.configure(state=state)

    def start_estimate(self):
        self.invalidate()
        try:
            if self.calibration is None:
                raise ValueError("Choose a valid camera calibration first.")
            if not self.paths["left"].get() or not self.paths["right"].get():
                raise ValueError("Choose both photographs.")
            baseline_m = positive_number(self.baseline.get(), "Camera separation") / 100
            measured_m = positive_number(self.measured.get(), "Measured distance") if self.measured.get().strip() else None
        except ValueError as error:
            self.status.set(str(error))
            messagebox.showerror("Check inputs", str(error))
            return
        self.capture_context = {
            "left_path": self.paths["left"].get(), "right_path": self.paths["right"].get(),
            "baseline_m": baseline_m, "measured_distance_m": measured_m,
            "calibration": self.calibration.to_dict(),
        }
        self.set_busy(True)
        self.status.set("Processing full-resolution photographs…")
        self.future = self.executor.submit(estimate_pair, self.capture_context["left_path"], self.capture_context["right_path"], self.calibration, baseline_m, measured_m)
        self.root.after(100, self.poll_estimate)

    def poll_estimate(self):
        if self.closed:
            return
        if not self.future.done():
            self.root.after(100, self.poll_estimate)
            return
        self.set_busy(False)
        try:
            result = self.future.result()
        except Exception as error:
            self.invalidate()
            reason = str(error)
            if isinstance(error, (ValueError, cv2.error, OSError)):
                try:
                    saved = save_rejection(PROJECT_DIRECTORY / "results", reason, **self.capture_context)
                    self.status.set(f"Pair rejected: {reason} Rejection recorded in {saved.name}.")
                except Exception as save_error:
                    self.status.set(f"Pair rejected: {reason}. Could not record rejection: {save_error}")
            else:
                self.status.set(f"Unexpected processing error: {reason}")
            messagebox.showerror("No valid distance estimate", reason)
            return
        self.result = result
        self.preview_images = list(annotated_views(result))
        self.refresh_previews()
        self.metrics.set(f"Distance: {result.distance_m:.3f} m     |     Depth Z: {result.depth_m:.3f} m")
        details = "Disparities: " + ", ".join(f"{d:.2f}" for d in result.disparities_px) + f" px  ·  Max vertical mismatch: {result.vertical_mismatch_px.max():.2f} px"
        if result.measured_distance_m is not None:
            error = result.to_dict()["measurement_error"]
            details += f"  ·  Error: {error['absolute_m']:.3f} m ({error['percentage']:.2f}%)"
        self.disparities.set(details)
        self.save_button.configure(state="normal")
        warning = " Outside the planned 0.5–3 m distance range." if not 0.5 <= result.distance_m <= 3 else ""
        evidence_note = "SYNTHETIC TEST ONLY; excluded from physical accuracy evaluation." if result.evidence_kind == "synthetic" else "Pair passed numerical checks. Save this capture once; physical accuracy still requires ruler measurements."
        self.status.set(evidence_note + warning)

    def refresh_previews(self, event=None):
        if not self.preview_images:
            return
        self.tk_images = []
        for canvas, image in zip(self.canvases, self.preview_images):
            picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            picture.thumbnail((max(1, canvas.winfo_width() - 16), max(1, canvas.winfo_height() - 16)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(picture)
            self.tk_images.append(photo)
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() / 2, canvas.winfo_height() / 2, image=photo)

    def save_current(self):
        if self.result is None:
            return
        try:
            directory = save_run(self.result, PROJECT_DIRECTORY / "results")
        except Exception as error:
            messagebox.showerror("Could not save", str(error))
            return
        self.save_button.configure(state="disabled")
        self.status.set(f"Saved original photos, annotated comparison and result.json to {directory}")

    def summarize(self):
        try:
            summary = summarize_runs(PROJECT_DIRECTORY / "results")
        except Exception as error:
            messagebox.showerror("Could not summarize", str(error))
            return
        status = "TARGET MET" if summary["full_range_target_met"] else "TARGET NOT YET DEMONSTRATED"
        messagebox.showinfo("Physical evaluation", f"{status}\nAccepted: {summary['accepted']}\nRejected: {summary['rejected']}\nSee evaluation_summary.json and evaluation_runs.csv in results.")

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stereo distance assignment: prepare targets, calibrate, or open the desktop app.")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--prepare-targets", action="store_true")
    commands.add_argument("--calibrate", type=Path, metavar="IMAGE_DIRECTORY")
    commands.add_argument("--summarize", action="store_true")
    parser.add_argument("--square-mm", type=float, default=25)
    parser.add_argument("--output", type=Path, help="Calibration JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.prepare_targets:
            for path in prepare_targets(PROJECT_DIRECTORY / "data" / "targets"):
                print(path)
        elif args.calibrate:
            if not args.calibrate.is_dir():
                raise ValueError("Calibration image directory does not exist.")
            paths = sorted(p for p in args.calibrate.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
            calibration = calibrate_camera(paths, square_size_m=args.square_mm / 1000, progress=print)
            path = save_calibration(calibration, args.output or PROJECT_DIRECTORY / "data" / "camera_calibration.json")
            print(f"Calibration saved: {path}\nUsable views: {calibration.usable_views}\nRMS: {calibration.rms_px:.4f} px")
            if not calibration.quality_passed:
                print("QUALITY GATE FAILED. Saved for diagnosis; the app will not accept this calibration.", file=sys.stderr)
                return 1
            print("Numerical quality gate passed. Keep camera settings fixed and validate with physical distances.")
        elif args.summarize:
            summary = summarize_runs(PROJECT_DIRECTORY / "results")
            print(f"Physical captures: {summary['physical_runs']}, accepted: {summary['accepted']}, rejected: {summary['rejected']}")
            print("Full-range target met:", summary["full_range_target_met"])
        else:
            root = tk.Tk()
            style = ttk.Style(root)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            StereoWindow(root)
            root.mainloop()
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
