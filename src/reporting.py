"""Save self-contained run records and summarize measured physical trials."""

from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import shutil
import uuid

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from .geometry import Estimate


TARGET_DISTANCES_M = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def annotated_views(result: Estimate) -> tuple[np.ndarray, np.ndarray]:
    """Draw corrected pixel coordinates onto the full-resolution corrected images."""
    views = []
    for image, corners in ((result.left_image, result.left_corners), (result.right_image, result.right_corners)):
        if image is None:
            raise ValueError("No source image is attached to this result.")
        view = image.copy()
        thickness = max(2, image.shape[1] // 800)
        text_scale = max(0.6, image.shape[1] / 2000)
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(view, [points], True, (70, 210, 70), thickness, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(view, tuple(point), thickness * 3, (0, 190, 255), -1, cv2.LINE_AA)
            cv2.putText(view, str(index + 1), tuple(point + [12, -12]), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 100, 230), thickness, cv2.LINE_AA)
        views.append(view)
    return views[0], views[1]


def comparison_image(result: Estimate) -> np.ndarray:
    left, right = annotated_views(result)
    height, width = left.shape[:2]
    panel_height = max(180, width // 6)
    canvas = np.full((height + panel_height, 2 * width, 3), 245, dtype=np.uint8)
    canvas[panel_height:, :width] = left
    canvas[panel_height:, width:] = right
    scale = max(0.55, width / 1700)
    thickness = max(1, width // 1000)
    line_height = panel_height // 5
    lines = [
        f"{result.evidence_kind.upper()} | Left / Right | Baseline {result.baseline_m * 100:.2f} cm",
        f"Depth Z: {result.depth_m:.4f} m | Distance to marker centre: {result.distance_m:.4f} m",
        "Disparities (px): " + ", ".join(f"{d:.2f}" for d in result.disparities_px),
        f"Max vertical mismatch: {result.vertical_mismatch_px.max():.2f} px",
    ]
    if result.measured_distance_m is not None:
        error = result.to_dict()["measurement_error"]
        lines[-1] += f" | Measured {result.measured_distance_m:.4f} m | Error {error['percentage']:.2f}%"
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (20, (index + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (45, 45, 45), thickness, cv2.LINE_AA)
    return canvas


def new_run_directory(directory: str | Path) -> Path:
    directory = Path(directory).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    run_directory = directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def file_hash(path: str) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_record(record: dict, directory: Path) -> None:
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    record["run_id"] = directory.name
    record["environment"] = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version, "platform": platform.platform(), "backend": "CPU"}
    record["input_sha256"] = {side: file_hash(record.get(f"{side}_path", "")) for side in ("left", "right")}
    (directory / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def save_run(result: Estimate, output_directory: str | Path) -> Path:
    directory = new_run_directory(output_directory)
    comparison = comparison_image(result)
    Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)).save(directory / "comparison.png")
    # Preserve original photographs alongside the corrected annotated view.
    for side, path in (("left", result.left_path), ("right", result.right_path)):
        if path and Path(path).is_file():
            shutil.copy2(path, directory / f"{side}_original{Path(path).suffix.lower()}")
    write_record(result.to_dict(), directory)
    return directory


def save_rejection(
    output_directory: str | Path, reason: str, left_path: str, right_path: str,
    baseline_m: float | None, measured_distance_m: float | None,
    calibration: dict | None = None,
) -> Path:
    directory = new_run_directory(output_directory)
    record = {
        "schema_version": 1, "outcome": "rejected",
        "evidence_kind": "synthetic" if calibration and calibration.get("metadata", {}).get("evidence_kind") == "synthetic" else "physical_capture",
        "reason": reason, "left_path": left_path, "right_path": right_path,
        "baseline_m": baseline_m, "measured_distance_m": measured_distance_m,
        "calibration": calibration,
        "calibration_id": hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest() if calibration else None,
    }
    write_record(record, directory)
    return directory


def summarize_runs(results_directory: str | Path) -> dict:
    """Count failures too; simulated evidence never contributes to physical accuracy."""
    directory = Path(results_directory).resolve()
    rows, invalid, superseded = [], [], []
    capture_indices = {}
    for path in sorted(directory.glob("*/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("evidence_kind") != "physical_capture":
                continue
            measured = record.get("measured_distance_m")
            estimated = record.get("distance_m")
            outcome = record.get("outcome")
            if outcome not in ("accepted", "rejected"):
                raise ValueError("Unknown outcome")
            if measured is not None and (not isinstance(measured, (int, float)) or not np.isfinite(measured) or measured <= 0):
                raise ValueError("Invalid measured distance")
            if outcome == "accepted" and (not isinstance(estimated, (int, float)) or not np.isfinite(estimated) or estimated <= 0):
                raise ValueError("Invalid estimated distance")
            error = abs(estimated - measured) if outcome == "accepted" and measured is not None else None
            hashes = record.get("input_sha256", {})
            verified_identity = bool(hashes.get("left") and hashes.get("right"))
            row = {
                "run_id": record.get("run_id", path.parent.name), "outcome": outcome,
                "measured_distance_m": measured, "distance_m": estimated,
                "absolute_error_m": error,
                "percentage_error": 100 * error / measured if error is not None else None,
                "reason": record.get("reason", ""),
                "verified_capture_identity": verified_identity,
            }
            if verified_identity:
                # Processing the same two photographs again is not another physical capture.
                identity = tuple(sorted((hashes["left"], hashes["right"])))
                if identity in capture_indices:
                    index = capture_indices[identity]
                    superseded.append(rows[index]["run_id"])
                    rows[index] = row
                else:
                    capture_indices[identity] = len(rows)
                    rows.append(row)
            else:
                rows.append(row)
        except (OSError, ValueError, TypeError) as error:
            invalid.append({"file": str(path), "reason": str(error)})
    bins = []
    for target in TARGET_DISTANCES_M:
        # Assign a capture only when its entered ruler measurement is within 1 cm of the target.
        trials = [r for r in rows if r["verified_capture_identity"] and r["measured_distance_m"] is not None and abs(r["measured_distance_m"] - target) <= 0.0100001]
        accepted = [r for r in trials if r["outcome"] == "accepted"]
        distances = [r["distance_m"] for r in accepted]
        bins.append({
            "target_m": target, "attempts": len(trials), "accepted": len(accepted),
            "rejected": len(trials) - len(accepted),
            "mean_absolute_error_m": float(np.mean([r["absolute_error_m"] for r in accepted])) if accepted else None,
            "mean_percentage_error": float(np.mean([r["percentage_error"] for r in accepted])) if accepted else None,
            "repeatability_sample_std_m": float(np.std(distances, ddof=1)) if len(distances) >= 2 else None,
        })
    ranges = {}
    for label, targets in (("near", TARGET_DISTANCES_M[:3]), ("far", TARGET_DISTANCES_M[3:])):
        values = [b for b in bins if b["target_m"] in targets]
        complete = all(b["attempts"] >= 3 and b["accepted"] >= 2 for b in values)
        # Give each test distance equal weight so many repetitions at one distance cannot dominate.
        mape = float(np.mean([b["mean_percentage_error"] for b in values])) if all(b["accepted"] for b in values) else None
        ranges[label] = {"mean_percentage_error": mape, "coverage_complete": complete, "passes_target": complete and mape is not None and mape <= 10}
    summary = {
        "physical_runs": len(rows), "accepted": sum(r["outcome"] == "accepted" for r in rows),
        "rejected": sum(r["outcome"] == "rejected" for r in rows),
        "runs_without_ground_truth": sum(r["measured_distance_m"] is None for r in rows),
        "distances": bins, "ranges": ranges,
        "full_range_target_met": all(r["passes_target"] for r in ranges.values()) and not invalid,
        "invalid_records": invalid,
        "superseded_duplicate_runs": superseded,
        "runs_without_capture_identity": sum(not r["verified_capture_identity"] for r in rows),
        "notes": "Physical captures only. 3 attempts and >=2 accepted per target; equal-weight distance MAPE <=10% in near and far ranges. Duplicate image pairs use their latest record. Unverifiable image pairs do not count toward coverage.",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (directory / "evaluation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "outcome", "measured_distance_m", "distance_m", "absolute_error_m", "percentage_error", "reason", "verified_capture_identity"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Physical stereo evaluation", "", "Synthetic runs are excluded; repeated processing of identical photographs counts once.", "",
             f"**Full 0.5–3 m target: {'MET' if summary['full_range_target_met'] else 'NOT YET DEMONSTRATED'}**", "",
             f"Unique physical pairs: {len(rows)}. Accepted: {summary['accepted']}. Rejected: {summary['rejected']}.", "",
             "| Target (m) | Attempts | Accepted | Rejected | Mean error (m) | Mean error (%) | Repeatability std (m) |",
             "|---|---|---|---|---|---|---|"]
    for item in bins:
        def display(value):
            return f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {item['target_m']:g} | {item['attempts']} | {item['accepted']} | {item['rejected']} | {display(item['mean_absolute_error_m'])} | {display(item['mean_percentage_error'])} | {display(item['repeatability_sample_std_m'])} |")
    lines += ["", "Mean percentage error gives each distance equal weight. Repeatability is sample standard deviation of estimates at each distance.", ""]
    for label, item in ranges.items():
        mape = f"{item['mean_percentage_error']:.2f}%" if item["mean_percentage_error"] is not None else "unavailable"
        lines.append(f"- {label.title()} range: error {mape}; coverage {'complete' if item['coverage_complete'] else 'incomplete'}; target {'met' if item['passes_target'] else 'not demonstrated'}.")
    lines += ["", f"Superseded duplicate runs: {len(superseded)}. Runs without ground truth: {summary['runs_without_ground_truth']}. Runs without verifiable image identity: {summary['runs_without_capture_identity']}.",
              "", "Three independent captures at every target distance and at least two accepted per distance are required. Mean percentage error must be <=10% for both ranges. Read the CSV for individual failures and check capture conditions before interpreting the scores."]
    (directory / "evaluation_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
