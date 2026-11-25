"""Canonical side-view video and CSVs containing only the camera-side limbs.

Depth is not observable from monocular keypoints. For lateral footage, a dog
facing right exposes its anatomical right side; a left-facing dog exposes its
left side. Mirroring exchanges these labels. Confidence never selects a side.
"""

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

from .pipeline import _verify_output, draw_overlay, frame_rows
from .runtime import TrackingError
from .schema import CSV_FIELDS
from .tracking import DogTracker


CANVAS_SIZE = 640
DOG_SIZE = 480.0
TRAINING_FIELDS = CSV_FIELDS + ["source_limb", "normalized_dog_size_px"]
NEAR_LIMBS = {"front": "right-fore", "hind": "right-hind"}


class VideoQualityError(TrackingError):
    """A recording cannot supply enough trustworthy side-view observations."""


def heading_sign(points, scores, box, confidence):
    """Return +1/-1 only for a sufficiently horizontal, visible body axis."""
    if points is None:
        return 0

    def visible(indices):
        return all(np.isfinite(points[i]).all() and math.isfinite(float(scores[i]))
                   and scores[i] >= confidence for i in indices)

    if visible((3, 4)):  # AP-10K: neck and root of tail
        head, tail = points[3], points[4]
    elif visible((5, 8, 11, 14)):
        head = (points[5] + points[8]) / 2
        tail = (points[11] + points[14]) / 2
    else:
        return 0
    dx, dy = head - tail
    if abs(dx) < 0.25 * (box[2] - box[0]) or abs(dx) < 1.5 * abs(dy):
        return 0
    return 1 if dx > 0 else -1


def normalization_transform(box, flip, canvas_size=CANVAS_SIZE, dog_size=DOG_SIZE):
    """Keep aspect ratio; center the dog and scale its longest bbox side."""
    x1, y1, x2, y2 = box
    if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
        raise VideoQualityError("Nieprawidłowy obszar psa.")
    scale = dog_size / max(x2 - x1, y2 - y1)
    sx = -scale if flip else scale
    matrix = np.array([[sx, 0, canvas_size / 2 - sx * (x1 + x2) / 2],
                       [0, scale, canvas_size / 2 - scale * (y1 + y2) / 2]], np.float32)
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    normalized_box = (canvas_size / 2 - bw / 2, canvas_size / 2 - bh / 2,
                      canvas_size / 2 + bw / 2, canvas_size / 2 + bh / 2)
    return matrix, normalized_box


def _scan_video(video, scan_path, pose, detector, confidence, progress):
    cap = cv2.VideoCapture(str(video))
    tracker = DogTracker()
    headings = Counter()
    try:
        if not cap.isOpened():
            raise VideoQualityError(f"Nie można otworzyć nagrania: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoQualityError("Nagranie ma nieprawidłowy FPS; przekoduj je przed treningiem.")
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        expected = max(0, round(raw_count)) if math.isfinite(raw_count) else 0
        count, size = 0, None
        with scan_path.open("w", encoding="utf-8") as handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                current_size = frame.shape[:2]
                if size is not None and size != current_size:
                    raise VideoQualityError("Wymiary klatek zmieniają się w nagraniu.")
                size = current_size
                selected, status = tracker.select(detector.detect(frame))
                box, dog_score, heading = None, None, 0
                if selected is not None:
                    raw_box, raw_score = selected
                    box, dog_score = list(map(float, raw_box)), float(raw_score)
                    points, scores = pose.predict(frame, box)
                    heading = heading_sign(points, scores, box, confidence)
                    headings[heading] += 1
                    status = "detected" if heading else "not_side_view"
                handle.write(json.dumps({"box": box, "dog_score": dog_score,
                                         "heading": heading, "status": status}, allow_nan=False) + "\n")
                count += 1
                if progress:
                    progress("scan", count, expected)
        if not count or (expected and count < expected):
            raise VideoQualityError(f"Niepełne nagranie: odczytano {count} z {expected} klatek.")
        lateral_count = headings[1] + headings[-1]
        if not lateral_count:
            raise VideoQualityError("Nie wykryto wiarygodnego widoku psa z boku.")
        direction = 1 if headings[1] >= headings[-1] else -1
        if headings[direction] / lateral_count < 0.85:
            raise VideoQualityError("Pies zmienia kierunek; podziel film na przejścia w jednym kierunku.")
        return {"fps": fps, "frames": count, "source_size": list(reversed(size)),
                "direction": direction, "direction_counts": dict(headings)}
    finally:
        cap.release()


def normalize_and_track(video, output_dir, pose, detector, *, confidence=0.3,
                        min_valid_frames=20, min_coverage=0.5, progress=None):
    """Scan orientation, normalize pixels, then estimate pose in canonical frames.

    A single pose inference supplies BOTH near-side limbs. The far-side indices
    are never written to either CSV, including when a near limb is occluded.
    Invalid frames retain their original timestamps and blank coordinates.
    """
    video, output_dir = Path(video), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / ".scan.jsonl"
    scan = _scan_video(video, scan_path, pose, detector, confidence, progress)
    flip = scan["direction"] == -1
    source_side = "left" if flip else "right"
    fps, count = scan["fps"], scan["frames"]
    paths = {"normalized": output_dir / "normalized.mp4", "overlay": output_dir / "tracking.mp4",
             "front": output_dir / "front.csv", "hind": output_dir / "hind.csv"}
    valid_counts, statuses = Counter(), Counter()
    cap = cv2.VideoCapture(str(video))
    try:
        with ExitStack() as stack:
            writers = {}
            for key in ("normalized", "overlay"):
                writer = cv2.VideoWriter(str(paths[key]), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (CANVAS_SIZE, CANVAS_SIZE))
                stack.callback(writer.release)
                if not writer.isOpened():
                    raise TrackingError("Nie można zapisać MP4. Sprawdź kodek i miejsce na dysku.")
                writers[key] = writer
            csv_writers = {}
            for which in NEAR_LIMBS:
                handle = stack.enter_context(paths[which].open("w", encoding="utf-8-sig", newline=""))
                csv_writers[which] = csv.DictWriter(handle, fieldnames=TRAINING_FIELDS)
                csv_writers[which].writeheader()
            observations = stack.enter_context(scan_path.open(encoding="utf-8"))
            for index, line in enumerate(observations):
                observation = json.loads(line)
                ok, frame = cap.read()
                if not ok or frame is None or list(reversed(frame.shape[:2])) != scan["source_size"]:
                    raise VideoQualityError("Nagranie zmieniło się podczas przetwarzania.")
                box, points, scores = None, None, None
                status = observation["status"]
                canonical = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8)
                if observation["box"] is not None and observation["heading"] == scan["direction"]:
                    matrix, box = normalization_transform(observation["box"], flip)
                    canonical = cv2.warpAffine(frame, matrix, (CANVAS_SIZE, CANVAS_SIZE),
                                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                    points, scores = pose.predict(canonical, box)
                    if heading_sign(points, scores, box, confidence) != 1:
                        points, scores, status = None, None, "uncertain_normalized_pose"
                    else:
                        status = "detected"
                elif observation["box"] is not None and observation["heading"]:
                    status = "opposite_direction"
                statuses[status] += 1
                writers["normalized"].write(canonical)
                overlay = canonical
                for which, limb in NEAR_LIMBS.items():
                    rows = frame_rows(index, fps, limb, points, scores, CANVAS_SIZE, CANVAS_SIZE,
                                      confidence, box, observation["dog_score"], status)
                    for row in rows:
                        row["source_limb"] = source_side + ("-fore" if which == "front" else "-hind")
                        row["normalized_dog_size_px"] = DOG_SIZE
                    csv_writers[which].writerows(rows)
                    if all(row["status"] == "detected" for row in rows):
                        valid_counts[which] += 1
                    overlay = draw_overlay(overlay, rows, box, index, fps, limb, status)
                cv2.putText(overlay, "NEAR SIDE: front + hind", (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                writers["overlay"].write(overlay)
                if progress:
                    progress("track", index + 1, count)
        for key in ("normalized", "overlay"):
            _verify_output(paths[key], count, CANVAS_SIZE, CANVAS_SIZE)
        for which in NEAR_LIMBS:
            if valid_counts[which] < min_valid_frames or valid_counts[which] / count < min_coverage:
                raise VideoQualityError(
                    f"Za mało pewnych klatek {which}: {valid_counts[which]}/{count}. "
                    "Potrzebny jest wyraźny widok obu kończyn od strony kamery."
                )
        metadata = {
            **scan, "flipped": flip, "source_camera_side": source_side,
            "normalized_camera_side": "right", "normalized_direction": "right",
            "canvas_size": [CANVAS_SIZE, CANVAS_SIZE], "dog_longest_bbox_side_px": DOG_SIZE,
            "coordinate_system": "normalized pixels; origin top-left; x right, y down",
            "near_side_policy": "side-view geometry, fixed right AP-10K limbs after mirroring; no far-side fallback",
            "valid_frames": dict(valid_counts), "frame_status_counts": dict(statuses),
            "outputs": {key: path.name for key, path in paths.items()},
        }
        (output_dir / "normalization.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        scan_path.unlink()
        return metadata
    finally:
        cap.release()
