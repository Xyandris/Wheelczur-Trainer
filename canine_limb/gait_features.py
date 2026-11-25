"""Temporal gait features from normalized, near-side tracker CSVs."""

from pathlib import Path
import csv
import math

import numpy as np

from .schema import LIMBS
from .training_video import VideoQualityError


FEATURE_VERSION = 1


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite number")
    return result


def _read_track(path, which_limb):
    if which_limb not in {"front", "hind"}:
        raise ValueError("which_limb musi mieć wartość front albo hind.")
    limb = "right-fore" if which_limb == "front" else "right-hind"
    joints = LIMBS[limb][1]
    frames = {}
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"frame", "time_s", "limb", "which_limb", "joint", "status",
                        "x_px", "y_px", "normalized_dog_size_px"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("wymagane są CSV z normalizacji treningowej, wraz z which_limb")
            for row in reader:
                if row["which_limb"] != which_limb or row["limb"] != limb or row["joint"] not in joints:
                    raise ValueError("plik zawiera inną kończynę lub stronę niż wybrana")
                index, time_s = int(row["frame"]), _number(row["time_s"])
                scale = _number(row["normalized_dog_size_px"])
                if index < 0 or time_s < 0 or scale <= 0:
                    raise ValueError("nieprawidłowy czas, numer klatki lub skala")
                entry = frames.setdefault(index, {"time": time_s, "joints": {}})
                if entry["time"] != time_s or row["joint"] in entry["joints"]:
                    raise ValueError("powtórzony punkt albo niespójny czas klatki")
                xy = None
                if row["status"] == "detected":
                    xy = np.array([_number(row["x_px"]), _number(row["y_px"])]) / scale
                entry["joints"][row["joint"]] = xy
        ordered = [frames[i] for i in sorted(frames)]
        times = np.array([entry["time"] for entry in ordered])
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("brak rosnącej osi czasu")
        valid_times, coordinates = [], []
        for entry in ordered:
            if set(entry["joints"]) != set(joints):
                continue
            points = [entry["joints"][joint] for joint in joints]
            if any(point is None for point in points):
                continue
            points = np.array(points)
            if np.min(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 1e-5:
                continue
            valid_times.append(entry["time"])
            coordinates.append(points)
        return times, np.array(valid_times), np.array(coordinates)
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoQualityError(f"Nieprawidłowy CSV {path}: {exc}") from exc


def _features(times, points, sample_rate):
    # Body translation and absolute canvas position do not enter the classifier.
    relative = points[:, 1:] - points[:, :1]
    a, b = points[:, 0] - points[:, 1], points[:, 2] - points[:, 1]
    angle = np.degrees(np.arccos(np.clip(
        np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1)))
    channels = {
        "mid_x": relative[:, 0, 0], "mid_y": relative[:, 0, 1],
        "paw_x": relative[:, 1, 0], "paw_y": relative[:, 1, 1],
        "joint_angle": angle,
        "extension": np.linalg.norm(relative[:, 1], axis=1),
        "upper_length": np.linalg.norm(a, axis=1), "lower_length": np.linalg.norm(b, axis=1),
    }
    grid = np.linspace(times[0], times[-1], max(3, round((times[-1] - times[0]) * sample_rate) + 1))
    signals = {name: np.interp(grid, times, signal) for name, signal in channels.items()}
    # Smooth before differentiation, without connecting long tracking gaps.
    for name in ("paw_x", "paw_y", "joint_angle"):
        padded = np.pad(signals[name], (1, 1), mode="edge")
        smoothed = np.convolve(padded, np.ones(3) / 3, mode="valid")
        signals[name + "_velocity"] = np.gradient(smoothed, grid)
    features = {}
    for name, signal in signals.items():
        for statistic, value in (("mean", np.mean(signal)), ("std", np.std(signal)),
                                 ("p10", np.percentile(signal, 10)), ("p90", np.percentile(signal, 90))):
            features[name + "_" + statistic] = float(value)
    signal = signals["paw_x"] - np.mean(signals["paw_x"])
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=float(grid[1] - grid[0]))
    band = (frequencies >= 0.3) & (frequencies <= 5.0)
    if band.any() and float(np.sum(power[band])) > 1e-10:
        peak = np.flatnonzero(band)[np.argmax(power[band])]
        features["paw_frequency_hz"] = float(frequencies[peak])
        features["paw_periodicity"] = float(power[peak] / np.sum(power[band]))
    else:
        features["paw_frequency_hz"] = features["paw_periodicity"] = 0.0
    return features


def feature_windows(path, which_limb, *, window_seconds=2.0, stride_seconds=1.0,
                    sample_rate=30, min_valid_frames=20, min_coverage=0.5):
    """Keep missing points missing; interpolate only gaps up to 150 ms.

    Resampling makes derivative and frequency features comparable across FPS.
    All windows from one source/group must remain together during validation.
    """
    if (not all(math.isfinite(v) for v in (window_seconds, stride_seconds, sample_rate, min_coverage))
            or window_seconds < 0.5 or stride_seconds <= 0 or sample_rate < 5
            or min_valid_frames < 3 or not 0 < min_coverage <= 1):
        raise ValueError("Nieprawidłowe ustawienia okien treningowych.")
    all_times, times, points = _read_track(path, which_limb)
    if len(times) < min_valid_frames:
        raise VideoQualityError(f"Za mało kompletnych klatek {which_limb} w {path}.")
    duration = all_times[-1] - all_times[0]
    if duration < 0.5:
        raise VideoQualityError("Nagranie jest krótsze niż 0,5 sekundy.")
    length = min(window_seconds, duration)
    starts = list(np.arange(all_times[0], all_times[-1] - length + 1e-7, stride_seconds))
    tail_start = all_times[-1] - length
    if tail_start - starts[-1] > stride_seconds / 2:
        starts.append(tail_start)
    windows = []
    for start in starts:
        end = start + length
        selected = (times >= start - 1e-7) & (times <= end + 1e-7)
        expected = int(np.count_nonzero((all_times >= start - 1e-7) & (all_times <= end + 1e-7)))
        t, p = times[selected], points[selected]
        if len(t) < min_valid_frames or len(t) / max(1, expected) < min_coverage:
            continue
        if t[-1] - t[0] < 0.5 or np.max(np.diff(t)) > 0.15 + 1e-7:
            continue
        features = _features(t, p, sample_rate)
        if not np.isfinite(list(features.values())).all():
            continue
        windows.append({"start_s": float(start), "end_s": float(end), "features": features})
    if not windows:
        raise VideoQualityError(f"Brak ciągłego, wiarygodnego okna ruchu {which_limb} w {path}.")
    return windows
