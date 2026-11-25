"""Training integration: real video encoding and fitting, deterministic pose doubles."""

from dataclasses import replace
from pathlib import Path
import csv
import json

import cv2
import joblib
import numpy as np
import pytest

import launcher
from canine_limb import training
from canine_limb.gait_features import feature_windows
from canine_limb.pipeline import frame_rows
from canine_limb.runtime import TrackingError
from canine_limb.schema import LIMBS
from canine_limb.training import TrainingConfig, discover_videos, predict_csv, run_training
from canine_limb.training_video import (
    DOG_SIZE, TRAINING_FIELDS, VideoQualityError, heading_sign,
    normalization_transform, normalize_and_track,
)


@pytest.fixture
def training_video(tmp_path):
    def create(path=None, count=66, fps=30, left=False):
        path = path or tmp_path / "przejście psa.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
        assert writer.isOpened()
        for _ in range(count):
            writer.write(np.full((240, 320, 3), 130 if left else 30, np.uint8))
        writer.release()
        return path
    return create


@pytest.fixture
def training_models(monkeypatch):
    class Pose:
        info = {"device": "test"}
        missing_near = False

        def __init__(self, *args):
            self.calls = self.canonical_calls = 0

        def predict(self, frame, box):
            self.calls += 1
            points, scores = np.full((17, 2), 100.0), np.ones(17)
            if frame.shape[1] == 640:
                self.canonical_calls += 1
                phase = self.canonical_calls / 30 * 2 * np.pi * 1.5
                points[3], points[4] = (440, 210), (200, 210)
                points[[8, 9, 10]] = [(410, 230), (400 + 12 * np.sin(phase), 320),
                                         (410 + 35 * np.sin(phase), 450 - 8 * np.cos(phase))]
                points[[14, 15, 16]] = [(230, 240), (240 + 15 * np.sin(phase), 340),
                                           (240 + 30 * np.sin(phase), 450 - 8 * np.cos(phase))]
                # Distinct far-side coordinates must never enter the CSVs.
                points[[5, 6, 7, 11, 12, 13]] = (91, 92)
                if self.missing_near and self.canonical_calls == 5:
                    scores[10] = 0.01
            else:
                points[3], points[4] = (210, 80), (90, 80)
                if frame[0, 0, 0] > 80:
                    points[:, 0] = 319 - points[:, 0]
            return points, scores

    class Detector:
        info = {"device": "test"}

        def __init__(self, *args):
            pass

        def detect(self, frame):
            return [((50, 40, 270, 200), 0.95)]

    monkeypatch.setattr(training, "PoseModel", Pose)
    monkeypatch.setattr(training, "DogDetector", Detector)
    return Pose, Detector


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("flip", [False, True])
def test_normalization_centers_scales_and_preserves_aspect_ratio(flip):
    matrix, box = normalization_transform((10, 20, 210, 120), flip)
    corners = np.array([[10, 20, 1], [210, 120, 1]], dtype=float) @ matrix.T
    assert np.mean(corners, axis=0) == pytest.approx([320, 320])
    assert abs(corners[1, 0] - corners[0, 0]) == pytest.approx(DOG_SIZE)
    assert corners[1, 1] - corners[0, 1] == pytest.approx(DOG_SIZE / 2)
    assert (corners[1, 0] < corners[0, 0]) == flip
    assert box == pytest.approx([80, 200, 560, 440])


@pytest.mark.parametrize("left", [False, True])
def test_only_near_limbs_are_exported_after_normalizing(training_video, training_models, tmp_path, left):
    Pose, Detector = training_models
    Pose.missing_near = True
    output = tmp_path / "tracking"
    result = normalize_and_track(training_video(left=left), output, Pose(), Detector())
    assert result["flipped"] == left
    assert result["normalized_direction"] == "right"
    for which, limb in (("front", "right-fore"), ("hind", "right-hind")):
        rows = read_csv(output / f"{which}.csv")
        assert len(rows) == 66 * 3
        assert {r["which_limb"] for r in rows} == {which}
        assert {r["limb"] for r in rows} == {limb}
        assert {r["source_limb"] for r in rows} == {limb.replace("right", "left") if left else limb}
        assert all(r["x_px"] != "91.0" for r in rows)
    missing = [r for r in read_csv(output / "front.csv") if r["frame"] == "4" and r["joint"] == "front_paw"][0]
    assert missing["status"] == "low_confidence" and missing["x_px"] == ""
    cap = cv2.VideoCapture(str(output / "normalized.mp4"))
    ok, frame = cap.read()
    assert ok and frame.shape == (640, 640, 3)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 66
    cap.release()
    assert not (output / ".scan.jsonl").exists()


def test_uncertain_body_direction_is_not_inferred_from_confidence():
    points = np.full((17, 2), 100.0)
    scores = np.ones(17)
    assert heading_sign(points, scores, (0, 0, 300, 200), 0.3) == 0
    points[3], points[4] = (220, 100), (100, 100)
    assert heading_sign(points, scores, (0, 0, 300, 200), 0.3) == 1
    points[3], points[4] = (100, 100), (220, 100)
    assert heading_sign(points, scores, (0, 0, 300, 200), 0.3) == -1


def test_turning_video_is_rejected(training_video, training_models, tmp_path):
    Pose, Detector = training_models

    class TurningPose(Pose):
        def predict(self, frame, box):
            points, scores = super().predict(frame, box)
            if self.calls > 33:
                points[:, 0] = 319 - points[:, 0]
            return points, scores

    with pytest.raises(VideoQualityError, match="zmienia kierunek"):
        normalize_and_track(training_video(), tmp_path / "turning", TurningPose(), Detector())


def test_visible_far_side_does_not_replace_occluded_near_side(training_video, training_models, tmp_path):
    Pose, Detector = training_models

    class OccludedPose(Pose):
        def predict(self, frame, box):
            points, scores = super().predict(frame, box)
            if frame.shape[1] == 640:
                scores[[8, 9, 10]] = 0.01
            return points, scores

    output = tmp_path / "occluded"
    with pytest.raises(VideoQualityError, match="Za mało pewnych klatek front"):
        normalize_and_track(training_video(), output, OccludedPose(), Detector())
    rows = read_csv(output / "front.csv")
    assert all(row["status"] == "low_confidence" and row["x_px"] == "" for row in rows)


def feature_csv(path, fps=30, missing=()):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        for index in range(fps * 3 + 1):
            t = index / fps
            points = np.full((17, 2), 100.0)
            points[[8, 9, 10]] = [(400, 200), (410 + 8 * np.sin(t * 8), 300),
                                       (400 + 30 * np.sin(t * 8), 440)]
            rows = frame_rows(index, fps, "right-fore", None if index in missing else points,
                              np.ones(17), 640, 640, 0.3)
            for row in rows:
                row.update(source_limb="right-fore", normalized_dog_size_px=DOG_SIZE)
            writer.writerows(rows)
    return path


def test_feature_resampling_uses_time_and_is_comparable_across_fps(tmp_path):
    low = feature_windows(feature_csv(tmp_path / "30.csv", 30), "front")[0]["features"]
    high = feature_windows(feature_csv(tmp_path / "60.csv", 60), "front")[0]["features"]
    assert low["paw_x_velocity_std"] == pytest.approx(high["paw_x_velocity_std"], rel=0.03)
    assert low["joint_angle_mean"] == pytest.approx(high["joint_angle_mean"], rel=0.01)


def test_long_missing_segment_is_not_interpolated(tmp_path):
    path = feature_csv(tmp_path / "gap.csv", missing=range(25, 66))
    with pytest.raises(VideoQualityError, match="Brak ciągłego"):
        feature_windows(path, "front")
    with pytest.raises(VideoQualityError, match="inną kończynę"):
        feature_windows(path, "hind")


def make_dataset(root, training_video):
    for label in ("zdrowy", "chory"):
        for dog in ("pies_1", "pies_2"):
            # Unique dog IDs across classes; repeated IDs would correctly stay together.
            training_video(root / label / f"{label}_{dog}" / "walk.MP4")
    return root


def test_training_saves_two_loadable_models_and_versions_with_grouped_validation(
        training_video, training_models, tmp_path):
    dataset = make_dataset(tmp_path / "dane treningowe", training_video)
    config = TrainingConfig(dataset, output_root=tmp_path / "model", trees=12)
    first = run_training(config)
    version = Path(first["output_dir"])
    assert version.name == "model_V001"
    metadata = json.loads((version / "metadata.json").read_text())
    assert metadata["accepted_videos"] == 4
    assert set(metadata["validation_train_groups"]).isdisjoint(metadata["validation_test_groups"])
    for which in ("front", "hind"):
        assert metadata["validation"][which]["status"] == "held_out_groups"
        bundle = joblib.load(version / f"{which}_model.joblib")
        assert bundle["which_limb"] == which
        assert set(bundle["estimator"].classes_) == {"zdrowy", "chory"}
        csv_path = next((version / "data").rglob(f"{which}.csv"))
        prediction = predict_csv(version, csv_path, which)
        assert prediction["prediction"] in {"zdrowy", "chory"}
        assert sum(prediction["scores"].values()) == pytest.approx(1)
    rows = read_csv(version / "training_features.csv")
    assert {r["which_limb"] for r in rows} == {"front", "hind"}
    for group in {r["group_id"] for r in rows}:
        assert len({r["validation_split"] for r in rows if r["group_id"] == group}) == 1
    original = (version / "front_model.joblib").read_bytes()
    second = run_training(replace(config, trees=8))
    assert Path(second["output_dir"]).name == "model_V002"
    assert (version / "front_model.joblib").read_bytes() == original
    assert not list(config.output_root.glob(".training-*"))
    assert not list(config.output_root.glob("*.lock"))


def test_failed_training_does_not_publish_partial_model(training_video, training_models, tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset", training_video)
    output = tmp_path / "model"
    previous = output / "model_V007"
    previous.mkdir(parents=True)
    (previous / "keep.txt").write_text("preserve")

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(training.joblib, "dump", fail)
    with pytest.raises(OSError, match="simulated disk"):
        run_training(TrainingConfig(dataset, output_root=output, trees=2))
    assert sorted(p.name for p in output.iterdir()) == ["model_V007"]
    assert (previous / "keep.txt").read_text() == "preserve"


def test_empty_or_bad_class_never_creates_model(training_video, training_models, tmp_path):
    dataset = tmp_path / "dataset"
    training_video(dataset / "zdrowy" / "good.mp4")
    training_video(dataset / "chory" / "too_short.mp4", count=4)
    output = tmp_path / "model"
    with pytest.raises(TrackingError, match="muszą pozostać obie klasy"):
        run_training(TrainingConfig(dataset, output_root=output, trees=2))
    assert list(output.iterdir()) == []


def test_discovery_requires_both_classes_and_groups_same_dog(tmp_path):
    with pytest.raises(TrackingError, match="Brak podfolderu zdrowy"):
        discover_videos(tmp_path)
    for label in ("zdrowy", "chory"):
        path = tmp_path / label / "reksio" / "walk.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
    records = discover_videos(tmp_path)
    assert {r["group_id"] for r in records} == {"dog:reksio"}
    test_groups, note = training._holdout_groups(records, 42)
    assert test_groups == set() and "bez oceny" in note


def test_version_reservation_uses_maximum_and_concurrent_lock(tmp_path):
    (tmp_path / "model_V009").mkdir()
    (tmp_path / ".model_V010.lock").touch()
    with training._next_version(tmp_path) as path:
        assert path.name == "model_V011"
        with training._next_version(tmp_path) as concurrent:
            assert concurrent.name == "model_V012"
    assert not (tmp_path / ".model_V011.lock").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_launcher_training_command_preserves_paths_with_spaces(training_video, tmp_path, monkeypatch, device):
    dataset = make_dataset(tmp_path / "folder ze spacjami", training_video)
    monkeypatch.setattr(launcher, "check", lambda _: True)
    monkeypatch.setattr(launcher, "check_models", lambda: True)
    calls = []
    monkeypatch.setattr(launcher, "run_command", lambda args, **kw: calls.append((args, kw)) or True)
    assert launcher.train_models(device, dataset)
    arguments, kwargs = calls[-1]
    assert arguments[1] == launcher.ROOT / "train_gait.py"
    assert arguments[2] == dataset
    assert arguments[4] == ("cuda:0" if device == "cuda" else "cpu")
    assert kwargs["tracking"]


def test_launcher_menu_and_noninteractive_training(monkeypatch):
    monkeypatch.setattr(launcher, "require_project", lambda: True)
    calls = []
    monkeypatch.setattr(launcher, "perform", lambda *args, **kw: calls.append((args, kw)) or True)
    answers = iter(["9", "10", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert launcher.main([]) == 0
    assert [call[0] for call in calls] == [("train", "cpu"), ("train", "cuda")]
    assert launcher.main(["--train", "cpu", "--dataset", "dane ze spacjami"]) == 0
    assert calls[-1][1]["dataset"] == Path("dane ze spacjami")
    with pytest.raises(SystemExit):
        launcher.main(["--dataset", "dane"])


@pytest.mark.parametrize("limb", LIMBS)
def test_standard_tracker_csv_also_records_which_limb(limb):
    rows = frame_rows(0, 30, limb, None, None, 320, 240, 0.3)
    assert {r["which_limb"] for r in rows} == {"front" if limb.endswith("fore") else "hind"}
