"""Train separate front/hind healthy-vs-affected gait classifiers from MP4s."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from . import __version__
from .gait_features import FEATURE_VERSION, feature_windows
from .models import MODEL_HASHES, DogDetector, PoseModel
from .runtime import TrackingError, parse_device
from .training_video import CANVAS_SIZE, DOG_SIZE, VideoQualityError, normalize_and_track


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "model"
LABELS = ("zdrowy", "chory")


@dataclass
class TrainingConfig:
    dataset: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    device: str = "cpu"
    models_dir: Path | None = None
    confidence: float = 0.3
    detector_confidence: float = 0.3
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    min_valid_frames: int = 20
    min_coverage: float = 0.5
    trees: int = 300
    seed: int = 42


def discover_videos(dataset):
    dataset = Path(dataset).expanduser().resolve()
    records = []
    for label in LABELS:
        folder = dataset / label
        if not folder.is_dir():
            raise TrackingError(f"Brak podfolderu {label}: {folder}")
        paths = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"),
                       key=lambda p: p.as_posix().casefold())
        if not paths:
            raise TrackingError(f"Folder {folder} nie zawiera nagrań .mp4.")
        for path in paths:
            relative = path.relative_to(dataset)
            within_label = path.relative_to(folder)
            # Same named dog under both labels stays in the same validation group.
            group = ("dog:" + within_label.parts[0] if len(within_label.parts) > 1
                     else "video:" + relative.as_posix())
            identifier = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
            records.append({"path": path, "source_id": relative.as_posix(), "group_id": group,
                            "label": label, "clip_id": identifier})
    return records


@contextmanager
def _next_version(root):
    """Reserve a monotonic version across concurrent cooperating trainers."""
    number = 1
    for path in root.iterdir():
        match = re.fullmatch(r"\.?model_V(\d+)(?:\.lock)?", path.name)
        if match:
            number = max(number, int(match[1]) + 1)
    while True:
        destination = root / f"model_V{number:03d}"
        lock = root / f".{destination.name}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            number += 1
            continue
        os.close(descriptor)
        if destination.exists():
            lock.unlink()
            number += 1
            continue
        try:
            yield destination
        finally:
            lock.unlink(missing_ok=True)
        return


def _feature_config(config):
    return {"window_seconds": config.window_seconds, "stride_seconds": config.stride_seconds,
            "sample_rate": 30, "min_valid_frames": config.min_valid_frames,
            "min_coverage": config.min_coverage}


def _validate(config):
    parse_device(config.device)
    if (not math.isfinite(config.confidence) or config.confidence < 0
            or not math.isfinite(config.detector_confidence) or not 0 <= config.detector_confidence <= 1
            or not math.isfinite(config.min_coverage) or not 0 < config.min_coverage <= 1
            or not math.isfinite(config.window_seconds) or config.window_seconds < 0.5
            or not math.isfinite(config.stride_seconds) or config.stride_seconds <= 0
            or config.min_valid_frames < 3 or config.trees < 1 or config.seed < 0):
        raise TrackingError("Nieprawidłowe parametry treningu.")


def _holdout_groups(records, seed):
    groups = np.array([r["group_id"] for r in records])
    y = np.array([r["label"] for r in records])
    unique = set(groups)
    if len(unique) < 4 or any(len(set(groups[y == label])) < 2 for label in LABELS):
        return set(), "Za mało niezależnych grup do walidacji; model wytrenowano bez oceny skuteczności."
    split = GroupShuffleSplit(n_splits=64, test_size=max(2, math.ceil(len(unique) * 0.25)), random_state=seed)
    for train, test in split.split(np.zeros(len(y)), y, groups):
        if set(y[train]) == set(LABELS) and set(y[test]) == set(LABELS):
            return set(groups[test]), None
    return set(), "Nie udało się wydzielić obu klas bez rozdzielania grup; brak wyniku walidacji."


def _new_classifier(config):
    return RandomForestClassifier(n_estimators=config.trees, max_depth=12, min_samples_leaf=1,
                                  class_weight="balanced_subsample", random_state=config.seed, n_jobs=-1)


def _fit(classifier, samples, names):
    counts = Counter(s["group_id"] for s in samples)
    weights = np.array([1 / counts[s["group_id"]] for s in samples])
    classifier.fit(np.array([[s["features"][name] for name in names] for s in samples]),
                   [s["label"] for s in samples], sample_weight=weights)
    return classifier


def _evaluate(classifier, samples, names, which):
    x = np.array([[s["features"][name] for name in names] for s in samples])
    probabilities = classifier.predict_proba(x)
    predictions = []
    for source in sorted({s["source_id"] for s in samples}):
        indices = [i for i, s in enumerate(samples) if s["source_id"] == source]
        mean = probabilities[indices].mean(axis=0)
        sample = samples[indices[0]]
        predictions.append({"source_id": source, "group_id": sample["group_id"], "which_limb": which,
                            "label": sample["label"], "prediction": str(classifier.classes_[mean.argmax()]),
                            **{f"score_{label}": float(mean[list(classifier.classes_).index(label)]) for label in LABELS}})
    truth, predicted = [p["label"] for p in predictions], [p["prediction"] for p in predictions]
    metrics = {"status": "held_out_groups", "unit": "source_video", "videos": len(predictions),
               "accuracy": float(accuracy_score(truth, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
               "confusion_labels": list(LABELS), "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist()}
    return metrics, predictions


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_training(config, progress=None):
    _validate(config)
    records = discover_videos(config.dataset)
    # Fail on missing/corrupt model or runtime before producing any version.
    pose = PoseModel(config.device, config.models_dir)
    detector = DogDetector(config.device, config.models_dir, config.detector_confidence)
    root = Path(config.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    feature_config = _feature_config(config)
    accepted, skipped, samples = [], [], []
    with tempfile.TemporaryDirectory(prefix=".training-", dir=root) as temporary:
        stage = Path(temporary) / "version"
        stage.mkdir()
        for index, record in enumerate(records, 1):
            source = record["source_id"]
            if progress:
                progress(f"Training: nagranie {index}/{len(records)}: {source}")
            relative_output = Path("data") / record["label"] / record["clip_id"]
            clip_dir = stage / relative_output

            def frame_progress(phase, done, total):
                if progress:
                    progress(f"Processed {source}: {phase} {done}/{total}")

            try:
                tracking = normalize_and_track(record["path"], clip_dir, pose, detector,
                    confidence=config.confidence, min_valid_frames=config.min_valid_frames,
                    min_coverage=config.min_coverage, progress=frame_progress)
                clip_samples = []
                for which in ("front", "hind"):
                    windows = feature_windows(clip_dir / f"{which}.csv", which, **feature_config)
                    clip_samples.extend({**window, "source_id": source, "group_id": record["group_id"],
                                         "label": record["label"], "which_limb": which} for window in windows)
            except VideoQualityError as exc:
                skipped.append({"source_id": source, "reason": str(exc)})
                shutil.rmtree(clip_dir, ignore_errors=True)
                if progress:
                    progress(f"Skipped: {source}: {exc}")
                continue
            samples.extend(clip_samples)
            accepted.append({k: str(v) if isinstance(v, Path) else v for k, v in record.items() if k != "path"})
            accepted[-1].update({"output_dir": relative_output.as_posix(), "frames": tracking["frames"],
                                 "valid_frames": tracking["valid_frames"]})
        if {record["label"] for record in accepted} != set(LABELS):
            reasons = "; ".join(f"{r['source_id']}: {r['reason']}" for r in skipped[:4])
            raise TrackingError("Po kontroli nagrań muszą pozostać obie klasy: zdrowy i chory. " + reasons)
        test_groups, validation_note = _holdout_groups(accepted, config.seed)
        if validation_note and progress:
            progress("Training: " + validation_note)
        names = list(samples[0]["features"])
        metrics, validation_predictions = {}, []
        for which in ("front", "hind"):
            selected = [sample for sample in samples if sample["which_limb"] == which]
            if progress:
                progress(f"Training: uczenie {which}, {len(selected)} okien ruchu")
            train = [s for s in selected if s["group_id"] not in test_groups]
            test = [s for s in selected if s["group_id"] in test_groups]
            if test:
                evaluation_model = _fit(_new_classifier(config), train, names)
                metrics[which], predictions = _evaluate(evaluation_model, test, names, which)
                validation_predictions.extend(predictions)
            else:
                metrics[which] = {"status": "not_available", "reason": validation_note}
            # The saved model learns from every accepted recording after evaluation.
            model = _fit(_new_classifier(config), selected, names)
            joblib.dump({"estimator": model, "which_limb": which, "feature_names": names,
                         "feature_version": FEATURE_VERSION, "feature_config": feature_config,
                         "labels": list(LABELS), "sklearn_version": sklearn.__version__},
                        stage / f"{which}_model.joblib", compress=3)
        with (stage / "training_features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            fields = ["source_id", "group_id", "label", "which_limb", "start_s", "end_s", "validation_split"] + names
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                writer.writerow({**{k: v for k, v in sample.items() if k != "features"}, **sample["features"],
                                 "validation_split": ("test" if sample["group_id"] in test_groups else
                                                      "train" if test_groups else "not_evaluated")})
        with (stage / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_id", "group_id", "which_limb", "label", "prediction",
                                                       "score_zdrowy", "score_chory"])
            writer.writeheader()
            writer.writerows(validation_predictions)
        _write_json(stage / "dataset_manifest.json", {"accepted": accepted, "skipped": skipped})
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
        metadata = {
            "program_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": "healthy_vs_affected_gait", "algorithm": "RandomForestClassifier",
            "which_limb": ["front", "hind"], "labels": list(LABELS), "config": settings,
            "accepted_videos": len(accepted), "skipped_videos": len(skipped),
            "class_video_counts": dict(Counter(r["label"] for r in accepted)),
            "feature_version": FEATURE_VERSION, "feature_names": names, "feature_config": feature_config,
            "validation": metrics, "validation_test_groups": sorted(test_groups),
            "validation_train_groups": sorted({r["group_id"] for r in accepted} - test_groups),
            "saved_model_training": "all accepted videos; evaluation used separate temporary models",
            "normalization": {"direction": "right", "canvas_size": CANVAS_SIZE, "dog_bbox_longest_side": DOG_SIZE},
            "tracker_model_sha256": MODEL_HASHES, "pose_runtime": pose.info, "detector_runtime": detector.info,
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
            "processing_seconds": round(time.monotonic() - started, 3),
        }
        with _next_version(root) as destination:
            metadata["model_version"] = destination.name
            _write_json(stage / "metadata.json", metadata)
            # Same filesystem: only the completed, coherent version becomes visible.
            stage.rename(destination)
        return {"output_dir": str(destination), **metadata}


def predict_csv(model_version, csv_path, which_limb):
    if which_limb not in {"front", "hind"}:
        raise TrackingError("Wybierz which_limb: front albo hind.")
    bundle = joblib.load(Path(model_version) / f"{which_limb}_model.joblib")
    if bundle["which_limb"] != which_limb or bundle["feature_version"] != FEATURE_VERSION:
        raise TrackingError("Niezgodny format lub rodzaj kończyny w zapisanym modelu.")
    windows = feature_windows(csv_path, which_limb, **bundle["feature_config"])
    x = np.array([[window["features"][name] for name in bundle["feature_names"]] for window in windows])
    model = bundle["estimator"]
    mean = model.predict_proba(x).mean(axis=0)
    return {"which_limb": which_limb, "prediction": str(model.classes_[mean.argmax()]),
            "scores": {str(label): float(score) for label, score in zip(model.classes_, mean)}, "windows": len(windows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Trening zdrowy/chory z folderów MP4 zdrowy i chory.")
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu lub cuda:N; dotyczy trackera, klasyfikacja używa CPU")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--predict-csv", type=Path, help="Oceń znormalizowany CSV zamiast treningu")
    parser.add_argument("--model-version", type=Path)
    parser.add_argument("--which-limb", choices=("front", "hind"))
    args = parser.parse_args(argv)
    if args.predict_csv:
        if args.dataset or not args.model_version or not args.which_limb:
            parser.error("--predict-csv wymaga --model-version i --which-limb, bez DATASET")
    elif not args.dataset or args.model_version or args.which_limb:
        parser.error("Podaj DATASET albo użyj --predict-csv z --model-version i --which-limb")
    try:
        if args.predict_csv:
            print(json.dumps(predict_csv(args.model_version, args.predict_csv, args.which_limb), ensure_ascii=False, indent=2))
            return 0
        last_update = 0.0

        def progress(message):
            nonlocal last_update
            now = time.monotonic()
            if not message.startswith("Processed ") or now - last_update >= 1:
                print(message, flush=True)
                last_update = now

        result = run_training(TrainingConfig(
            dataset=args.dataset, output_root=args.output_root, device=args.device, models_dir=args.models_dir,
            confidence=args.confidence, detector_confidence=args.detector_confidence,
            window_seconds=args.window_seconds, stride_seconds=args.stride_seconds), progress)
        print(f"Finished: trening zakończony, {result['accepted_videos']} nagrań, pominięto {result['skipped_videos']}")
        print(f"model: {result['output_dir']}")
        for which, metrics in result["validation"].items():
            if metrics["status"] == "held_out_groups":
                print(f"Training: walidacja {which}: balanced accuracy = {metrics['balanced_accuracy']:.3f}")
        return 0
    except KeyboardInterrupt:
        print("\nPrzerwano trening. Nie zapisano niepełnej wersji modelu.", file=sys.stderr)
        return 130
    except (TrackingError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
