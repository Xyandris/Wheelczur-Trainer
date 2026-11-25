"""Evaluate one MP4 with the saved, separate front/hind gait classifiers."""

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import time
import uuid
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from . import __version__
from .gait_features import FEATURE_VERSION, feature_windows
from .model_versions import resolve_version
from .models import DogDetector, PoseModel
from .runtime import TrackingError, parse_device
from .training import DEFAULT_OUTPUT_ROOT, LABELS
from .training_video import CANVAS_SIZE, DOG_SIZE, normalize_and_track


SCORE_NOTE = ("Procenty są wynikami klasyfikatorów, a nie potwierdzonym prawdopodobieństwem choroby. "
              "Skuteczność modelu sprawdź w wynikach walidacji na osobnych nagraniach.")
LIMB_NAMES = {"front": "Przednia kończyna", "hind": "Tylna kończyna"}


@dataclass
class EvaluationConfig:
    video: Path
    model_version: Path | None = None
    output_dir: Path | None = None  # parent for unique evaluation runs
    device: str = "cpu"
    models_dir: Path | None = None


def _load_models(version):
    try:
        metadata = json.loads((version / "metadata.json").read_text(encoding="utf-8"))
        normalization = metadata["normalization"]
        if (normalization["canvas_size"] != CANVAS_SIZE
                or normalization["dog_bbox_longest_side"] != DOG_SIZE
                or normalization["direction"] != "right"):
            raise ValueError("niezgodna normalizacja obrazu")
        config = metadata["config"]
        confidence, detector_confidence = float(config["confidence"]), float(config["detector_confidence"])
        if (not math.isfinite(confidence) or confidence < 0 or not math.isfinite(detector_confidence)
                or not 0 <= detector_confidence <= 1):
            raise ValueError("nieprawidłowe progi trackera")
        bundles = {}
        for which in LIMB_NAMES:
            # A different sklearn version may deserialize an incompatible tree.
            # Stop with actionable information instead of silently trusting it.
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                bundle = joblib.load(version / f"{which}_model.joblib")
            model = bundle["estimator"]
            names = bundle["feature_names"]
            if (bundle["which_limb"] != which or bundle["feature_version"] != FEATURE_VERSION
                    or set(model.classes_) != set(LABELS) or not names or len(set(names)) != len(names)
                    or model.n_features_in_ != len(names)):
                raise ValueError(f"niezgodny format modelu {which}")
            feature_config = bundle["feature_config"]
            if (not 0 < feature_config["min_coverage"] <= 1 or feature_config["min_valid_frames"] < 3
                    or feature_config["window_seconds"] < 0.5 or feature_config["stride_seconds"] <= 0
                    or feature_config["sample_rate"] < 5
                    or not np.isfinite(list(feature_config.values())).all()):
                raise ValueError(f"nieprawidłowe ustawienia cech {which}")
            bundles[which] = bundle
        return metadata, bundles, confidence, detector_confidence
    except InconsistentVersionWarning as exc:
        raise TrackingError(
            f"Model zapisano w scikit-learn {exc.original_sklearn_version}, a środowisko ma "
            f"{exc.current_sklearn_version}. W środowisku użytym do oceny zainstaluj "
            f"scikit-learn=={exc.original_sklearn_version}; nie trzeba ponawiać treningu."
        ) from exc
    except (KeyError, TypeError, AttributeError, ValueError, EOFError, IndexError) as exc:
        raise TrackingError(f"Nie można użyć wersji {version.name}: {exc}") from exc


def _predict(bundle, path):
    which = bundle["which_limb"]
    windows = feature_windows(path, which, **bundle["feature_config"])
    try:
        x = np.array([[w["features"][name] for name in bundle["feature_names"]] for w in windows])
    except KeyError as exc:
        raise TrackingError(f"Niezgodne cechy zapisanej wersji: {exc}") from exc
    model = bundle["estimator"]
    probabilities = np.asarray(model.predict_proba(x))
    if (probabilities.shape != (len(windows), 2) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise TrackingError(f"Model {which} zwrócił nieprawidłowe wyniki klas.")
    mean = probabilities.mean(axis=0)
    scores = {str(label): float(score) for label, score in zip(model.classes_, mean)}
    tied = abs(scores["zdrowy"] - scores["chory"]) <= 1e-12
    result = {"which_limb": which, "prediction": "niejednoznaczny" if tied else max(scores, key=scores.get),
              "confidence": float(max(scores.values())), "scores": scores, "windows": len(windows)}
    rows = []
    for window, values in zip(windows, probabilities):
        scores_by_label = dict(zip(model.classes_, map(float, values)))
        rows.append({"which_limb": which, "start_s": window["start_s"], "end_s": window["end_s"],
                     "score_zdrowy": scores_by_label["zdrowy"], "score_chory": scores_by_label["chory"]})
    return result, rows


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_html(result):
    cards = []
    for which, model in result["models"].items():
        rows = []
        for label in LABELS:
            percent = model["scores"][label] * 100
            rows.append(f'<div class="score"><span>{label.capitalize()}</span><strong>{percent:.1f}%</strong></div>'
                        f'<div class="bar"><div class="{label}" style="width:{percent:.4f}%"></div></div>')
        validation = model["validation"]
        if validation.get("status") == "held_out_groups":
            validation_text = f'Walidacja: balanced accuracy {validation["balanced_accuracy"] * 100:.1f}% '
            validation_text += f'na {validation["videos"]} osobnych nagraniach.'
        else:
            validation_text = "Dla tej wersji nie ma wyniku walidacji na osobnych grupach."
        cards.append(f'<section><h2>{LIMB_NAMES[which]} <small>{which}</small></h2>'
                     f'<p class="prediction">Wskazanie modelu: {escape(model["prediction"].upper())}</p>'
                     + "".join(rows) + f'<p>Okna ruchu: {model["windows"]}. '
                     f'Kompletne klatki: {model["valid_frames"]}/{result["tracking"]["frames"]}.</p>'
                     f'<p class="muted">{escape(validation_text)}</p></section>')
    return ('<!doctype html><html lang="pl"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Wheelczur — ocena ruchu</title><style>'
            'body{font:16px/1.55 system-ui,sans-serif;background:#f1f5f9;color:#172536;margin:0;padding:28px}'
            'main{max-width:960px;margin:auto}h1{margin-bottom:6px}h2{font-size:21px}small,.muted{color:#536478}'
            '.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px}'
            'section,.note{background:white;border-radius:12px;padding:24px;margin:20px 0;box-shadow:0 2px 8px #1231}'
            '.score{display:flex;justify-content:space-between;margin-top:14px}.bar{height:14px;background:#e2e8f0;border-radius:10px;overflow:hidden}'
            '.bar div{height:100%}.zdrowy{background:#168069}.chory{background:#d27b21}.prediction{font-weight:700}'
            'video{display:block;width:100%;max-height:640px;background:black;border-radius:12px}a{color:#075cac}'
            'code{overflow-wrap:anywhere}footer{margin:22px 0;color:#536478}</style><main>'
            '<h1>Ocena ruchu psa</h1>'
            f'<p>Film: <strong>{escape(Path(result["input_video"]).name)}</strong><br>'
            f'Wersja modeli: <code>{escape(result["model_version"])}</code></p>'
            f'<p>{escape(result["agreement"])}</p><div class="models">' + "".join(cards) + '</div>'
            f'<p class="note">{escape(SCORE_NOTE)} Wyniki dotyczą klasy całego nagrania; '
            'etykiety treningowe nie określały, która kończyna jest chora.</p>'
            '<h2>Podgląd śledzonych kończyn</h2>'
            '<video controls preload="metadata" src="tracking/tracking.mp4"></video>'
            '<p>Jeśli przeglądarka nie odtwarza MP4, otwórz '
            '<a href="tracking/tracking.mp4">tracking.mp4</a> w odtwarzaczu wideo.</p>'
            '<p><a href="wyniki.csv">Wyniki CSV</a> · <a href="wyniki.json">Pełne wyniki JSON</a> · '
            '<a href="wyniki_okien.csv">Wyniki poszczególnych okien ruchu</a></p>'
            '<footer>Porównuj nowe nagrania z boku. Sprawdź, czy punkty na podglądzie dotyczą właściwych kończyn.</footer></main></html>')


def run_evaluation(config, progress=None):
    parse_device(config.device)
    video = Path(config.video).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise TrackingError(f"Podaj istniejące nagranie .mp4: {video}")
    version = resolve_version(config.model_version, DEFAULT_OUTPUT_ROOT)
    metadata, bundles, confidence, detector_confidence = _load_models(version)
    root = (Path(config.output_dir).expanduser().resolve() if config.output_dir
            else video.parent / f"{video.stem}_ocena")
    # Never write evaluation reports inside a trained version.
    # This also prevents an accidental --output-dir from touching model files.
    if root == version or root.is_relative_to(version):
        raise TrackingError("Wybierz folder wyników poza zapisaną wersją modelu.")
    if progress:
        progress(f"Ocena: używam {version}")
    pose = PoseModel(config.device, config.models_dir)
    detector = DogDetector(config.device, config.models_dir, detector_confidence)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".ocena-", dir=root) as temporary:
        stage = Path(temporary) / "wynik"
        stage.mkdir()

        def frame_progress(phase, done, total):
            if progress:
                progress(f"Processed {phase}: {done}/{total}")

        tracking = normalize_and_track(video, stage / "tracking", pose, detector,
            confidence=confidence,
            min_valid_frames=max(b["feature_config"]["min_valid_frames"] for b in bundles.values()),
            min_coverage=max(b["feature_config"]["min_coverage"] for b in bundles.values()),
            progress=frame_progress)
        models, window_rows = {}, []
        for which, bundle in bundles.items():
            model, rows = _predict(bundle, stage / "tracking" / f"{which}.csv")
            model.update(valid_frames=tracking["valid_frames"][which],
                         coverage=tracking["valid_frames"][which] / tracking["frames"],
                         validation=metadata.get("validation", {}).get(which, {"status": "not_available"}))
            models[which] = model
            window_rows.extend(rows)
        predictions = {m["prediction"] for m in models.values()}
        if len(predictions) == 1 and "niejednoznaczny" not in predictions:
            agreement = "Oba modele wskazują: " + next(iter(predictions)).upper() + "."
        else:
            agreement = "Modele dają różne lub niejednoznaczne wyniki; sprawdź wyniki każdej kończyny."
        result = {
            "program_version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_video": str(video), "model_version": version.name, "model_directory": str(version),
            "models": models, "agreement": agreement, "score_note": SCORE_NOTE,
            "score_aggregation": "mean class scores across valid temporal windows, independently per limb",
            "model_sha256": {which: _sha256(version / f"{which}_model.joblib") for which in LIMB_NAMES},
            "tracking": tracking, "device": config.device,
            "runtime_versions": {"sklearn": sklearn.__version__, "numpy": np.__version__},
            "processing_seconds": round(time.monotonic() - started, 3),
            "outputs": {"report": "raport.html", "json": "wyniki.json", "csv": "wyniki.csv",
                        "window_csv": "wyniki_okien.csv", "video": "tracking/tracking.mp4"},
        }
        _write_json(stage / "wyniki.json", result)
        with (stage / "wyniki.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model_version", "which_limb", "prediction", "confidence_percent",
                                                       "zdrowy_percent", "chory_percent", "windows", "valid_frames", "coverage"])
            writer.writeheader()
            for which, model in models.items():
                writer.writerow({"model_version": version.name, "which_limb": which, "prediction": model["prediction"],
                    "confidence_percent": model["confidence"] * 100, "zdrowy_percent": model["scores"]["zdrowy"] * 100,
                    "chory_percent": model["scores"]["chory"] * 100, "windows": model["windows"],
                    "valid_frames": model["valid_frames"], "coverage": model["coverage"]})
        with (stage / "wyniki_okien.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["which_limb", "start_s", "end_s", "score_zdrowy", "score_chory"])
            writer.writeheader()
            writer.writerows(window_rows)
        (stage / "raport.html").write_text(_report_html(result), encoding="utf-8")
        destination = root / ("ocena_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        if destination.exists():
            raise TrackingError(f"Wynik już istnieje: {destination}")
        stage.rename(destination)
    return {**result, "output_dir": str(destination)}


def print_results(result):
    print(f"Finished: ocena nagrania zakończona; model {result['model_version']}")
    for which, model in result["models"].items():
        print(f"Ocena: {LIMB_NAMES[which]} ({which}): {model['prediction'].upper()}")
        print(f"Ocena:   zdrowy: {model['scores']['zdrowy'] * 100:.1f}% | chory: {model['scores']['chory'] * 100:.1f}%")
        print(f"Ocena:   przeanalizowano {model['windows']} okien; kompletne klatki {model['valid_frames']}/{result['tracking']['frames']}")
    print("Ocena: " + result["agreement"])
    print("Ocena: " + SCORE_NOTE)
    for label in ("report", "json", "csv", "video"):
        prefix = "raport" if label == "report" else label
        print(f"{prefix}: {Path(result['output_dir']) / result['outputs'][label]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Oceń jeden MP4 zapisanymi modelami zdrowy/chory, bez ponownego treningu.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model-version", type=Path, help="Folder model_Vxxx; domyślnie najnowsza kompletna wersja w model/")
    parser.add_argument("--device", default="cpu", help="cpu lub cuda:N dla trackera")
    parser.add_argument("--models-dir", type=Path, help="Folder modeli ONNX trackera")
    parser.add_argument("--output-dir", type=Path, help="Folder nadrzędny raportów; domyślnie <film>_ocena/")
    args = parser.parse_args(argv)
    try:
        last_update = 0.0

        def progress(message):
            nonlocal last_update
            now = time.monotonic()
            if not message.startswith("Processed ") or now - last_update >= 1:
                print(message, flush=True)
                last_update = now

        result = run_evaluation(EvaluationConfig(args.video, args.model_version, args.output_dir, args.device, args.models_dir), progress)
        print_results(result)
        return 0
    except KeyboardInterrupt:
        print("\nPrzerwano ocenę. Nie zapisano niepełnego raportu.", file=sys.stderr)
        return 130
    except (TrackingError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
