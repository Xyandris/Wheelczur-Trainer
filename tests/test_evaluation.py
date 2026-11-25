"""MP4 inference uses pre-existing bundles and preserves their bytes."""

from pathlib import Path
import csv
import json
import sys

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

import launcher
from canine_limb import evaluation
from canine_limb.evaluation import EvaluationConfig, run_evaluation
from canine_limb.gait_features import FEATURE_VERSION, feature_windows
from canine_limb.model_versions import MODEL_FILES, available_versions, resolve_version
from canine_limb.runtime import TrackingError
from canine_limb.training_video import VideoQualityError
from test_training import feature_csv, training_models, training_video


@pytest.fixture
def saved_version(tmp_path):
    """Bundle schema is identical to the previously shipped training code."""
    path = tmp_path / "model" / "model_V001"
    path.mkdir(parents=True)
    feature_config = {"window_seconds": 2.0, "stride_seconds": 1.0, "sample_rate": 30,
                      "min_valid_frames": 20, "min_coverage": 0.5}
    names = list(feature_windows(feature_csv(tmp_path / "features.csv"), "front")[0]["features"])
    for which in ("front", "hind"):
        y = (["zdrowy"] * 3 + ["chory"] if which == "front" else ["zdrowy"] + ["chory"] * 3)
        classifier = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=42, n_jobs=1)
        classifier.fit(np.zeros((4, len(names))), y)
        joblib.dump({"estimator": classifier, "which_limb": which, "feature_names": names,
                     "feature_version": FEATURE_VERSION, "feature_config": feature_config,
                     "labels": ["zdrowy", "chory"], "sklearn_version": evaluation.sklearn.__version__},
                    path / f"{which}_model.joblib")
    (path / "metadata.json").write_text(json.dumps({
        "normalization": {"canvas_size": 640, "dog_bbox_longest_side": 480.0, "direction": "right"},
        "config": {"confidence": 0.3, "detector_confidence": 0.3},
        "validation": {"front": {"status": "held_out_groups", "balanced_accuracy": 0.75, "videos": 4},
                       "hind": {"status": "not_available"}},
    }), encoding="utf-8")
    return path


@pytest.fixture
def evaluation_models(training_models, monkeypatch):
    Pose, Detector = training_models
    monkeypatch.setattr(evaluation, "PoseModel", Pose)
    monkeypatch.setattr(evaluation, "DogDetector", Detector)


def test_mp4_scores_use_correct_class_order_and_do_not_retrain_or_change_models(
        saved_version, training_video, evaluation_models, monkeypatch):
    originals = {p: p.read_bytes() for p in saved_version.iterdir()}

    def no_training(*args, **kwargs):
        raise AssertionError("Inference must never fit a classifier")

    monkeypatch.setattr(RandomForestClassifier, "fit", no_training)
    video = training_video()
    result = run_evaluation(EvaluationConfig(video, saved_version))
    assert result["models"]["front"]["prediction"] == "zdrowy"
    assert result["models"]["hind"]["prediction"] == "chory"
    assert result["models"]["front"]["scores"] == pytest.approx({"zdrowy": 0.75, "chory": 0.25})
    assert result["models"]["hind"]["scores"] == pytest.approx({"zdrowy": 0.25, "chory": 0.75})
    assert "różne" in result["agreement"]
    output = Path(result["output_dir"])
    for path in result["outputs"].values():
        assert (output / path).is_file()
    with (output / "wyniki.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["which_limb"] == "front" and float(rows[0]["zdrowy_percent"]) == 75
    assert rows[1]["which_limb"] == "hind" and float(rows[1]["chory_percent"]) == 75
    html = (output / "raport.html").read_text()
    assert "75.0%" in html and "Przednia kończyna" in html and "Tylna kończyna" in html
    assert "balanced accuracy 75.0%" in html
    assert "nie ma wyniku walidacji" in html
    json_result = json.loads((output / "wyniki.json").read_text())
    assert json_result["models"] == result["models"]
    second = run_evaluation(EvaluationConfig(video, saved_version))
    assert second["output_dir"] != result["output_dir"]
    for path, data in originals.items():
        assert path.read_bytes() == data
    assert not list(output.parent.glob(".ocena-*"))


def test_bad_video_produces_no_report_or_scores(saved_version, training_video, evaluation_models, tmp_path):
    output = tmp_path / "results"
    video = training_video(count=4)
    with pytest.raises(VideoQualityError, match="Za mało pewnych"):
        run_evaluation(EvaluationConfig(video, saved_version, output))
    assert list(output.iterdir()) == []


def test_incompatible_model_fails_before_tracking(saved_version, training_video, monkeypatch):
    bundle_path = saved_version / "front_model.joblib"
    bundle = joblib.load(bundle_path)
    bundle["feature_version"] += 1
    joblib.dump(bundle, bundle_path)
    monkeypatch.setattr(evaluation, "PoseModel", lambda *a: pytest.fail("Tracker must not load for incompatible bundles"))
    with pytest.raises(TrackingError, match="niezgodny format modelu front"):
        run_evaluation(EvaluationConfig(training_video(), saved_version))


def test_different_sklearn_version_has_actionable_error(saved_version, training_video, monkeypatch):
    def load(*args):
        raise InconsistentVersionWarning(estimator_name="RandomForestClassifier", current_sklearn_version="1.9.0",
                                         original_sklearn_version="1.8.0")
    monkeypatch.setattr(evaluation.joblib, "load", load)
    with pytest.raises(TrackingError, match=r"scikit-learn==1\.8\.0"):
        run_evaluation(EvaluationConfig(training_video(), saved_version))


def test_result_folder_cannot_be_inside_model_version(saved_version, training_video):
    with pytest.raises(TrackingError, match="poza zapisaną wersją"):
        run_evaluation(EvaluationConfig(training_video(), saved_version, saved_version / "results"))
    assert not (saved_version / "results").exists()


def test_numeric_latest_complete_version_and_explicit_path(saved_version):
    root = saved_version.parent
    for name in ("model_V009", "model_V010", "model_V999"):
        directory = root / name
        directory.mkdir()
        if name != "model_V999":
            for filename in MODEL_FILES:
                (directory / filename).touch()
    assert [p.name for p in available_versions(root)] == ["model_V010", "model_V009", "model_V001"]
    assert resolve_version(None, root).name == "model_V010"
    assert resolve_version("model_V001", root) == saved_version
    with pytest.raises(ValueError, match="Niepełna"):
        resolve_version(root / "model_V999", root)
    with pytest.raises(ValueError, match="Brak wytrenowanej"):
        resolve_version(None, root / "absent")


def test_cli_outputs_both_percentages(saved_version, training_video, evaluation_models, capsys):
    assert evaluation.main([str(training_video()), "--model-version", str(saved_version)]) == 0
    output = capsys.readouterr().out
    assert "Przednia kończyna (front): ZDROWY" in output
    assert "Tylna kończyna (hind): CHORY" in output
    assert "zdrowy: 75.0% | chory: 25.0%" in output
    assert "raport:" in output


def test_html_escapes_untrusted_video_name(saved_version, training_video, evaluation_models):
    result = run_evaluation(EvaluationConfig(training_video(), saved_version))
    result["input_video"] = '/tmp/<img src=x onerror="alert(1)">.mp4'
    html = evaluation._report_html(result)
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_launcher_prompts_latest_or_named_version(saved_version, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", saved_version.parent.parent)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert launcher.prompt_model_version() == saved_version
    assert launcher.prompt_model_version("model_V001") == saved_version
    assert launcher.prompt_model_version(f'"{saved_version}"') == saved_version


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_launcher_evaluation_command(saved_version, training_video, monkeypatch, device):
    video = training_video()
    monkeypatch.setattr(launcher, "check", lambda _: True)
    monkeypatch.setattr(launcher, "check_models", lambda: True)
    calls = []
    monkeypatch.setattr(launcher, "run_command", lambda args, **kw: calls.append((args, kw)) or True)
    assert launcher.evaluate_video(device, f'"{video}"', saved_version)
    args, kwargs = calls[-1]
    assert args[1] == launcher.ROOT / "evaluate_gait.py"
    assert args[2] == video
    assert args[4] == saved_version
    assert args[6] == ("cuda:0" if device == "cuda" else "cpu")
    assert kwargs["tracking"]


def test_launcher_menu_options_and_cli(monkeypatch):
    monkeypatch.setattr(launcher, "require_project", lambda: True)
    calls = []
    monkeypatch.setattr(launcher, "perform", lambda *args, **kw: calls.append((args, kw)) or True)
    answers = iter(["11", "12", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert launcher.main([]) == 0
    assert [call[0] for call in calls] == [("evaluate", "cpu"), ("evaluate", "cuda")]
    assert launcher.main(["--evaluate", "cpu", "--video", "walk.mp4", "--model-version", "model_V001"]) == 0
    assert calls[-1][1]["video"] == Path("walk.mp4")
    with pytest.raises(SystemExit):
        launcher.main(["--video", "walk.mp4"])


def test_launcher_quiet_mode_shows_results_and_report(capsys):
    assert launcher.run_command([sys.executable, "-c", 'print("Ocena: zdrowy: 75.0% | chory: 25.0%"); print("raport: wynik.html")'], tracking=True)
    output = capsys.readouterr().out
    assert "zdrowy: 75.0%" in output and "raport: wynik.html" in output
