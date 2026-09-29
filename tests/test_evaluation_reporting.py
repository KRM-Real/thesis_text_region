from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import evaluate_scores, write_evaluation_report
from src.evaluation.feature_analysis import FeatureAnalysisReport, FeatureInfluenceRecord, write_feature_analysis
from src.evaluation.report import (
    EvaluationReportError,
    _validate_reference_checksum_lineage,
    discover_evaluation_runs,
    load_evaluation_report,
)
from src.cli import _reference_input_checksums
from src.reproducibility import sha256_file, sha256_json


def _build_report(tmp_path: Path, *, with_errors: bool = False) -> tuple[Path, dict[str, str]]:
    run_path = tmp_path / "evaluation-run"
    reports_path = run_path / "reports"
    model_id = "model-test"
    threshold = 0.5
    if with_errors:
        labels = np.asarray([0, 1])
        scores = np.asarray([0.9, 0.1])
    else:
        labels = np.asarray([0, 1])
        scores = np.asarray([0.1, 0.9])
    metrics = evaluate_scores(labels, scores, threshold)
    predictions = [
        {
            "sample_id": f"sample-{index}",
            "true_label": int(label),
            "score": float(score),
            "predicted_label": "ai_generated" if score >= threshold else "non_ai_generated",
            "threshold": threshold,
            "region_count": 2,
            "no_detection": False,
            "input_sha256": f"{index + 1:064x}",
        }
        for index, (label, score) in enumerate(zip(labels, scores))
    ]
    write_evaluation_report(reports_path, metrics, predictions, "# Test evaluation report\n")
    errors = {
        "status": "descriptive_only",
        "false_positives": [predictions[0]] if with_errors else [],
        "false_negatives": [predictions[1]] if with_errors else [],
    }
    (reports_path / "error_review.json").write_text(json.dumps(errors), encoding="utf-8")
    package_manifest = {
        "model_id": model_id,
        "feature_schema_version": "features-1",
        "feature_schema_sha256": "schema",
        "split_manifest_sha256": "split",
        "detector_config_sha256": "detector",
        "feature_config_sha256": "feature",
        "decision_threshold": threshold,
    }
    run_manifest = {
        "run_id": "run-test-evaluate",
        "status": "completed",
        "selected_model_id": model_id,
        "feature_schema_version": "features-1",
        "feature_schema_sha256": "schema",
        "split_manifest_sha256": "split",
        "detector": {"config_sha256": "detector"},
        "threshold": threshold,
        "sample_counts": {"test": 2},
        "completed_at_utc": "2026-09-29T00:00:00Z",
    }
    (run_path / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    return run_path, package_manifest


def test_load_evaluation_report_validates_metrics_and_predictions(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)

    report = load_evaluation_report(run_path, package_manifest)

    assert report.run_id == "run-test-evaluate"
    assert report.model_id == "model-test"
    assert report.metrics["sample_count"] == 2
    assert report.confusion_matrix == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert len(report.predictions) == 2
    assert report.errors == []


def test_reference_checksum_reuse_requires_exact_predictions(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    reference = load_evaluation_report(run_path, package_manifest)
    current_predictions = [dict(row) for row in reference.predictions]

    checksums = _reference_input_checksums(reference, current_predictions)

    assert checksums == {"sample-0": f"{1:064x}", "sample-1": f"{2:064x}"}
    current_predictions[0]["score"] = 0.2
    with pytest.raises(ValueError, match="score does not match"):
        _reference_input_checksums(reference, current_predictions)


def test_report_loader_validates_reference_checksum_lineage(tmp_path: Path):
    reference_dir, package_manifest = _build_report(tmp_path / "reference")
    reference_run_manifest_path = reference_dir / "run_manifest.json"
    reference_manifest = json.loads(reference_run_manifest_path.read_text(encoding="utf-8"))
    reference_manifest["run_id"] = "run-reference"
    reference_run_manifest_path.write_text(json.dumps(reference_manifest), encoding="utf-8")
    reference = load_evaluation_report(reference_dir, package_manifest)

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    current_manifest = {
        "run_id": "run-current",
        "status": "completed",
        "selected_model_id": package_manifest["model_id"],
        "feature_schema_sha256": "schema",
        "split_manifest_sha256": "split",
        "detector": {"config_sha256": "detector"},
        "threshold": 0.5,
    }
    metadata = {
        "input_checksum_provenance": "validated_reference_evaluation",
        "reference_evaluation_run_path": str(reference_dir),
        "reference_evaluation_run_id": "run-reference",
        "reference_test_predictions_sha256": sha256_file(reference_dir / "reports" / "test_predictions.csv"),
    }

    _validate_reference_checksum_lineage(current_dir, current_manifest, metadata, reference.predictions)

    metadata["reference_evaluation_run_path"] = str(reference_dir).replace("/", "\\")
    _validate_reference_checksum_lineage(current_dir, current_manifest, metadata, reference.predictions)

    metadata["reference_test_predictions_sha256"] = "0" * 64
    with pytest.raises(EvaluationReportError, match="prediction hash mismatch"):
        _validate_reference_checksum_lineage(current_dir, current_manifest, metadata, reference.predictions)


def test_load_evaluation_report_returns_descriptive_errors(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path, with_errors=True)

    report = load_evaluation_report(run_path, package_manifest)

    assert {row["sample_id"] for row in report.errors} == {"sample-0", "sample-1"}
    assert report.confusion_matrix == {"tn": 0, "fp": 1, "fn": 1, "tp": 0}


def test_load_evaluation_report_rejects_model_mismatch(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    package_manifest["model_id"] = "different-model"

    with pytest.raises(EvaluationReportError, match="model ID"):
        load_evaluation_report(run_path, package_manifest)


def test_load_evaluation_report_rejects_threshold_label_drift(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    predictions_path = run_path / "reports" / "test_predictions.csv"
    rows = list(csv.DictReader(predictions_path.open(encoding="utf-8", newline="")))
    rows[0]["predicted_label"] = "ai_generated"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EvaluationReportError, match="threshold rule"):
        load_evaluation_report(run_path, package_manifest)


def test_legacy_report_remains_loadable_after_evaluation_only_configuration_change(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    active_detector = {}
    active_features = {}
    detector_hash = sha256_json(active_detector)
    feature_hash = sha256_json(active_features)
    package_manifest["detector_config_sha256"] = detector_hash
    package_manifest["feature_config_sha256"] = feature_hash
    run_manifest_path = run_path / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["configuration_sha256"] = "older-overall-config"
    run_manifest["detector"]["config_sha256"] = detector_hash
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    report = load_evaluation_report(
        run_path,
        package_manifest,
        {"_sha256": "new-overall-config", "detector": active_detector, "features": active_features},
    )

    assert report.active_configuration_matches is False
    assert report.feature_analysis is None


def test_load_evaluation_report_validates_optional_feature_analysis_artifacts(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    package_manifest["feature_columns"] = ["geometry.signal"]
    package_manifest["feature_schema_sha256"] = "a" * 64
    package_manifest["split_manifest_sha256"] = "b" * 64
    influence = FeatureInfluenceRecord(
        feature="geometry.signal",
        group="geometry_density",
        permutation_f1_drop_mean=0.25,
        permutation_f1_drop_std=0.0,
        permutation_accuracy_drop_mean=0.25,
        permutation_accuracy_drop_std=0.0,
        permutation_abs_score_change_mean=0.3,
        non_ai_count=1,
        ai_count=1,
        non_ai_median=0.1,
        non_ai_iqr=0.0,
        ai_median=0.9,
        ai_iqr=0.0,
        non_ai_missing_rate=0.0,
        ai_missing_rate=0.0,
        cliffs_delta_ai_vs_non_ai=1.0,
        standardized_coefficient=1.2,
        missingness_indicator_coefficient=None,
        mean_abs_linear_contribution=0.6,
    )
    analysis = FeatureAnalysisReport(
        schema_version="feature-analysis-1",
        run_id="run-test-evaluate",
        model_id="model-test",
        model_family="logistic_regression",
        feature_schema_version="features-1",
        status="software_test",
        analysis_config={"permutation_repeats": 2, "random_seed": 42},
        test_sample_count=2,
        test_class_counts={"non_ai_generated": 1, "ai_generated": 1},
        group_summary=[{"feature_group": "all_enabled", "display_name": "All enabled groups", "status": "completed", "test_f1": 1.0}],
        feature_influence=[influence],
        redundancy=[],
        interpretation_notes=["Generated test fixture only."],
    )
    paths = write_feature_analysis(run_path / "reports", analysis)
    run_manifest_path = run_path / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["feature_schema_sha256"] = "a" * 64
    run_manifest["split_manifest_sha256"] = "b" * 64
    run_manifest["feature_analysis"] = {
        "status": analysis.status,
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "source_feature_matrix_path": "features/image_features.csv",
        "source_feature_matrix_sha256": "c" * 64,
        "source_feature_schema_path": "features/feature_schema.json",
        "source_feature_schema_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
    }
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    report = load_evaluation_report(run_path, package_manifest)

    assert report.feature_analysis == analysis
    assert report.feature_analysis.feature_influence[0].feature == "geometry.signal"


def test_discover_evaluation_runs_filters_by_model_and_report_artifacts(tmp_path: Path):
    run_path, package_manifest = _build_report(tmp_path)
    training_like = tmp_path / "training-run"
    training_like.mkdir()
    (training_like / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "selected_model_id": package_manifest["model_id"]}),
        encoding="utf-8",
    )

    discovered = discover_evaluation_runs(tmp_path, package_manifest["model_id"])

    assert discovered == [run_path]


def test_streamlit_evaluation_report_renders_when_a_matching_artifact_exists():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    assert not any(widget.label.startswith("Use deterministic fixture detector") for widget in app.checkbox)
    if not any(widget.label == "Evaluation run" for widget in app.selectbox):
        pytest.skip("The workspace has no completed evaluation artifact")
    assert {metric.label for metric in app.metric} >= {"Accuracy", "Precision", "Recall", "F1 score"}
    assert any(button.label == "Download metrics JSON" for button in app.download_button)
    assert any("held-out evaluation run" in item.value for item in app.success)
    assert any("not a recalculation for the image uploaded above" in item.value for item in app.caption)
    if any(button.label == "Download feature analysis JSON" for button in app.download_button):
        status_messages = [item.value for item in [*app.warning, *app.info]]
        assert any(
            marker in message
            for message in status_messages
            for marker in (
                "Post-hoc pilot analysis",
                "generated-fixture software checks",
                "marked predeclared",
            )
        )
        assert any(button.label == "Download feature influence CSV" for button in app.download_button)
