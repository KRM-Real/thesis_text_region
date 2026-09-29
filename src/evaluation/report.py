"""Validated, read-only loading of evaluation artifacts for reporting clients."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..reproducibility import sha256_file, sha256_json
from .feature_analysis import FeatureAnalysisReport
from .metrics import evaluate_scores


REQUIRED_REPORT_FILES = (
    "run_manifest.json",
    "reports/test_metrics.json",
    "reports/confusion_matrix.csv",
    "reports/test_predictions.csv",
    "reports/error_review.json",
    "reports/summary.md",
)


class EvaluationReportError(ValueError):
    """Raised when an evaluation report is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class EvaluationReport:
    """A validated evaluation run safe for read-only presentation."""

    run_path: Path
    run_id: str
    model_id: str
    metrics: dict[str, Any]
    confusion_matrix: dict[str, int]
    predictions: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    summary_markdown: str
    run_manifest: dict[str, Any]
    feature_analysis: FeatureAnalysisReport | None = None
    feature_group_ablation: list[dict[str, Any]] = field(default_factory=list)
    active_configuration_matches: bool | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationReportError(f"Could not read JSON artifact: {path}") from exc


def _require_file(run_path: Path, relative: str) -> Path:
    path = run_path / relative
    if not path.is_file():
        raise EvaluationReportError(f"Evaluation report is incomplete; missing {relative}")
    return path


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationReportError(f"{name} must be a JSON object")
    return value


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if expected in (None, "") or actual in (None, ""):
        raise EvaluationReportError(f"Evaluation report is missing compatibility metadata: {name}")
    if actual != expected:
        raise EvaluationReportError(f"Evaluation report {name} does not match the selected model package")


def _require_float_equal(name: str, actual: Any, expected: Any) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise EvaluationReportError(f"Evaluation report contains an invalid {name}") from exc
    if not math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-12):
        raise EvaluationReportError(f"Evaluation report {name} does not match the selected model package")


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"1", "true", "yes"}:
        return True
    if str(value).strip().lower() in {"0", "false", "no", ""}:
        return False
    raise EvaluationReportError(f"Prediction field {field} is not boolean-like")


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise EvaluationReportError(f"Could not read prediction artifact: {path}") from exc
    required = {"sample_id", "true_label", "score", "predicted_label", "threshold", "region_count", "no_detection", "input_sha256"}
    if not rows or not required.issubset(rows[0]):
        raise EvaluationReportError("Prediction artifact is missing required fields")
    parsed: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            true_label = int(row["true_label"])
            score = float(row["score"])
            threshold = float(row["threshold"])
            region_count = float(row["region_count"])
        except (TypeError, ValueError) as exc:
            raise EvaluationReportError(f"Prediction row {row_number} contains an invalid numeric value") from exc
        if true_label not in {0, 1} or not math.isfinite(score) or not math.isfinite(threshold) or not math.isfinite(region_count):
            raise EvaluationReportError(f"Prediction row {row_number} contains an invalid value")
        predicted_label = str(row["predicted_label"])
        if predicted_label not in {"ai_generated", "non_ai_generated"}:
            raise EvaluationReportError(f"Prediction row {row_number} contains an invalid predicted label")
        parsed.append(
            {
                "sample_id": str(row["sample_id"]),
                "true_label": true_label,
                "score": score,
                "predicted_label": predicted_label,
                "threshold": threshold,
                "region_count": int(region_count) if region_count.is_integer() else region_count,
                "no_detection": _parse_bool(row["no_detection"], "no_detection"),
                "input_sha256": str(row["input_sha256"]),
            }
        )
    return parsed


def _validate_reference_checksum_lineage(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    """Fail closed if reused checksums no longer match their reference evaluation."""
    provenance = analysis_metadata.get("input_checksum_provenance")
    if provenance is None:
        return  # Earlier feature-analysis reports did not record this field.
    if provenance in {"audited_manifest", "not_reverified"}:
        return
    if provenance != "validated_reference_evaluation":
        raise EvaluationReportError("Feature analysis contains an unknown input-checksum provenance")

    raw_reference_path = analysis_metadata.get("reference_evaluation_run_path")
    reference_run_id = analysis_metadata.get("reference_evaluation_run_id")
    predictions_hash = analysis_metadata.get("reference_test_predictions_sha256")
    if not isinstance(raw_reference_path, str) or not raw_reference_path.strip() or not isinstance(reference_run_id, str) or not reference_run_id:
        raise EvaluationReportError("Feature analysis is missing its reference evaluation location or ID")
    if not isinstance(predictions_hash, str) or re.fullmatch(r"[0-9a-f]{64}", predictions_hash) is None:
        raise EvaluationReportError("Feature analysis has an invalid reference prediction hash")

    # Historical run manifests were created on Windows and may store relative
    # paths with backslashes. Normalize separators before resolving them on a
    # Linux Streamlit host.
    reference_dir = Path(raw_reference_path.replace("\\", "/"))
    if not reference_dir.is_absolute():
        reference_dir = Path.cwd() / reference_dir
    if reference_dir.resolve() == run_dir.resolve():
        raise EvaluationReportError("Feature analysis cannot reference its own evaluation run")
    manifest_path = reference_dir / "run_manifest.json"
    predictions_path = reference_dir / "reports" / "test_predictions.csv"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise EvaluationReportError("Feature analysis reference evaluation artifacts are missing")
    if sha256_file(predictions_path) != predictions_hash:
        raise EvaluationReportError("Feature analysis reference prediction hash mismatch")

    reference_manifest = _require_mapping(_read_json(manifest_path), "reference run_manifest")
    for field_name, expected in (
        ("run_id", reference_run_id),
        ("status", "completed"),
        ("selected_model_id", run_manifest.get("selected_model_id")),
        ("feature_schema_sha256", run_manifest.get("feature_schema_sha256")),
        ("split_manifest_sha256", run_manifest.get("split_manifest_sha256")),
    ):
        if reference_manifest.get(field_name) != expected:
            raise EvaluationReportError(f"Feature analysis reference evaluation {field_name} mismatch")
    reference_detector = reference_manifest.get("detector", {})
    current_detector = run_manifest.get("detector", {})
    if not isinstance(reference_detector, dict) or reference_detector.get("config_sha256") != current_detector.get("config_sha256"):
        raise EvaluationReportError("Feature analysis reference detector configuration mismatch")
    _require_float_equal("reference decision threshold", reference_manifest.get("threshold"), run_manifest.get("threshold"))

    reference_predictions = _read_predictions(predictions_path)
    reference_by_id = {row["sample_id"]: row for row in reference_predictions}
    current_by_id = {str(row["sample_id"]): row for row in predictions}
    if len(reference_by_id) != len(reference_predictions) or len(current_by_id) != len(predictions):
        raise EvaluationReportError("Feature analysis reference contains duplicate sample IDs")
    if reference_by_id.keys() != current_by_id.keys():
        raise EvaluationReportError("Feature analysis reference test sample IDs do not match current predictions")
    for sample_id, current in current_by_id.items():
        reference = reference_by_id[sample_id]
        if reference["true_label"] != current["true_label"] or reference["predicted_label"] != current["predicted_label"]:
            raise EvaluationReportError("Feature analysis reference labels do not match current predictions")
        for field_name in ("score", "threshold"):
            if not math.isclose(reference[field_name], current[field_name], rel_tol=1e-9, abs_tol=1e-12):
                raise EvaluationReportError(f"Feature analysis reference {field_name} does not match current predictions")
        checksum = reference["input_sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", checksum) is None or checksum != current["input_sha256"].lower():
            raise EvaluationReportError("Feature analysis reused an invalid or mismatched image checksum")


def _read_confusion_matrix(path: Path) -> dict[str, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise EvaluationReportError(f"Could not read confusion matrix artifact: {path}") from exc
    if len(rows) != 1 or not {"tn", "fp", "fn", "tp"}.issubset(rows[0]):
        raise EvaluationReportError("Confusion matrix artifact has an invalid shape")
    try:
        values = {key: int(rows[0][key]) for key in ("tn", "fp", "fn", "tp")}
    except (TypeError, ValueError) as exc:
        raise EvaluationReportError("Confusion matrix contains a non-integer count") from exc
    if any(value < 0 for value in values.values()):
        raise EvaluationReportError("Confusion matrix contains a negative count")
    return values


def _read_feature_group_ablation(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise EvaluationReportError("Could not read feature-group ablation artifact") from exc
    required = {"feature_group", "feature_count", "status"}
    if rows and not required.issubset(rows[0]):
        raise EvaluationReportError("Feature-group ablation artifact is missing required fields")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    numeric_fields = {
        "feature_count", "threshold", "validation_f1", "test_zero_division", "test_accuracy",
        "test_precision", "test_recall", "test_f1", "test_sample_count", "test_tn", "test_fp",
        "test_fn", "test_tp",
    }
    for row in rows:
        group = str(row.get("feature_group", ""))
        if not group or group in seen or row.get("status") not in {"completed", "unavailable"}:
            raise EvaluationReportError("Feature-group ablation artifact contains an invalid group row")
        seen.add(group)
        normalized: dict[str, Any] = dict(row)
        for field_name in numeric_fields & row.keys():
            raw = row[field_name]
            if raw in (None, ""):
                normalized[field_name] = None
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise EvaluationReportError("Feature-group ablation contains a non-numeric metric") from exc
            if not math.isfinite(number):
                raise EvaluationReportError("Feature-group ablation contains a non-finite metric")
            normalized[field_name] = int(number) if field_name in {"feature_count", "test_zero_division", "test_sample_count", "test_tn", "test_fp", "test_fn", "test_tp"} else number
        parsed.append(normalized)
    return parsed


def _read_feature_analysis(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    predictions: list[dict[str, Any]],
) -> FeatureAnalysisReport | None:
    analysis_path = run_dir / "reports" / "feature_analysis.json"
    sidecars = (
        "feature_influence.csv",
        "feature_group_summary.csv",
        "feature_redundancy.csv",
    )
    analysis_metadata = run_manifest.get("feature_analysis")
    present = [path.is_file() for path in (analysis_path, *(run_dir / "reports" / name for name in sidecars))]
    if not analysis_metadata and not any(present):
        return None
    if not isinstance(analysis_metadata, dict) or not all(present):
        raise EvaluationReportError("Feature analysis artifacts are incomplete or missing their run-manifest metadata")
    hashes = analysis_metadata.get("artifact_sha256")
    expected_files = {"feature_analysis.json", *sidecars}
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise EvaluationReportError("Feature analysis artifact hash manifest is incomplete")
    for name in expected_files:
        artifact = run_dir / "reports" / name
        expected_hash = str(hashes.get(name, ""))
        if not expected_hash or sha256_file(artifact) != expected_hash:
            raise EvaluationReportError(f"Feature analysis artifact hash mismatch: {name}")
    source_hashes = {
        "source_feature_matrix_sha256": analysis_metadata.get("source_feature_matrix_sha256"),
        "source_feature_schema_sha256": analysis_metadata.get("source_feature_schema_sha256"),
        "split_manifest_sha256": analysis_metadata.get("split_manifest_sha256"),
    }
    for name, digest in source_hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise EvaluationReportError(f"Feature analysis contains an invalid {name}")
    if source_hashes["source_feature_schema_sha256"] != run_manifest.get("feature_schema_sha256"):
        raise EvaluationReportError("Feature analysis input schema hash does not match the evaluation manifest")
    if source_hashes["split_manifest_sha256"] != run_manifest.get("split_manifest_sha256"):
        raise EvaluationReportError("Feature analysis split hash does not match the evaluation manifest")
    _validate_reference_checksum_lineage(run_dir, run_manifest, analysis_metadata, predictions)
    try:
        report = FeatureAnalysisReport.from_dict(_read_json(analysis_path))
    except (ValueError, TypeError) as exc:
        raise EvaluationReportError(f"Feature analysis artifact is invalid: {exc}") from exc
    if report.run_id != str(run_manifest.get("run_id", "")):
        raise EvaluationReportError("Feature analysis run ID does not match the evaluation manifest")
    if report.model_id != str(package_manifest.get("model_id", "")):
        raise EvaluationReportError("Feature analysis model ID does not match the selected model package")
    if report.status != str(analysis_metadata.get("status", "")):
        raise EvaluationReportError("Feature analysis status does not match the evaluation manifest")
    if report.feature_schema_version != str(package_manifest.get("feature_schema_version", "")):
        raise EvaluationReportError("Feature analysis schema does not match the selected model package")
    if report.test_sample_count != len(predictions):
        raise EvaluationReportError("Feature analysis sample count does not match prediction rows")
    package_columns = package_manifest.get("feature_columns")
    if isinstance(package_columns, list):
        features = [item.feature for item in report.feature_influence]
        if len(features) != len(set(features)) or set(features) != set(package_columns):
            raise EvaluationReportError("Feature analysis columns do not match the selected model package")
    return report


def _error_records(error_review: Mapping[str, Any], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if error_review.get("status") != "descriptive_only":
        raise EvaluationReportError("Error review must be marked descriptive_only")
    expected = {
        row["sample_id"]: row
        for row in predictions
        if (row["true_label"] == 0 and row["predicted_label"] == "ai_generated")
        or (row["true_label"] == 1 and row["predicted_label"] == "non_ai_generated")
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for error_type in ("false_positives", "false_negatives"):
        entries = error_review.get(error_type, [])
        if not isinstance(entries, list):
            raise EvaluationReportError(f"Error review field {error_type} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("sample_id", "")) not in expected:
                raise EvaluationReportError("Error review does not match the prediction artifact")
            sample_id = str(entry["sample_id"])
            if sample_id in seen:
                raise EvaluationReportError("Error review contains a duplicate sample")
            seen.add(sample_id)
            expected_row = expected[sample_id]
            for key in ("true_label", "predicted_label", "score", "threshold", "region_count", "no_detection", "input_sha256"):
                if key not in entry:
                    raise EvaluationReportError("Error review does not match the prediction artifact")
                if key in {"score", "threshold", "region_count"}:
                    try:
                        matches = math.isclose(float(entry[key]), float(expected_row[key]), rel_tol=1e-9, abs_tol=1e-12)
                    except (TypeError, ValueError):
                        matches = False
                elif key == "no_detection":
                    try:
                        matches = _parse_bool(entry[key], key) == bool(expected_row[key])
                    except EvaluationReportError:
                        matches = False
                else:
                    matches = str(entry[key]) == str(expected_row[key])
                if not matches:
                    raise EvaluationReportError("Error review does not match the prediction artifact")
            records.append(expected[sample_id])
    if seen != set(expected):
        raise EvaluationReportError("Error review does not contain every prediction error")
    return records


def _validate_metric_consistency(metrics: Mapping[str, Any], predictions: list[dict[str, Any]], confusion: Mapping[str, int]) -> None:
    required = {"positive_class", "threshold", "threshold_convention", "sample_count", "accuracy", "precision", "recall", "f1", "confusion_matrix"}
    if not required.issubset(metrics):
        raise EvaluationReportError("Test metrics artifact is missing required fields")
    if metrics["positive_class"] != "ai_generated" or metrics["threshold_convention"] != "score >= threshold => ai_generated":
        raise EvaluationReportError("Evaluation report uses an unsupported label or threshold convention")
    try:
        sample_count = int(metrics["sample_count"])
    except (TypeError, ValueError) as exc:
        raise EvaluationReportError("Metric sample count is invalid") from exc
    if sample_count != len(predictions):
        raise EvaluationReportError("Metric sample count does not match prediction rows")
    metric_confusion = metrics["confusion_matrix"]
    try:
        normalized_metric_confusion = {key: int(metric_confusion.get(key, -1)) for key in confusion} if isinstance(metric_confusion, dict) else {}
    except (TypeError, ValueError) as exc:
        raise EvaluationReportError("Metric confusion matrix contains an invalid count") from exc
    if normalized_metric_confusion != dict(confusion):
        raise EvaluationReportError("Metric confusion matrix does not match confusion_matrix.csv")
    scores = np.asarray([row["score"] for row in predictions], dtype=float)
    labels = np.asarray([row["true_label"] for row in predictions], dtype=int)
    threshold = float(metrics["threshold"])
    recomputed = evaluate_scores(labels, scores, threshold)
    for key in ("accuracy", "precision", "recall", "f1"):
        if not math.isclose(float(metrics[key]), float(recomputed[key]), rel_tol=1e-9, abs_tol=1e-12):
            raise EvaluationReportError(f"Stored metric {key} does not match prediction rows")
    for row in predictions:
        expected_label = "ai_generated" if row["score"] >= threshold else "non_ai_generated"
        if row["predicted_label"] != expected_label:
            raise EvaluationReportError("Prediction label does not follow the stored threshold rule")


def load_evaluation_report(
    run_path: str | Path,
    package_manifest: Mapping[str, Any],
    active_config: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Load and validate one completed evaluation run against a model package."""

    run_dir = Path(run_path)
    for relative in REQUIRED_REPORT_FILES:
        _require_file(run_dir, relative)
    run_manifest = _require_mapping(_read_json(run_dir / "run_manifest.json"), "run_manifest")
    metrics = _require_mapping(_read_json(run_dir / "reports" / "test_metrics.json"), "test_metrics")
    error_review = _require_mapping(_read_json(run_dir / "reports" / "error_review.json"), "error_review")
    if run_manifest.get("status") != "completed":
        raise EvaluationReportError("Evaluation run is not completed")
    _require_equal("model ID", run_manifest.get("selected_model_id"), package_manifest.get("model_id"))
    _require_equal("feature schema version", run_manifest.get("feature_schema_version"), package_manifest.get("feature_schema_version"))
    _require_equal("feature schema hash", run_manifest.get("feature_schema_sha256"), package_manifest.get("feature_schema_sha256"))
    _require_equal("split manifest hash", run_manifest.get("split_manifest_sha256"), package_manifest.get("split_manifest_sha256"))
    _require_equal("detector configuration hash", run_manifest.get("detector", {}).get("config_sha256"), package_manifest.get("detector_config_sha256"))
    _require_float_equal("decision threshold", run_manifest.get("threshold"), package_manifest.get("decision_threshold"))
    active_configuration_matches: bool | None = None
    if active_config is not None:
        expected_config_hash = active_config.get("_sha256")
        if expected_config_hash:
            # Evaluation-only settings can change without changing the fitted
            # detector, feature schema, or model package. Keep older compatible
            # reports loadable and surface the full-config difference to UI.
            active_configuration_matches = run_manifest.get("configuration_sha256") == expected_config_hash
        _require_equal("active detector configuration hash", run_manifest.get("detector", {}).get("config_sha256"), sha256_json(active_config.get("detector", {})))
        _require_equal("active feature configuration hash", package_manifest.get("feature_config_sha256"), sha256_json(active_config.get("features", {})))
    predictions = _read_predictions(run_dir / "reports" / "test_predictions.csv")
    confusion = _read_confusion_matrix(run_dir / "reports" / "confusion_matrix.csv")
    _validate_metric_consistency(metrics, predictions, confusion)
    sample_counts = run_manifest.get("sample_counts")
    if not isinstance(sample_counts, dict) or sample_counts.get("test") != len(predictions):
        raise EvaluationReportError("Run-manifest test count does not match prediction rows")
    errors = _error_records(error_review, predictions)
    feature_analysis = _read_feature_analysis(run_dir, run_manifest, package_manifest, predictions)
    feature_group_ablation = _read_feature_group_ablation(run_dir / "reports" / "feature_group_ablation.csv")
    try:
        summary = (run_dir / "reports" / "summary.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationReportError("Could not read evaluation summary") from exc
    run_id = str(run_manifest.get("run_id", run_dir.name))
    return EvaluationReport(
        run_path=run_dir,
        run_id=run_id,
        model_id=str(package_manifest["model_id"]),
        metrics=dict(metrics),
        confusion_matrix=confusion,
        predictions=predictions,
        errors=errors,
        summary_markdown=summary,
        run_manifest=run_manifest,
        feature_analysis=feature_analysis,
        feature_group_ablation=feature_group_ablation,
        active_configuration_matches=active_configuration_matches,
    )


def discover_evaluation_runs(artifacts_root: str | Path, model_id: str) -> list[Path]:
    """Return completed-looking evaluation runs for one selected package."""

    root = Path(artifacts_root)
    candidates: list[tuple[datetime, Path]] = []
    for manifest_path in root.glob("*/run_manifest.json"):
        try:
            manifest = _require_mapping(_read_json(manifest_path), "run_manifest")
        except EvaluationReportError:
            continue
        if manifest.get("status") != "completed" or manifest.get("selected_model_id") != model_id:
            continue
        reports_dir = manifest_path.parent / "reports"
        if not reports_dir.is_dir() or not (reports_dir / "test_metrics.json").is_file():
            continue
        raw_time = str(manifest.get("completed_at_utc") or manifest.get("started_at_utc") or "")
        try:
            completed = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        except ValueError:
            completed = datetime.min.replace(tzinfo=timezone.utc)
        candidates.append((completed, manifest_path.parent))
    return [path for _, path in sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)]
