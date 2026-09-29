"""Post-hoc global diagnostics for the frozen image-level classifier."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..models.train import FittedCandidate, predict_scores
from ..reproducibility import write_json


ANALYSIS_SCHEMA_VERSION = "feature-analysis-1"
GROUP_LABELS = {
    "geometry_density": "Geometry and density",
    "geometry_plus_spacing": "Geometry plus spacing and alignment",
    "stroke_shape": "Stroke and shape",
    "edge_contrast_sharpness": "Edges, gradients, contrast, and sharpness",
    "texture": "Texture and local pixel relationships",
    "all_enabled": "All enabled groups (reference)",
}
GROUP_MEMBERS = {
    "geometry_density": {"geometry_density"},
    "geometry_plus_spacing": {"geometry_density", "spacing_alignment"},
    "stroke_shape": {"stroke_shape"},
    "edge_contrast_sharpness": {"edge_gradient", "contrast_sharpness"},
    "texture": {"texture"},
}
ALLOWED_ANALYSIS_STATUSES = {"posthoc_pilot", "predeclared_final", "software_test"}


@dataclass(frozen=True)
class FeatureInfluenceRecord:
    feature: str
    group: str
    permutation_f1_drop_mean: float | None
    permutation_f1_drop_std: float | None
    permutation_accuracy_drop_mean: float | None
    permutation_accuracy_drop_std: float | None
    permutation_abs_score_change_mean: float | None
    non_ai_count: int
    ai_count: int
    non_ai_median: float | None
    non_ai_iqr: float | None
    ai_median: float | None
    ai_iqr: float | None
    non_ai_missing_rate: float | None
    ai_missing_rate: float | None
    cliffs_delta_ai_vs_non_ai: float | None
    standardized_coefficient: float | None
    missingness_indicator_coefficient: float | None
    mean_abs_linear_contribution: float | None


@dataclass(frozen=True)
class FeatureAnalysisReport:
    schema_version: str
    run_id: str
    model_id: str
    model_family: str
    feature_schema_version: str
    status: str
    analysis_config: dict[str, Any]
    test_sample_count: int
    test_class_counts: dict[str, int]
    group_summary: list[dict[str, Any]]
    feature_influence: list[FeatureInfluenceRecord]
    redundancy: list[dict[str, Any]]
    interpretation_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureAnalysisReport:
        if value.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            raise ValueError("feature analysis uses an unsupported schema version")
        status = str(value.get("status", ""))
        if status not in ALLOWED_ANALYSIS_STATUSES:
            raise ValueError("feature analysis has an unsupported interpretation status")
        try:
            features = [FeatureInfluenceRecord(**item) for item in value["feature_influence"]]
            groups = list(value["group_summary"])
            redundancy = list(value["redundancy"])
            test_count = int(value["test_sample_count"])
            counts = {str(key): int(count) for key, count in value["test_class_counts"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feature analysis artifact is incomplete or malformed") from exc
        if test_count < 0 or sum(counts.values()) != test_count or set(counts) != {"non_ai_generated", "ai_generated"}:
            raise ValueError("feature analysis class counts do not match its sample count")
        if any(not feature.feature or not feature.group for feature in features):
            raise ValueError("feature analysis contains an unnamed feature or group")
        return cls(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            run_id=str(value.get("run_id", "")),
            model_id=str(value.get("model_id", "")),
            model_family=str(value.get("model_family", "")),
            feature_schema_version=str(value.get("feature_schema_version", "")),
            status=status,
            analysis_config=dict(value.get("analysis_config", {})),
            test_sample_count=test_count,
            test_class_counts=counts,
            group_summary=groups,
            feature_influence=features,
            redundancy=redundancy,
            interpretation_notes=[str(note) for note in value.get("interpretation_notes", [])],
        )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quartiles(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return None, None
    return float(np.median(numeric)), float(np.percentile(numeric, 75) - np.percentile(numeric, 25))


def _cliffs_delta(ai_values: pd.Series, non_ai_values: pd.Series) -> float | None:
    ai = pd.to_numeric(ai_values, errors="coerce").dropna().to_numpy(dtype=float)
    non_ai = pd.to_numeric(non_ai_values, errors="coerce").dropna().to_numpy(dtype=float)
    if ai.size == 0 or non_ai.size == 0:
        return None
    ordered = np.sort(non_ai)
    ai_greater = np.searchsorted(ordered, ai, side="left")
    ai_less = non_ai.size - np.searchsorted(ordered, ai, side="right")
    return float(np.mean((ai_greater - ai_less) / non_ai.size))


def linear_component_mapping(candidate: FittedCandidate) -> list[tuple[str, str]] | None:
    """Map transformed columns to source feature and component names."""

    estimator = candidate.estimator
    if candidate.family == "logistic_regression":
        pass
    elif candidate.family == "svm" and candidate.hyperparameters.get("kernel") == "linear":
        pass
    else:
        return None
    if not hasattr(estimator, "coef_") or np.asarray(estimator.coef_).shape[0] != 1:
        return None
    steps = getattr(candidate.preprocessor, "named_steps", {})
    imputer = steps.get("imputer")
    if imputer is None:
        return None
    indicators = getattr(getattr(imputer, "indicator_", None), "features_", np.asarray([], dtype=int))
    mapping = [(name, "value") for name in candidate.feature_columns]
    mapping.extend((candidate.feature_columns[int(index)], "missingness_indicator") for index in indicators)
    if len(mapping) != np.asarray(estimator.coef_).shape[1]:
        return None
    return mapping


def _linear_coefficients(candidate: FittedCandidate) -> tuple[list[tuple[str, str]], np.ndarray, float] | None:
    mapping = linear_component_mapping(candidate)
    if mapping is None:
        return None
    coefficients = np.asarray(candidate.estimator.coef_, dtype=float).reshape(-1)
    intercepts = np.asarray(candidate.estimator.intercept_, dtype=float).reshape(-1)
    if intercepts.size != 1 or not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        return None
    return mapping, coefficients, float(intercepts[0])


def _feature_group_summary(
    ablation_rows: list[dict[str, Any]],
    test: pd.DataFrame,
    group_by_column: dict[str, str],
) -> list[dict[str, Any]]:
    full_row = next(
        (row for row in ablation_rows if row.get("feature_group") == "all_enabled" and row.get("status") == "completed"),
        None,
    )
    full_f1 = _finite(full_row.get("test_f1")) if full_row else None
    no_detection_rate = _finite(pd.to_numeric(test.get("quality.no_detection"), errors="coerce").mean()) if "quality.no_detection" in test else None
    result: list[dict[str, Any]] = []
    for source in ablation_rows:
        name = str(source.get("feature_group", ""))
        members = GROUP_MEMBERS.get(name, set(group_by_column.values()) if name == "all_enabled" else set())
        columns = [column for column, group in group_by_column.items() if name == "all_enabled" or group in members]
        present = [column for column in columns if column in test]
        missing_rate = None
        if present and len(test):
            missing_rate = float(test[present].apply(pd.to_numeric, errors="coerce").isna().to_numpy().mean())
        row: dict[str, Any] = {
            "feature_group": name,
            "display_name": GROUP_LABELS.get(name, name.replace("_", " ").title()),
            "feature_count": int(source.get("feature_count", len(present)) or 0),
            "model_family": str(source.get("model_family", "")),
            "status": str(source.get("status", "unavailable")),
            "threshold": _finite(source.get("threshold")),
            "validation_f1": _finite(source.get("validation_f1")),
            "test_accuracy": _finite(source.get("test_accuracy")),
            "test_precision": _finite(source.get("test_precision")),
            "test_recall": _finite(source.get("test_recall")),
            "test_f1": _finite(source.get("test_f1")),
            "delta_test_f1_vs_all": None,
            "test_missing_cell_rate": missing_rate,
            "test_no_detection_rate": no_detection_rate,
        }
        for count_name in ("test_sample_count", "test_tn", "test_fp", "test_fn", "test_tp"):
            try:
                row[count_name] = int(source[count_name]) if source.get(count_name) not in (None, "") else None
            except (TypeError, ValueError):
                row[count_name] = None
        if full_f1 is not None and row["test_f1"] is not None:
            row["delta_test_f1_vs_all"] = row["test_f1"] - full_f1
        result.append(row)
    order = {name: index for index, name in enumerate((*GROUP_LABELS.keys(),))}
    return sorted(result, key=lambda row: (order.get(row["feature_group"], len(order)), row["feature_group"]))


def _redundancy_pairs(
    development: pd.DataFrame,
    columns: list[str],
    group_by_column: dict[str, str],
    threshold: float,
) -> list[dict[str, Any]]:
    if len(columns) < 2 or len(development) < 2:
        return []
    correlations = development[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman", min_periods=2)
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = _finite(correlations.loc[left, right])
            if value is not None and abs(value) >= threshold:
                result.append({
                    "feature_a": left,
                    "group_a": group_by_column[left],
                    "feature_b": right,
                    "group_b": group_by_column[right],
                    "spearman_r": value,
                    "absolute_spearman_r": abs(value),
                    "source_split": "train_and_validation",
                })
    return sorted(result, key=lambda row: (-row["absolute_spearman_r"], row["feature_a"], row["feature_b"]))


def compute_feature_analysis(
    frame: pd.DataFrame,
    schema: dict[str, Any],
    candidate: FittedCandidate,
    ablation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    run_id: str,
    model_id: str,
) -> FeatureAnalysisReport:
    """Compute fixed-model test diagnostics and development-only redundancy."""

    status = str(settings.get("status", "posthoc_pilot"))
    if status not in ALLOWED_ANALYSIS_STATUSES:
        raise ValueError(f"unsupported feature analysis status: {status}")
    repeats = int(settings.get("permutation_repeats", 20))
    seed = int(settings.get("random_seed", 42))
    top_count = int(settings.get("top_global_features", 15))
    threshold = float(settings.get("correlation_threshold", 0.90))
    if repeats < 1 or top_count < 1 or not 0 < threshold <= 1:
        raise ValueError("feature analysis settings are outside their valid ranges")
    if str(settings.get("correlation_method", "spearman")) != "spearman":
        raise ValueError("only Spearman correlation is supported for redundancy diagnostics")

    columns = list(candidate.feature_columns)
    schema_columns = list(schema.get("model_columns", []))
    entries = schema.get("columns", [])
    group_by_column = {
        str(entry.get("name")): str(entry.get("group"))
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") and entry.get("group")
        and entry.get("model_input", True) and not entry.get("diagnostic_only", False)
    }
    if not columns or columns != schema_columns:
        raise ValueError("feature analysis model columns do not match the locked feature schema")
    if set(columns) - set(group_by_column) or set(columns) - set(frame.columns):
        raise ValueError("feature analysis is missing a model feature or its registered group")
    if not {"label", "split"}.issubset(frame.columns):
        raise ValueError("feature analysis requires image-level labels and split assignments")
    test = frame[frame["split"].astype(str) == "test"].copy()
    development = frame[frame["split"].astype(str).isin({"train", "validation"})].copy()
    if test.empty or development.empty:
        raise ValueError("feature analysis requires test and development rows")
    labels = pd.to_numeric(test["label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("feature analysis labels must be numeric binary image labels")
    y_true = labels.astype(int)
    if np.unique(y_true).size != 2:
        raise ValueError("feature analysis test split must contain both classes")

    test_x = test[columns].apply(pd.to_numeric, errors="coerce")
    base_scores = predict_scores(candidate, test_x)
    baseline_prediction = base_scores >= float(candidate.threshold)
    baseline_f1 = float(f1_score(y_true, baseline_prediction, zero_division=0))
    baseline_accuracy = float(accuracy_score(y_true, baseline_prediction))
    repeats_rng = np.random.default_rng(seed)
    linear = _linear_coefficients(candidate)
    linear_by_component: dict[tuple[str, str], float] = {}
    mean_abs_by_feature: dict[str, float] = {}
    if linear is not None:
        component_mapping, coefficients, _ = linear
        transformed = np.asarray(candidate.preprocessor.transform(test_x), dtype=float)
        if transformed.ndim != 2 or transformed.shape[1] != len(component_mapping):
            raise ValueError("fitted preprocessing output does not match the linear model coefficients")
        contributions = transformed * coefficients.reshape(1, -1)
        grouped_contributions: dict[str, np.ndarray] = {}
        for index, (feature, component) in enumerate(component_mapping):
            linear_by_component[(feature, component)] = float(coefficients[index])
            grouped_contributions.setdefault(feature, np.zeros(len(test), dtype=float))
            grouped_contributions[feature] += contributions[:, index]
        mean_abs_by_feature = {name: float(np.abs(values).mean()) for name, values in grouped_contributions.items()}

    ai_mask = y_true == 1
    non_ai_mask = y_true == 0
    ai_count, non_ai_count = int(ai_mask.sum()), int(non_ai_mask.sum())
    influence: list[FeatureInfluenceRecord] = []
    for column in columns:
        values = test_x[column]
        ai_values, non_ai_values = values.iloc[ai_mask], values.iloc[non_ai_mask]
        ai_median, ai_iqr = _quartiles(ai_values)
        non_ai_median, non_ai_iqr = _quartiles(non_ai_values)
        f1_drops: list[float] = []
        accuracy_drops: list[float] = []
        absolute_score_changes: list[float] = []
        original_values = values.to_numpy(copy=True)
        for _ in range(repeats):
            permuted = test_x.copy()
            permuted[column] = original_values[repeats_rng.permutation(len(original_values))]
            scores = predict_scores(candidate, permuted)
            predictions = scores >= float(candidate.threshold)
            f1_drops.append(baseline_f1 - float(f1_score(y_true, predictions, zero_division=0)))
            accuracy_drops.append(baseline_accuracy - float(accuracy_score(y_true, predictions)))
            absolute_score_changes.append(float(np.mean(np.abs(base_scores - scores))))
        f1_array = np.asarray(f1_drops, dtype=float)
        accuracy_array = np.asarray(accuracy_drops, dtype=float)
        ai_missing = float(values.iloc[ai_mask].isna().mean())
        non_ai_missing = float(values.iloc[non_ai_mask].isna().mean())
        influence.append(FeatureInfluenceRecord(
            feature=column,
            group=group_by_column[column],
            permutation_f1_drop_mean=float(f1_array.mean()),
            permutation_f1_drop_std=float(f1_array.std(ddof=0)),
            permutation_accuracy_drop_mean=float(accuracy_array.mean()),
            permutation_accuracy_drop_std=float(accuracy_array.std(ddof=0)),
            permutation_abs_score_change_mean=float(np.mean(absolute_score_changes)),
            non_ai_count=non_ai_count,
            ai_count=ai_count,
            non_ai_median=non_ai_median,
            non_ai_iqr=non_ai_iqr,
            ai_median=ai_median,
            ai_iqr=ai_iqr,
            non_ai_missing_rate=non_ai_missing,
            ai_missing_rate=ai_missing,
            cliffs_delta_ai_vs_non_ai=_cliffs_delta(ai_values, non_ai_values),
            standardized_coefficient=linear_by_component.get((column, "value")),
            missingness_indicator_coefficient=linear_by_component.get((column, "missingness_indicator")),
            mean_abs_linear_contribution=mean_abs_by_feature.get(column),
        ))
    influence.sort(key=lambda item: (
        -(item.permutation_f1_drop_mean if item.permutation_f1_drop_mean is not None else -math.inf),
        -(item.permutation_abs_score_change_mean if item.permutation_abs_score_change_mean is not None else -math.inf),
        item.feature,
    ))

    group_summary = _feature_group_summary(ablation_rows, test, group_by_column)
    redundancy = _redundancy_pairs(development, columns, group_by_column, threshold)
    class_counts = {
        "non_ai_generated": non_ai_count,
        "ai_generated": ai_count,
    }
    effective_settings = {
        "status": status,
        "permutation_repeats": repeats,
        "random_seed": seed,
        "top_global_features": top_count,
        "top_local_features": int(settings.get("top_local_features", 5)),
        "correlation_method": "spearman",
        "correlation_threshold": threshold,
        "permutation_ranking_metric": "mean_test_f1_decrease",
        "permutation_split": "test",
        "redundancy_split": "train_and_validation",
    }
    notes = [
        "Feature-group ablation is the primary feature-usefulness comparison for the thesis objectives.",
        "Individual permutation results and class-separation summaries are descriptive for this held-out split and were not used to fit, tune, or select the model.",
        "Linear coefficients and local contributions describe fitted model associations; they are not causal effects or proof of image origin.",
        "Correlated features can share or obscure importance; redundancy pairs use development rows only.",
        "Quality and no-detection indicators describe processing conditions and are not authenticity evidence.",
        "This report is limited to the documented receipt dataset, model, configuration, and split.",
    ]
    return FeatureAnalysisReport(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        run_id=run_id,
        model_id=model_id,
        model_family=candidate.family,
        feature_schema_version=str(schema.get("schema_version", "")),
        status=status,
        analysis_config=effective_settings,
        test_sample_count=len(test),
        test_class_counts=class_counts,
        group_summary=group_summary,
        feature_influence=influence,
        redundancy=redundancy,
        interpretation_notes=notes,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_feature_analysis(output_dir: str | Path, report: FeatureAnalysisReport) -> dict[str, Path]:
    """Write versioned JSON and CSV artifacts; return their exact paths."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "feature_analysis.json"
    influence_path = target / "feature_influence.csv"
    groups_path = target / "feature_group_summary.csv"
    redundancy_path = target / "feature_redundancy.csv"
    write_json(json_path, report.to_dict())
    influence_rows = [asdict(row) for row in report.feature_influence]
    influence_fields = list(FeatureInfluenceRecord.__dataclass_fields__)
    _write_csv(influence_path, influence_rows, influence_fields)
    group_fields = sorted({key for row in report.group_summary for key in row})
    _write_csv(groups_path, report.group_summary, group_fields)
    redundancy_fields = ["feature_a", "group_a", "feature_b", "group_b", "spearman_r", "absolute_spearman_r", "source_split"]
    _write_csv(redundancy_path, report.redundancy, redundancy_fields)
    return {
        "feature_analysis.json": json_path,
        "feature_influence.csv": influence_path,
        "feature_group_summary.csv": groups_path,
        "feature_redundancy.csv": redundancy_path,
    }


def human_summary(report: FeatureAnalysisReport) -> str:
    """Create concise, bounded-language feature interpretation text."""

    completed = [row for row in report.group_summary if row.get("status") == "completed" and row.get("test_f1") is not None]
    completed.sort(key=lambda row: (-float(row["test_f1"]), str(row["feature_group"])))
    top = report.feature_influence[: int(report.analysis_config.get("top_global_features", 15))]
    strongest = top[0] if top else None
    lines = [
        "## Feature interpretation",
        "",
        f"Interpretation status: `{report.status}`. Feature-group comparison is the primary evidence; individual-feature rankings are supporting diagnostics.",
        "",
        "Permutation importance describes how this frozen model's held-out F1 changed when one feature column was shuffled. It is descriptive and was not used to fit, tune, or select the model.",
    ]
    if completed:
        lines.extend(["", "### Feature groups", ""])
        for row in report.group_summary:
            if row.get("status") != "completed":
                continue
            f1_value = row.get("test_f1")
            delta = row.get("delta_test_f1_vs_all")
            f1_text = "unavailable" if f1_value is None else f"{float(f1_value):.4f}"
            delta_text = "unavailable" if delta is None else f"{float(delta):+.4f}"
            lines.append(f"- {row['display_name']}: test F1 {f1_text}; difference from all enabled {delta_text}.")
    if strongest is not None:
        importance = strongest.permutation_f1_drop_mean
        lines.extend([
            "",
            "### Individual-feature diagnostic",
            "",
            f"The largest measured permutation F1 decrease was for `{strongest.feature}` ({strongest.group}), at {importance:.4f} on this test split. This ranking can be affected by correlated features and the selected model.",
        ])
    lines.extend(["", "### Limits", ""])
    lines.extend(f"- {note}" for note in report.interpretation_notes)
    return "\n".join(lines)
