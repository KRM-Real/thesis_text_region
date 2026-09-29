"""Local explanations for supported linear model packages."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models.train import FittedCandidate
from ..evaluation.feature_analysis import linear_component_mapping


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    group: str
    component: str
    raw_value: float | None
    transformed_value: float
    coefficient: float
    contribution: float


@dataclass(frozen=True)
class PredictionExplanation:
    schema_version: str
    traceability: dict[str, str]
    status: str
    method: str
    model_family: str
    decision_scale: str
    score: float
    threshold: float
    predicted_label: str
    baseline_to_threshold: float | None
    decision_margin: float | None
    margin_over_threshold: float | None
    feature_contribution_sum: float | None
    remaining_contribution: float | None
    score_reconstructed: float | None
    score_reconstruction_error: float | None
    top_toward_ai: list[FeatureContribution]
    top_toward_non_ai: list[FeatureContribution]
    group_contributions: list[dict[str, Any]]
    feature_values: list[dict[str, Any]]
    reason: str | None
    interpretation_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _logit(probability: float) -> float:
    if probability <= 0:
        return -math.inf
    if probability >= 1:
        return math.inf
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _feature_groups(schema: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(entry["name"]): str(entry["group"])
        for entry in schema
        if isinstance(entry, dict) and entry.get("name") and entry.get("group")
    }


def explain_prediction(
    candidate: FittedCandidate,
    values: dict[str, Any],
    schema: list[dict[str, Any]],
    score: float,
    predicted_label: str,
    *,
    top_n: int = 5,
    traceability: dict[str, str] | None = None,
) -> PredictionExplanation:
    """Return exact margin contributions for LR/linear SVM or safe global-only status."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    columns = list(candidate.feature_columns)
    lineage = {str(key): str(value) for key, value in (traceability or {}).items() if value not in (None, "")}
    groups = _feature_groups(schema)
    feature_values = [
        {"feature": column, "group": groups.get(column, "unavailable"), "value": _finite(values.get(column))}
        for column in columns
    ]
    mapping = linear_component_mapping(candidate)
    if candidate.family not in {"logistic_regression", "svm"} or mapping is None:
        if candidate.family == "random_forest":
            reason = "Random Forest has no additive linear explanation in this prototype. Review the measured feature values and global permutation analysis."
        else:
            reason = "This nonlinear or unsupported model has no additive local explanation in this prototype. Review the measured feature values and global permutation analysis."
        return PredictionExplanation(
            schema_version="prediction-explanation-2",
            traceability=lineage,
            status="global_only",
            method="global_permutation_and_feature_values",
            model_family=candidate.family,
            decision_scale="unavailable",
            score=float(score),
            threshold=float(candidate.threshold),
            predicted_label=predicted_label,
            baseline_to_threshold=None,
            decision_margin=None,
            margin_over_threshold=None,
            feature_contribution_sum=None,
            remaining_contribution=None,
            score_reconstructed=None,
            score_reconstruction_error=None,
            top_toward_ai=[],
            top_toward_non_ai=[],
            group_contributions=[],
            feature_values=feature_values,
            reason=reason,
            interpretation_note="Measured visual values and global importance are descriptive. They do not prove why the image was generated.",
        )

    missing_groups = [column for column in columns if column not in groups]
    if missing_groups:
        return PredictionExplanation(
            schema_version="prediction-explanation-2",
            traceability=lineage,
            status="unavailable",
            method="linear_contributions",
            model_family=candidate.family,
            decision_scale="unavailable",
            score=float(score),
            threshold=float(candidate.threshold),
            predicted_label=predicted_label,
            baseline_to_threshold=None,
            decision_margin=None,
            margin_over_threshold=None,
            feature_contribution_sum=None,
            remaining_contribution=None,
            score_reconstructed=None,
            score_reconstruction_error=None,
            top_toward_ai=[],
            top_toward_non_ai=[],
            group_contributions=[],
            feature_values=feature_values,
            reason="The fitted package feature columns do not all have registered groups; a reliable explanation cannot be produced.",
            interpretation_note="No feature-level contribution is shown because the feature mapping is incomplete.",
        )

    estimator = candidate.estimator
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    intercept_values = np.asarray(estimator.intercept_, dtype=float).reshape(-1)
    if intercept_values.size != 1 or len(mapping) != coefficients.size or list(getattr(estimator, "classes_", [])) != [0, 1]:
        reason = "The fitted linear model and preprocessing columns are incompatible; no contribution is shown."
        return PredictionExplanation(
            schema_version="prediction-explanation-2", traceability=lineage, status="unavailable", method="linear_contributions",
            model_family=candidate.family, decision_scale="unavailable", score=float(score), threshold=float(candidate.threshold),
            predicted_label=predicted_label, baseline_to_threshold=None, decision_margin=None,
            margin_over_threshold=None, feature_contribution_sum=None, remaining_contribution=None,
            score_reconstructed=None, score_reconstruction_error=None, top_toward_ai=[], top_toward_non_ai=[],
            group_contributions=[], feature_values=feature_values, reason=reason,
            interpretation_note="No feature-level contribution is shown because the fitted schema could not be verified.",
        )
    row = pd.DataFrame([{column: values[column] for column in columns}], columns=columns)
    transformed = np.asarray(candidate.preprocessor.transform(row), dtype=float)
    if transformed.ndim != 2 or transformed.shape != (1, coefficients.size):
        raise ValueError("transformed inference row does not match the fitted linear model")
    intercept = float(intercept_values[0])
    contributions_array = transformed[0] * coefficients
    margin = float(intercept + contributions_array.sum())
    fitted_margin = float(np.asarray(estimator.decision_function(transformed), dtype=float).reshape(-1)[0])
    if not math.isclose(margin, fitted_margin, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("linear feature contributions do not reconstruct the fitted model margin")

    family = candidate.family
    if family == "logistic_regression":
        decision_scale = "log_odds"
        cutoff = _logit(float(candidate.threshold))
        reconstructed_score = _sigmoid(margin)
        estimator_classes = list(getattr(estimator, "classes_", []))
        if estimator_classes != [0, 1]:
            raise ValueError("logistic model class order is incompatible with the binary label contract")
        error = abs(reconstructed_score - float(score))
        if not math.isclose(reconstructed_score, float(score), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("linear explanation does not reconstruct the logistic model score")
        scale_note = "Contributions add in log-odds space. The displayed probability is the logistic transform of that sum, so probability contributions are not additive."
    else:
        decision_scale = "decision_margin"
        cutoff = float(candidate.threshold)
        reconstructed_score = margin
        error = abs(reconstructed_score - float(score))
        if not math.isclose(reconstructed_score, float(score), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("linear explanation does not reconstruct the SVM decision score")
        scale_note = "Contributions add in the SVM decision-margin scale; positive values move toward AI-generated and negative values toward non-AI-generated."

    contribution_records: list[FeatureContribution] = []
    grouped: dict[str, float] = {}
    for index, (feature, component) in enumerate(mapping):
        raw_value = _finite(values.get(feature))
        contribution = float(contributions_array[index])
        contribution_records.append(FeatureContribution(
            feature=feature,
            group=groups[feature],
            component=component,
            raw_value=raw_value,
            transformed_value=float(transformed[0, index]),
            coefficient=float(coefficients[index]),
            contribution=contribution,
        ))
        grouped[groups[feature]] = grouped.get(groups[feature], 0.0) + contribution
    positive = sorted((item for item in contribution_records if item.contribution > 0), key=lambda item: (-item.contribution, item.feature, item.component))
    negative = sorted((item for item in contribution_records if item.contribution < 0), key=lambda item: (item.contribution, item.feature, item.component))
    selected_ids = {(item.feature, item.component) for item in (*positive[:top_n], *negative[:top_n])}
    remaining = float(sum(item.contribution for item in contribution_records if (item.feature, item.component) not in selected_ids))
    grouped_rows = [
        {"group": group, "contribution": float(value), "direction": "AI-generated" if value >= 0 else "Non-AI-generated"}
        for group, value in sorted(grouped.items(), key=lambda item: (-abs(item[1]), item[0]))
    ]
    return PredictionExplanation(
        schema_version="prediction-explanation-2",
        traceability=lineage,
        status="available",
        method="fitted_linear_model_contributions",
        model_family=family,
        decision_scale=decision_scale,
        score=float(score),
        threshold=float(candidate.threshold),
        predicted_label=predicted_label,
        baseline_to_threshold=_finite(intercept - cutoff),
        decision_margin=margin,
        margin_over_threshold=_finite(margin - cutoff),
        feature_contribution_sum=float(contributions_array.sum()),
        remaining_contribution=remaining,
        score_reconstructed=float(reconstructed_score),
        score_reconstruction_error=float(error),
        top_toward_ai=positive[:top_n],
        top_toward_non_ai=negative[:top_n],
        group_contributions=grouped_rows,
        feature_values=feature_values,
        reason=None,
        interpretation_note=scale_note + " Contributions are fitted associations, not causes or proof of origin.",
    )
