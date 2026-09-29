"""Required metrics, confusion matrices, and bounded evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "positive_class": "ai_generated",
        "threshold": float(threshold),
        "threshold_convention": "score >= threshold => ai_generated",
        "zero_division": 0,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sample_count": int(len(y_true)),
    }


def write_evaluation_report(output_dir: str | Path, metrics: dict[str, Any], predictions: list[dict[str, Any]], summary: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = sorted({key for row in predictions for key in row}) if predictions else []
    with (target / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(predictions)
    (target / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
    with (target / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tn", "fp", "fn", "tp"])
        writer.writeheader()
        writer.writerow(metrics["confusion_matrix"])


def run_feature_group_ablation(
    frame: Any,
    schema: dict[str, Any],
    family: str,
    hyperparameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measure predeclared feature-group subsets on the same image split.

    The selected model family and settings are held fixed for this diagnostic;
    each subset still fits its imputer/scaler and threshold on train/validation
    only.  The test rows are touched only for the final per-subset metrics.
    """

    from ..models.train import fit_candidate, predict_scores, refit_selected

    group_sets = {
        "geometry_density": {"geometry_density"},
        "geometry_plus_spacing": {"geometry_density", "spacing_alignment"},
        "stroke_shape": {"stroke_shape"},
        "edge_contrast_sharpness": {"edge_gradient", "contrast_sharpness"},
        "texture": {"texture"},
        "all_enabled": None,
    }
    entries = schema.get("columns", [])
    model_columns = set(schema.get("model_columns", []))
    group_by_column = {entry.get("name"): entry.get("group") for entry in entries if entry.get("name") in model_columns}
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    rows: list[dict[str, Any]] = []
    for group_id, allowed_groups in group_sets.items():
        columns = sorted(column for column, group in group_by_column.items() if allowed_groups is None or group in allowed_groups)
        if "quality.no_detection" in model_columns and "quality.no_detection" not in columns:
            columns.append("quality.no_detection")
        row: dict[str, Any] = {"feature_group": group_id, "feature_count": len(columns), "model_family": family}
        if not columns:
            row["status"] = "unavailable"
            row["reason"] = "no columns in schema"
            rows.append(row)
            continue
        try:
            scale = family in {"logistic_regression", "svm"}
            candidate = fit_candidate(family, hyperparameters, scale, columns, train, validation, train["label"].astype(int).to_numpy(), validation["label"].astype(int).to_numpy())
            candidate = refit_selected(candidate, pd.concat([train, validation], ignore_index=True))
            test_scores = predict_scores(candidate, test)
            test_metrics = evaluate_scores(test["label"].astype(int).to_numpy(), test_scores, candidate.threshold)
            row.update({"status": "completed", "threshold": candidate.threshold, "validation_f1": candidate.validation_metrics["f1"], **{f"test_{key}": value for key, value in test_metrics.items() if key not in {"confusion_matrix", "positive_class", "threshold", "threshold_convention"}}})
            row.update({f"test_{key}": value for key, value in test_metrics["confusion_matrix"].items()})
        except Exception as exc:  # pragma: no cover - depends on data sufficiency
            row.update({"status": "failed", "reason": str(exc)})
        rows.append(row)
    return rows
