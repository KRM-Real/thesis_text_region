"""Leakage-safe traditional classifier training and validation selection."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FittedCandidate:
    family: str
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    preprocessor: Any
    estimator: Any
    threshold: float
    threshold_method: str
    validation_metrics: dict[str, float]
    validation_scores: np.ndarray


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _make_preprocessor(scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _candidates(config: dict[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    models = config.get("models", {})
    result: list[tuple[str, dict[str, Any], bool]] = []
    lr = models.get("logistic_regression", {})
    for c in lr.get("C", [0.1, 1.0, 10.0]):
        result.append(("logistic_regression", {"C": float(c), "max_iter": int(lr.get("max_iter", 2000)), "solver": "liblinear", "random_state": int(models.get("random_seed", 42))}, True))
    svm = models.get("svm", {})
    for kernel in svm.get("kernels", ["linear", "rbf"]):
        for c in svm.get("C", [0.1, 1.0, 10.0]):
            gammas = svm.get("gamma", ["scale"]) if kernel == "rbf" else ["scale"]
            for gamma in gammas:
                result.append(("svm", {"kernel": kernel, "C": float(c), "gamma": gamma}, True))
    forest = models.get("random_forest", {})
    for trees, depth, leaf in itertools.product(forest.get("n_estimators", [200]), forest.get("max_depth", [None]), forest.get("min_samples_leaf", [1])):
        result.append(("random_forest", {"n_estimators": int(trees), "max_depth": depth, "min_samples_leaf": int(leaf), "random_state": int(models.get("random_seed", 42)), "n_jobs": 1}, False))
    return result


def _estimator(family: str, params: dict[str, Any]) -> Any:
    if family == "logistic_regression":
        return LogisticRegression(**params)
    if family == "svm":
        return SVC(**params)
    if family == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"unsupported model family: {family}")


def _scores(estimator: Any, transformed: Any, family: str) -> np.ndarray:
    if family in {"logistic_regression", "random_forest"}:
        return np.asarray(estimator.predict_proba(transformed)[:, 1], dtype=float)
    return np.asarray(estimator.decision_function(transformed), dtype=float)


def _select_threshold(y_true: np.ndarray, scores: np.ndarray, family: str) -> tuple[float, str]:
    default = 0.5 if family in {"logistic_regression", "random_forest"} else 0.0
    candidates = sorted({float(default), *[float(value) for value in scores if np.isfinite(value)]})
    scored = [(_metrics(y_true, scores, threshold)["f1"], _metrics(y_true, scores, threshold)["precision"], -abs(threshold - default), threshold) for threshold in candidates]
    best = max(scored)
    if best[0] <= 0 and default in candidates:
        return default, "estimator_default_no_validation_signal"
    return float(best[3]), "validation_f1"


def fit_candidate(family: str, params: dict[str, Any], scale: bool, columns: list[str], train: pd.DataFrame, validation: pd.DataFrame, y_train: np.ndarray, y_validation: np.ndarray) -> FittedCandidate:
    preprocessor = _make_preprocessor(scale)
    train_values = preprocessor.fit_transform(train[columns])
    validation_values = preprocessor.transform(validation[columns])
    estimator = _estimator(family, params)
    estimator.fit(train_values, y_train)
    validation_scores = _scores(estimator, validation_values, family)
    threshold, threshold_method = _select_threshold(y_validation, validation_scores, family)
    return FittedCandidate(family, params, columns, preprocessor, estimator, threshold, threshold_method, _metrics(y_validation, validation_scores, threshold), validation_scores)


def train_and_select(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> FittedCandidate:
    if train.empty or validation.empty:
        raise ValueError("training and validation rows are required")
    y_train = train["label"].astype(int).to_numpy()
    y_validation = validation["label"].astype(int).to_numpy()
    candidates: list[FittedCandidate] = []
    errors: list[str] = []
    for family, params, scale in _candidates(config):
        try:
            candidates.append(fit_candidate(family, params, scale, feature_columns, train, validation, y_train, y_validation))
        except Exception as exc:
            errors.append(f"{family}:{params}:{exc}")
    if not candidates:
        raise RuntimeError("No candidate model could be trained: " + " | ".join(errors))
    candidates.sort(key=lambda item: (-item.validation_metrics["f1"], -item.validation_metrics["precision"], -item.validation_metrics["accuracy"], item.family, str(item.hyperparameters)))
    selected = candidates[0]
    comparison = [
        {
            "family": candidate.family,
            "hyperparameters": candidate.hyperparameters,
            "validation_metrics": candidate.validation_metrics,
            "threshold": candidate.threshold,
            "threshold_method": candidate.threshold_method,
        }
        for candidate in candidates
    ]
    selected.validation_metrics = {**selected.validation_metrics, "candidate_count": len(candidates), "candidate_errors": errors, "candidate_comparison": comparison}
    return selected


def refit_selected(selected: FittedCandidate, development: pd.DataFrame) -> FittedCandidate:
    y = development["label"].astype(int).to_numpy()
    preprocessor = _make_preprocessor(selected.family in {"logistic_regression", "svm"})
    transformed = preprocessor.fit_transform(development[selected.feature_columns])
    estimator = _estimator(selected.family, selected.hyperparameters)
    estimator.fit(transformed, y)
    selected.preprocessor = preprocessor
    selected.estimator = estimator
    return selected


def predict_scores(package: FittedCandidate, rows: pd.DataFrame) -> np.ndarray:
    transformed = package.preprocessor.transform(rows[package.feature_columns])
    return _scores(package.estimator, transformed, package.family)
