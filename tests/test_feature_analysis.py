from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.feature_analysis import (
    FeatureAnalysisReport,
    compute_feature_analysis,
    write_feature_analysis,
)
from src.inference.explanation import explain_prediction
from src.models.train import fit_candidate, predict_scores, refit_selected
from src.reproducibility import source_revision


@pytest.fixture
def fitted_linear_case():
    rng = np.random.default_rng(17)
    rows: list[dict[str, object]] = []
    sample_id = 0
    for split, count in (("train", 40), ("validation", 12), ("test", 20)):
        labels = np.tile([0, 1], count // 2)
        rng.shuffle(labels)
        for label in labels:
            signal = float(label * 2.0 + rng.normal(0, 0.5))
            texture = float(signal + rng.normal(0, 0.05))
            rows.append({
                "sample_id": f"fixture-{sample_id:03d}",
                "label": int(label),
                "split": split,
                "geometry.signal": signal,
                "texture.correlated": texture,
                "quality.no_detection": 0.0,
            })
            sample_id += 1
    frame = pd.DataFrame(rows)
    columns = ["geometry.signal", "texture.correlated", "quality.no_detection"]
    schema = {
        "schema_version": "features-1",
        "model_columns": columns,
        "columns": [
            {"name": columns[0], "group": "geometry_density", "model_input": True},
            {"name": columns[1], "group": "texture", "model_input": True},
            {"name": columns[2], "group": "quality", "model_input": True},
        ],
    }
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    candidate = fit_candidate(
        "logistic_regression",
        {"C": 1.0, "max_iter": 500, "solver": "liblinear", "random_state": 42},
        True,
        columns,
        train,
        validation,
        train["label"].to_numpy(dtype=int),
        validation["label"].to_numpy(dtype=int),
    )
    candidate = refit_selected(candidate, pd.concat([train, validation], ignore_index=True))
    ablation = [
        {"feature_group": "geometry_density", "feature_count": 2, "model_family": "logistic_regression", "status": "completed", "validation_f1": 0.8, "test_f1": 0.8, "test_accuracy": 0.8, "test_sample_count": 20, "test_tn": 8, "test_fp": 2, "test_fn": 2, "test_tp": 8},
        {"feature_group": "all_enabled", "feature_count": 3, "model_family": "logistic_regression", "status": "completed", "validation_f1": 0.9, "test_f1": 0.9, "test_accuracy": 0.9, "test_sample_count": 20, "test_tn": 9, "test_fp": 1, "test_fn": 1, "test_tp": 9},
    ]
    return frame, schema, candidate, ablation


def _analysis(frame, schema, candidate, ablation):
    return compute_feature_analysis(
        frame,
        schema,
        candidate,
        ablation,
        {
            "status": "software_test",
            "permutation_repeats": 3,
            "random_seed": 42,
            "top_global_features": 2,
            "top_local_features": 2,
            "correlation_method": "spearman",
            "correlation_threshold": 0.95,
        },
        run_id="fixture-evaluation",
        model_id="fixture-model",
    )


def test_feature_analysis_is_deterministic_and_reports_registered_features(fitted_linear_case, tmp_path):
    frame, schema, candidate, ablation = fitted_linear_case
    first = _analysis(frame, schema, candidate, ablation)
    second = _analysis(frame, schema, candidate, ablation)

    assert first.to_dict() == second.to_dict()
    assert {item.feature for item in first.feature_influence} == set(candidate.feature_columns)
    assert first.test_class_counts == {"non_ai_generated": 10, "ai_generated": 10}
    assert first.group_summary[0]["delta_test_f1_vs_all"] == pytest.approx(-0.1)
    assert first.redundancy
    assert {item["source_split"] for item in first.redundancy} == {"train_and_validation"}
    assert all(item.ai_count == 10 and item.non_ai_count == 10 for item in first.feature_influence)

    artifacts = write_feature_analysis(tmp_path / "reports", first)
    assert set(artifacts) == {
        "feature_analysis.json",
        "feature_influence.csv",
        "feature_group_summary.csv",
        "feature_redundancy.csv",
    }
    loaded = FeatureAnalysisReport.from_dict(__import__("json").loads(artifacts["feature_analysis.json"].read_text(encoding="utf-8")))
    assert loaded == first


def test_logistic_explanation_reconstructs_margin_probability_and_threshold(fitted_linear_case):
    frame, schema, candidate, _ = fitted_linear_case
    test_row = frame.loc[frame["split"] == "test"].iloc[0]
    values = {column: test_row[column] for column in candidate.feature_columns}
    row = pd.DataFrame([values], columns=candidate.feature_columns)
    score = float(predict_scores(candidate, row)[0])
    label = "ai_generated" if score >= candidate.threshold else "non_ai_generated"

    explanation = explain_prediction(
        candidate,
        values,
        schema["columns"],
        score,
        label,
        top_n=2,
    )

    assert explanation.status == "available"
    assert explanation.decision_scale == "log_odds"
    assert explanation.score_reconstructed == pytest.approx(score, abs=1e-8)
    assert explanation.baseline_to_threshold + explanation.feature_contribution_sum == pytest.approx(explanation.margin_over_threshold, abs=1e-8)
    assert len(explanation.top_toward_ai) <= 2
    assert len(explanation.top_toward_non_ai) <= 2


def test_prediction_explanation_records_traceability(fitted_linear_case):
    frame, schema, candidate, _ = fitted_linear_case
    row = frame.loc[frame["split"] == "test"].iloc[0]
    values = {column: row[column] for column in candidate.feature_columns}
    score = float(predict_scores(candidate, pd.DataFrame([values]))[0])
    label = "ai_generated" if score >= candidate.threshold else "non_ai_generated"
    identifiers = {
        "input_sha256": "a" * 64,
        "model_id": "fixture-model",
        "run_id": "fixture-run",
        "feature_schema_sha256": "b" * 64,
    }

    explanation = explain_prediction(
        candidate,
        values,
        schema["columns"],
        score,
        label,
        traceability=identifiers,
    )

    assert explanation.schema_version == "prediction-explanation-2"
    assert explanation.to_dict()["traceability"] == identifiers


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("svm", {"kernel": "linear", "C": 1.0, "gamma": "scale"}),
        ("svm", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("random_forest", {"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 1, "random_state": 42, "n_jobs": 1}),
    ],
)
def test_linear_svm_is_additive_and_unsupported_models_are_labeled_global_only(fitted_linear_case, family, params):
    frame, schema, _, _ = fitted_linear_case
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    candidate = fit_candidate(
        family,
        params,
        family == "svm",
        list(schema["model_columns"]),
        train,
        validation,
        train["label"].to_numpy(dtype=int),
        validation["label"].to_numpy(dtype=int),
    )
    candidate = refit_selected(candidate, pd.concat([train, validation], ignore_index=True))
    test_row = frame.loc[frame["split"] == "test"].iloc[0]
    values = {column: test_row[column] for column in candidate.feature_columns}
    score = float(predict_scores(candidate, pd.DataFrame([values]))[0])
    label = "ai_generated" if score >= candidate.threshold else "non_ai_generated"

    explanation = explain_prediction(candidate, values, schema["columns"], score, label, top_n=2)

    if family == "svm" and params["kernel"] == "linear":
        assert explanation.status == "available"
        assert explanation.decision_scale == "decision_margin"
        assert explanation.score_reconstructed == pytest.approx(score, abs=1e-8)
    else:
        assert explanation.status == "global_only"
        assert explanation.feature_values
        assert explanation.top_toward_ai == []


def test_feature_analysis_rejects_feature_schema_drift(fitted_linear_case):
    frame, schema, candidate, ablation = fitted_linear_case
    schema["model_columns"] = list(reversed(schema["model_columns"]))

    with pytest.raises(ValueError, match="locked feature schema"):
        _analysis(frame, schema, candidate, ablation)


def test_source_revision_fallback_prunes_user_data_directories(tmp_path, monkeypatch):
    import os
    import subprocess
    from pathlib import Path

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "data" / "nested").mkdir(parents=True)
    (tmp_path / "data" / "nested" / "private.csv").write_text("sensitive,values\n", encoding="utf-8")
    (tmp_path / "Dataset" / "images").mkdir(parents=True)
    (tmp_path / "Dataset" / "images" / "private.jpg").write_bytes(b"not opened")
    real_walk = os.walk
    visited: list[Path] = []

    def tracked_walk(*args, **kwargs):
        for current, directories, filenames in real_walk(*args, **kwargs):
            visited.append(Path(current))
            yield current, directories, filenames

    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr("src.reproducibility.os.walk", tracked_walk)
    monkeypatch.setattr("src.reproducibility.subprocess.run", no_git)

    revision = source_revision(tmp_path)

    assert revision.startswith("tree:")
    assert all(path.name.casefold() not in {"data", "dataset", "images", "nested"} for path in visited)
