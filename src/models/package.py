"""Versioned model package serialization and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from ..contracts import ModelPackageManifest
from ..reproducibility import sha256_file, write_json
from .train import FittedCandidate


def save_model_package(path: str | Path, candidate: FittedCandidate, manifest: ModelPackageManifest, metadata: dict[str, Any] | None = None) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate.estimator, target / "estimator.joblib")
    joblib.dump(candidate.preprocessor, target / "preprocessor.joblib")
    write_json(target / "package_manifest.json", manifest.to_dict())
    write_json(target / "feature_schema.json", {"schema_version": manifest.feature_schema_version, "columns": manifest.feature_columns})
    if metadata:
        if "feature_schema" in metadata:
            write_json(target / "feature_schema.json", metadata["feature_schema"])
        if "detector_config" in metadata:
            write_json(target / "detector_config.json", metadata["detector_config"])
        if "feature_config" in metadata:
            write_json(target / "feature_config.json", metadata["feature_config"])
        if "validation_metrics" in metadata:
            write_json(target / "validation_metrics.json", metadata["validation_metrics"])
        if "training_summary" in metadata:
            write_json(target / "training_summary.json", metadata["training_summary"])
        if "split_manifest_sha256" in metadata:
            (target / "split_manifest_sha256.txt").write_text(str(metadata["split_manifest_sha256"]) + "\n", encoding="utf-8")
    hashes = {
        str(path.relative_to(target)).replace("\\", "/"): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "package_hashes.json"
    }
    write_json(target / "package_hashes.json", hashes)
    return target


def load_model_package(path: str | Path, expected_feature_columns: list[str] | None = None) -> dict[str, Any]:
    target = Path(path)
    manifest_path = target / "package_manifest.json"
    if not manifest_path.is_file() or not (target / "estimator.joblib").is_file() or not (target / "preprocessor.joblib").is_file():
        raise ValueError(f"incomplete model package: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes_path = target / "package_hashes.json"
    if hashes_path.is_file():
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        for relative, expected in hashes.items():
            artifact = target / relative
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"model package artifact hash mismatch: {relative}")
    columns = list(manifest.get("feature_columns", []))
    if manifest.get("estimator_family") not in {"logistic_regression", "svm", "random_forest"}:
        raise ValueError("model package contains an unsupported estimator family")
    if not columns or not manifest.get("feature_schema_version"):
        raise ValueError("model package manifest is missing feature schema metadata")
    if expected_feature_columns is not None and columns != expected_feature_columns:
        raise ValueError("model package feature schema does not match the current pipeline")
    return {"path": target, "manifest": manifest, "estimator": joblib.load(target / "estimator.joblib"), "preprocessor": joblib.load(target / "preprocessor.joblib")}
