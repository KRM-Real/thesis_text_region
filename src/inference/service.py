"""Single locked inference path used by command line and UI clients."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

from ..contracts import PredictionRecord
from ..features.pipeline import FeatureExtractionResult, extract_features
from .explanation import PredictionExplanation, explain_prediction
from ..models.package import load_model_package
from ..models.train import FittedCandidate, predict_scores
from ..reproducibility import sha256_bytes, sha256_json
from ..text_regions.adapters import EasyOCRDetectorAdapter, FixtureTextDetector
from ..text_regions.protocols import TextRegionDetector
from ..text_regions.visualization import render_overlay


@dataclass
class InferenceResult:
    prediction: PredictionRecord
    extraction: FeatureExtractionResult
    overlay: Image.Image
    stages: list[dict[str, str]] = field(default_factory=list)
    explanation: PredictionExplanation | None = None


def predict_image_bytes(
    image_bytes: bytes,
    model_package: str | Path,
    config: dict[str, Any],
    *,
    category: str | None = None,
    fixture_detector: bool = False,
    detector: TextRegionDetector | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> InferenceResult:
    def stage(name: str, status: str = "completed") -> None:
        if progress:
            progress(name, status)

    input_sha = sha256_bytes(image_bytes)
    stage("image decoded", "running")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
    except Exception as exc:
        raise ValueError(f"image could not be decoded: {exc}") from exc
    stage("image decoded")
    package = load_model_package(model_package)
    manifest = package["manifest"]
    detector_hash = sha256_json(config.get("detector", {}))
    feature_hash = sha256_json(config.get("features", {}))
    if manifest.get("detector_config_sha256") and manifest["detector_config_sha256"] != detector_hash:
        raise ValueError("model package detector configuration does not match the active prototype configuration")
    if manifest.get("feature_config_sha256") and manifest["feature_config_sha256"] != feature_hash:
        raise ValueError("model package feature configuration does not match the active prototype configuration")
    if detector is not None and fixture_detector:
        raise ValueError("Choose either an injected detector or the fixture detector, not both")
    detector = detector or (FixtureTextDetector() if fixture_detector else EasyOCRDetectorAdapter(config))
    stage("text detector ready")
    stage("text regions localized", "running")
    extraction = extract_features(image, "inference", detector, config)
    stage("text regions localized")
    stage("crops and masks prepared")
    stage("visual features extracted")
    stage("image-level record built")
    columns = list(manifest["feature_columns"])
    missing_columns = [column for column in columns if column not in extraction.image_feature_record.values]
    if missing_columns:
        raise ValueError(f"model package schema is incompatible with current pipeline; missing columns: {missing_columns[:5]}")
    row = pd.DataFrame([{column: extraction.image_feature_record.values[column] for column in columns}])
    candidate = FittedCandidate(manifest["estimator_family"], manifest["hyperparameters"], columns, package["preprocessor"], package["estimator"], float(manifest["decision_threshold"]), manifest["threshold_method"], {}, __import__("numpy").asarray([]))
    score = float(predict_scores(candidate, row)[0])
    positive = score >= candidate.threshold
    warnings = ["Prototype result; not proof of authenticity."]
    if category != "receipt":
        warnings.append("Input category is unknown or not confirmed as receipt; interpretation is outside the thesis-valid category.")
    warnings.extend(extraction.detection.warnings)
    if extraction.image_feature_record.no_detection:
        warnings.append("No detected text-like regions; region-derived values were imputed by the trained pipeline.")
    if manifest.get("feature_schema_version") != extraction.image_feature_record.feature_schema_version:
        raise ValueError("model package feature schema version does not match the current pipeline")
    vector_hash = sha256_json(extraction.image_feature_record.values)
    prediction = PredictionRecord(
        schema_version="prediction-1",
        input_sha256=input_sha,
        model_id=manifest["model_id"],
        label="ai_generated" if positive else "non_ai_generated",
        display_label="AI-generated" if positive else "Non-AI-generated under the dataset rules",
        score=score,
        score_type="probability" if manifest["estimator_family"] != "svm" else "decision_score",
        threshold=candidate.threshold,
        region_count=extraction.image_feature_record.region_count,
        no_detection=extraction.image_feature_record.no_detection,
        feature_schema_version=extraction.image_feature_record.feature_schema_version,
        aggregation_version=extraction.image_feature_record.aggregation_version,
        detector_config_sha256=extraction.detection.detector_config_sha256,
        feature_vector_sha256=vector_hash,
        run_id=str(manifest.get("run_id", manifest["model_id"])),
        warnings=warnings,
        quality_flags=extraction.image_feature_record.quality_flags,
        feature_schema_sha256=str(manifest.get("feature_schema_sha256", "")),
    )
    feature_schema = [entry.to_dict() for entry in extraction.feature_schema]
    explanation = explain_prediction(
        candidate,
        extraction.image_feature_record.values,
        feature_schema,
        prediction.score,
        prediction.label,
        top_n=int(config.get("evaluation", {}).get("feature_analysis", {}).get("top_local_features", 5)),
        traceability={
            "input_sha256": prediction.input_sha256,
            "model_id": prediction.model_id,
            "run_id": prediction.run_id,
            "feature_schema_version": prediction.feature_schema_version,
            "feature_schema_sha256": prediction.feature_schema_sha256,
            "aggregation_version": prediction.aggregation_version,
            "detector_config_sha256": prediction.detector_config_sha256,
            "feature_vector_sha256": prediction.feature_vector_sha256,
        },
    )
    stage("model prediction returned")
    return InferenceResult(prediction, extraction, render_overlay(image, extraction), [], explanation)


def predict_image_path(image_path: str | Path, model_package: str | Path, config: dict[str, Any], **kwargs: Any) -> InferenceResult:
    path = Path(image_path)
    return predict_image_bytes(path.read_bytes(), model_package, config, **kwargs)
