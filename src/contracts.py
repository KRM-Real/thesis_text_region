"""Stable, semantic-free records shared by the prototype pipeline.

The contracts deliberately contain geometry and pixel-derived values only.  OCR
strings are not represented anywhere in this module, which gives the rest of the
application a small, auditable boundary against accidental semantic leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _json_value(value: Any) -> Any:
    """Convert nested dataclass values into JSON-friendly primitives."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    relative_path: str
    label: str | None
    category: str
    source_id: str
    group_id: str
    provenance_type: str
    generator_id: str | None
    acquisition_type: str
    width: int
    height: int
    format: str
    sha256: str
    split: str | None = None
    inclusion_status: str = "included"
    exclusion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class TextRegion:
    region_id: str
    bbox_px: tuple[int, int, int, int]
    bbox_norm: tuple[float, float, float, float]
    polygon_px: list[tuple[float, float]] | None = None
    polygon_norm: list[tuple[float, float]] | None = None
    confidence: float | None = None
    quality_flags: list[str] = field(default_factory=list)
    crop_bbox_px: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class DetectionResult:
    schema_version: str
    sample_id: str
    image_width: int
    image_height: int
    detector_name: str
    detector_version: str
    detector_config_sha256: str
    regions: list[TextRegion] = field(default_factory=list)
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class RegionFeatureResult:
    name: str
    group: str
    value: float | None
    status: str = "ok"
    formula_id: str = ""
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class ImageFeatureRecord:
    sample_id: str
    label: str | None
    values: dict[str, float | int | None]
    region_count: int
    no_detection: bool
    quality_flags: list[str] = field(default_factory=list)
    feature_schema_version: str = "features-1"
    aggregation_version: str = "aggregation-1"
    source_localization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class FeatureSchemaEntry:
    name: str
    group: str
    level: str
    formula_id: str
    schema_version: str
    source_fields: tuple[str, ...]
    dtype: str = "float64"
    expected_range: tuple[float | None, float | None] = (None, None)
    missing_policy: str = "nan_then_training_imputation"
    model_input: bool = True
    diagnostic_only: bool = False
    availability: str = "available"

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class PredictionRecord:
    schema_version: str
    input_sha256: str
    model_id: str
    label: str
    display_label: str
    score: float
    score_type: str
    threshold: float
    region_count: int
    no_detection: bool
    feature_schema_version: str
    aggregation_version: str
    detector_config_sha256: str
    feature_vector_sha256: str
    run_id: str
    warnings: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    feature_schema_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class ModelPackageManifest:
    model_id: str
    created_at_utc: str
    label_mapping: dict[str, int]
    estimator_family: str
    hyperparameters: dict[str, Any]
    feature_schema_version: str
    feature_columns: list[str]
    preprocessor_version: str
    detector_config_sha256: str
    feature_config_sha256: str
    split_manifest_sha256: str
    training_sample_count: int
    validation_sample_count: int
    decision_threshold: float
    threshold_method: str
    runtime_versions: dict[str, str]
    study_warning: str
    run_id: str | None = None
    feature_schema_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class RunManifest:
    run_id: str
    source_revision: str
    configuration_path: str
    configuration_sha256: str
    audited_manifest_path: str | None
    audited_manifest_sha256: str | None
    split_manifest_path: str | None
    split_manifest_sha256: str | None
    detector: dict[str, Any]
    crop_preprocessing: dict[str, Any]
    feature_schema_version: str
    feature_schema_sha256: str
    aggregation_version: str
    model_settings: dict[str, Any]
    selected_model_id: str | None
    threshold: float | None
    threshold_method: str | None
    random_seeds: dict[str, int]
    sample_counts: dict[str, int]
    software_versions: dict[str, str]
    hardware_context: dict[str, str]
    started_at_utc: str
    completed_at_utc: str | None
    status: str
    failure_reason: str | None = None
    feature_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))
