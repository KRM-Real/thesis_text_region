"""Shared data records for the exploratory pilot.

The records deliberately contain geometry and visual measurements only. OCR
recognized words are never represented in the model-facing data structures.
Configuration records live here as well so the Streamlit app, command-line
batch runner, and tests use exactly the same validation rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """A clipped, integer pixel rectangle in ``x, y, width, height`` form."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class FeatureResult:
    """A feature value or an explicit unavailable result."""

    name: str
    group: str
    status: str = "ok"
    value: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "status": self.status,
            "value": self.value,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DetectionConfig:
    """EasyOCR CRAFT localization settings used for a pilot run.

    These are detector settings, not a learned REAL/AI classifier. EasyOCR's
    localization API does not expose a confidence value for every returned
    box, so its score thresholds are retained as diagnostics and no synthetic
    per-region confidence is created.
    """

    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    bbox_min_score: float = 0.2
    min_size: int = 20
    canvas_size: int = 2560
    mag_ratio: float = 1.0
    languages: tuple[str, ...] = ("en",)
    gpu: bool = False

    def __post_init__(self) -> None:
        for name in ("text_threshold", "low_text", "link_threshold", "bbox_min_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if self.canvas_size <= 0:
            raise ValueError("canvas_size must be positive")
        if not math.isfinite(float(self.mag_ratio)) or self.mag_ratio <= 0:
            raise ValueError("mag_ratio must be a positive finite number")
        if not self.languages or any(not str(language).strip() for language in self.languages):
            raise ValueError("at least one detector language is required")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Detection-only preprocessing; feature crops remain native-resolution."""

    mode: str = "none"
    scale: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        if self.mode not in {"none", "clahe"}:
            raise ValueError("preprocessing mode must be 'none' or 'clahe'")
        if not math.isfinite(float(self.scale)) or self.scale not in {1.0, 1.5, 2.0}:
            raise ValueError("preprocessing scale must be one of 1.0, 1.5, or 2.0")
        if not math.isfinite(float(self.clahe_clip_limit)) or self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if len(self.clahe_grid_size) != 2 or any(int(value) <= 0 for value in self.clahe_grid_size):
            raise ValueError("clahe_grid_size must contain two positive integers")


@dataclass(frozen=True)
class AggregationConfig:
    """Image-level aggregation settings."""

    statistics: tuple[str, ...] = ("mean", "std")

    def __post_init__(self) -> None:
        allowed = {"mean", "std", "min", "max", "median"}
        if not self.statistics:
            raise ValueError("at least one aggregation statistic is required")
        unknown = set(self.statistics) - allowed
        if unknown:
            raise ValueError(f"Unsupported aggregation statistic(s): {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class BatchConfig:
    """Dataset-level expectations and artifact policy for the local runner."""

    expected_real_count: int = 300
    expected_ai_generated_count: int = 300
    save_artifacts: str = "all"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.expected_real_count < 0 or self.expected_ai_generated_count < 0:
            raise ValueError("expected class counts must be non-negative")
        if self.save_artifacts not in {"all", "failures", "none"}:
            raise ValueError("save_artifacts must be all, failures, or none")


@dataclass
class TextRegion:
    """Detector output plus crop and region-level visual measurements."""

    region_id: str
    bbox: BoundingBox
    polygon: list[tuple[int, int]] = field(default_factory=list)
    confidence: float | None = None
    crop_bbox: BoundingBox | None = None
    features: dict[str, FeatureResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, include_features: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "region_id": self.region_id,
            "bbox": self.bbox.to_dict(),
            "polygon": [[x, y] for x, y in self.polygon],
            "confidence": self.confidence,
            "crop_bbox": self.crop_bbox.to_dict() if self.crop_bbox else None,
            "warnings": list(self.warnings),
        }
        if include_features:
            data["features"] = {name: result.to_dict() for name, result in self.features.items()}
        return data


@dataclass
class FeatureConfig:
    """Settings for the initial experimental feature set."""

    edge_low_threshold: int = 100
    edge_high_threshold: int = 200
    minimum_region_size: int = 3
    density_enabled: bool = True
    enabled_features: tuple[str, ...] = (
        "sharpness_laplacian_variance",
        "edge_density",
        "intensity_std",
    )

    def __post_init__(self) -> None:
        if self.edge_low_threshold < 0 or self.edge_high_threshold > 255:
            raise ValueError("edge thresholds must be in the uint8 range")
        if self.edge_high_threshold <= self.edge_low_threshold:
            raise ValueError("edge_high_threshold must be greater than edge_low_threshold")
        if self.minimum_region_size < 1:
            raise ValueError("minimum_region_size must be positive")


@dataclass
class PilotConfig:
    """Reproducible settings recorded with each analysis."""

    detector_name: str = "EasyOCR CRAFT (detection-only)"
    detector_status: str = "PILOT_DECISION"
    crop_padding: int = 0
    aggregation: tuple[str, ...] = ("mean", "std")
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    image_mode: str = "RGB"
    label_values: tuple[str, ...] = ("REAL", "AI_GENERATED")

    def __post_init__(self) -> None:
        AggregationConfig(tuple(self.aggregation))
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": {
                "name": self.detector_name,
                "mode": "detection-only",
                "status": self.detector_status,
                "recognized_text_used": False,
                **asdict(self.detection),
            },
            "crop_padding": self.crop_padding,
            "aggregation": list(self.aggregation),
            "feature_config": asdict(self.feature_config),
            "preprocessing": asdict(self.preprocessing),
            "batch": asdict(self.batch),
            "image_mode": self.image_mode,
            "label_values": list(self.label_values),
        }


@dataclass
class ImageAnalysis:
    """Complete, serializable result for one image."""

    image_id: str
    width: int
    height: int
    channels: int
    file_type: str | None
    label: str | None
    detector: str
    detector_status: str
    detection_status: str
    regions: list[TextRegion] = field(default_factory=list)
    image_features: dict[str, FeatureResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def model_feature_names(self) -> list[str]:
        """Return only visual feature columns intended for later ML use."""

        configured = self.settings.get("model_feature_names")
        if configured is not None:
            return list(configured)
        return [
            name
            for name, result in self.image_features.items()
            if result.metadata.get("role") != "diagnostic"
        ]

    def model_feature_record(self) -> dict[str, Any]:
        """Return a fixed-length visual record without metadata or OCR content."""

        return {
            name: (
                self.image_features[name].value
                if name in self.image_features and self.image_features[name].status == "ok"
                else None
            )
            for name in self.model_feature_names
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "image_id": self.image_id,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "file_type": self.file_type,
            "label": self.label,
            "detector": self.detector,
            "detector_status": self.detector_status,
            "detection_status": self.detection_status,
            "regions": [region.to_dict() for region in self.regions],
            "image_features": {
                name: result.to_dict() for name, result in self.image_features.items()
            },
            "model_feature_names": self.model_feature_names,
            "model_feature_record": self.model_feature_record(),
            "warnings": list(self.warnings),
            "settings": self.settings,
            "diagnostics": self.diagnostics,
        }
