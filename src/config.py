"""Load and validate the shared pilot configuration.

The YAML file is intentionally small and human-editable. Both the Streamlit
interface and the batch runner call this module so a run cannot accidentally
use a different set of detector, preprocessing, feature, or aggregation
defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.types import (
    AggregationConfig,
    BatchConfig,
    DetectionConfig,
    FeatureConfig,
    PilotConfig,
    PreprocessingConfig,
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_pilot_config(path: str | Path = "configs/pilot.yaml") -> PilotConfig:
    """Read a YAML pilot configuration and return validated typed settings."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency setup issue
        raise RuntimeError("PyYAML is required to load configs/pilot.yaml") from exc

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"Could not read pilot configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Pilot configuration root must be a mapping")

    detector_raw = _mapping(raw.get("detector"), "detector")
    preprocessing_raw = _mapping(raw.get("preprocessing"), "preprocessing")
    features_raw = _mapping(raw.get("features"), "features")
    edge_raw = _mapping(raw.get("edge"), "edge")
    aggregation_raw = _mapping(raw.get("aggregation"), "aggregation")
    batch_raw = _mapping(raw.get("batch"), "batch")

    detection = DetectionConfig(
        text_threshold=float(detector_raw.get("text_threshold", 0.7)),
        low_text=float(detector_raw.get("low_text", 0.4)),
        link_threshold=float(detector_raw.get("link_threshold", 0.4)),
        bbox_min_score=float(detector_raw.get("bbox_min_score", 0.2)),
        min_size=int(detector_raw.get("min_size", 20)),
        canvas_size=int(detector_raw.get("canvas_size", 2560)),
        mag_ratio=float(detector_raw.get("mag_ratio", 1.0)),
        languages=tuple(detector_raw.get("languages", ("en",))),
        gpu=bool(detector_raw.get("gpu", False)),
    )
    grid = preprocessing_raw.get("clahe_grid_size", (8, 8))
    if isinstance(grid, list):
        grid = tuple(grid)
    preprocessing = PreprocessingConfig(
        mode=str(preprocessing_raw.get("mode", "none")),
        scale=float(preprocessing_raw.get("scale", 1.0)),
        clahe_clip_limit=float(preprocessing_raw.get("clahe_clip_limit", 2.0)),
        clahe_grid_size=tuple(grid),
    )
    enabled = tuple(
        name
        for name in (
            "sharpness_laplacian_variance",
            "edge_density",
            "intensity_std",
        )
        if bool(features_raw.get(name, True))
    )
    feature_config = FeatureConfig(
        edge_low_threshold=int(edge_raw.get("low_threshold", 100)),
        edge_high_threshold=int(edge_raw.get("high_threshold", 200)),
        minimum_region_size=int(raw.get("minimum_region_size", 3)),
        density_enabled=bool(features_raw.get("text_region_area_ratio", True)),
        enabled_features=enabled,
    )
    statistics = aggregation_raw.get("statistics", ("mean", "std"))
    if isinstance(statistics, str):
        statistics = (statistics,)
    aggregation = AggregationConfig(tuple(statistics))
    batch = BatchConfig(
        expected_real_count=int(batch_raw.get("expected_real_count", 300)),
        expected_ai_generated_count=int(batch_raw.get("expected_ai_generated_count", 300)),
        save_artifacts=str(batch_raw.get("save_artifacts", "all")),
        random_seed=int(batch_raw.get("random_seed", 42)),
    )
    detector_name = str(detector_raw.get("name", "EasyOCR CRAFT (detection-only)"))
    detector_status = str(detector_raw.get("status", "PILOT_DECISION"))
    labels = raw.get("label_values", ("REAL", "AI_GENERATED"))
    if isinstance(labels, str):
        labels = (labels,)
    return PilotConfig(
        detector_name=detector_name,
        detector_status=detector_status,
        crop_padding=int(raw.get("crop_padding", 0)),
        aggregation=aggregation.statistics,
        feature_config=feature_config,
        detection=detection,
        preprocessing=preprocessing,
        batch=batch,
        image_mode=str(raw.get("image_mode", "RGB")),
        label_values=tuple(labels),
    )


def config_to_yaml(config: PilotConfig) -> str:
    """Serialize resolved settings for a batch artifact without secrets."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to serialize the pilot configuration") from exc
    return yaml.safe_dump(config.to_dict(), sort_keys=False)
