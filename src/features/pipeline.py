"""Shared localization-to-feature pipeline used by batch, CLI, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
from PIL import Image

from ..contracts import DetectionResult, FeatureSchemaEntry, ImageFeatureRecord, RegionFeatureResult
from ..text_regions.pipeline import LocalizationArtifacts, localize_image
from .visual import aggregate_region_features, compute_region_features, image_geometry_features


@dataclass
class FeatureExtractionResult:
    sample_id: str
    label: str | None
    detection: DetectionResult
    artifacts: LocalizationArtifacts
    region_features: dict[str, dict[str, RegionFeatureResult]] = field(default_factory=dict)
    image_feature_record: ImageFeatureRecord | None = None
    feature_schema: list[FeatureSchemaEntry] = field(default_factory=list)


def extract_features(image: Image.Image, sample_id: str, detector: Any, config: dict[str, Any], label: str | None = None) -> FeatureExtractionResult:
    detection, artifacts = localize_image(image, sample_id, detector, config)
    feature_config = config.get("features", {})
    region_features: dict[str, dict[str, RegionFeatureResult]] = {}
    for region in detection.regions:
        padded = artifacts.padded_crops.get(region.region_id)
        if padded is None or padded.size == 0:
            region_features[region.region_id] = {}
            continue
        unpadded = artifacts.unpadded_crops.get(region.region_id)
        gray = cv2.cvtColor(unpadded if unpadded is not None and unpadded.size else padded, cv2.COLOR_RGB2GRAY)
        region_features[region.region_id] = compute_region_features(
            gray,
            region,
            (detection.image_height, detection.image_width),
            padded_rgb=padded,
            config=feature_config,
        )
    image_geometry = image_geometry_features(
        detection.regions,
        (detection.image_height, detection.image_width),
        float(feature_config.get("row_tolerance_ratio", 0.02)),
        list(feature_config.get("enabled_groups", [])),
    )
    values, schema, quality_flags = aggregate_region_features(
        list(region_features.values()),
        image_geometry,
        sample_id,
        label,
        str(feature_config.get("schema_version", "features-1")),
        list(feature_config.get("enabled_groups", [])),
        list(config.get("aggregation", {}).get("statistics", [])),
    )
    if detection.status == "no_detection":
        quality_flags.append("no_detection")
    if detection.status == "detector_unavailable":
        quality_flags.append("detector_unavailable")
    quality_flags.extend(
        f"{region_id}:{name}:{result.reason}"
        for region_id, features in region_features.items()
        for name, result in features.items()
        if result.status != "ok" and result.reason
    )
    record = ImageFeatureRecord(
        sample_id=sample_id,
        label=label,
        values=values,
        region_count=len(detection.regions),
        no_detection=not bool(detection.regions),
        quality_flags=sorted(set(quality_flags)),
        feature_schema_version=str(feature_config.get("schema_version", "features-1")),
        aggregation_version=str(config.get("aggregation", {}).get("schema_version", "aggregation-1")),
    )
    return FeatureExtractionResult(sample_id, label, detection, artifacts, region_features, record, schema)
