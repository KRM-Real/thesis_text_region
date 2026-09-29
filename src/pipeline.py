"""End-to-end orchestration for one image analysis.

The pipeline keeps detector input preparation, native-resolution feature
crops, region-level values, image-level aggregation, and diagnostics distinct.
That separation makes it possible to inspect preprocessing without allowing a
detector-only transformation or semantic OCR output into the model record.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.aggregation.aggregate import aggregate_region_features
from src.detection.types import RawDetection
from src.features.base import feature_registry
from src.features.contrast import compute_intensity_std
from src.features.density import compute_text_region_area_ratio
from src.features.edges import compute_edge_density, edge_map
from src.features.sharpness import compute_laplacian_variance
from src.preprocessing.image_ops import (
    crop_region,
    image_to_rgb_array,
    prepare_detection_image,
    restore_polygon_coordinates,
    to_grayscale,
)
from src.types import (
    BoundingBox,
    FeatureConfig,
    FeatureResult,
    ImageAnalysis,
    PilotConfig,
    TextRegion,
)
from src.utils.logging import log_analysis_summary


FEATURE_GROUPS = {
    "sharpness_laplacian_variance": "sharpness",
    "edge_density": "edge_rendering",
    "intensity_std": "contrast",
}

# The image record consumed by a later classifier stays deliberately compact.
# Additional statistics remain available in ``ImageAnalysis.image_features``
# for inspection, but are not emitted by the model-facing export.
MODEL_AGGREGATION_STATISTICS = ("mean", "std")


@dataclass
class AnalysisArtifacts:
    """Native and processed arrays kept outside the serializable analysis record."""

    crops: dict[str, np.ndarray] = field(default_factory=dict)
    grayscale: dict[str, np.ndarray] = field(default_factory=dict)
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    detection_input: np.ndarray | None = None

    # Preserve the original pilot API where the second return value was a
    # ``dict[region_id, crop]``.
    def __getitem__(self, region_id: str) -> np.ndarray:
        return self.crops[region_id]

    def get(self, region_id: str, default: Any = None) -> np.ndarray | Any:
        return self.crops.get(region_id, default)

    def items(self):
        return self.crops.items()

    def keys(self):
        return self.crops.keys()

    def __len__(self) -> int:
        return len(self.crops)


def expected_model_feature_names(config: PilotConfig) -> list[str]:
    """Return the compact, fixed visual model-vector order.

    Mean and population standard deviation are the pilot's stable model
    statistics. Optional min/max/median values may still be requested for
    inspection, but are intentionally excluded from this record.
    """

    names: list[str] = []
    for feature_name in config.feature_config.enabled_features:
        if feature_name not in FEATURE_GROUPS:
            continue
        names.extend(f"{feature_name}_{statistic}" for statistic in MODEL_AGGREGATION_STATISTICS)
    if config.feature_config.density_enabled:
        names.append("text_region_area_ratio")
    return names


def inspection_aggregation_statistics(config: PilotConfig) -> tuple[str, ...]:
    """Return model defaults plus any optional statistics requested for inspection."""

    return tuple(dict.fromkeys((*MODEL_AGGREGATION_STATISTICS, *config.aggregation)))


def expected_image_feature_names(config: PilotConfig) -> list[str]:
    """Return every image-level column, including optional inspection statistics."""

    names: list[str] = []
    for feature_name in config.feature_config.enabled_features:
        if feature_name not in FEATURE_GROUPS:
            continue
        names.extend(
            f"{feature_name}_{statistic}"
            for statistic in inspection_aggregation_statistics(config)
        )
    if config.feature_config.density_enabled:
        names.append("text_region_area_ratio")
    return names


def _box_from_polygon(polygon: list[tuple[float, float]]) -> BoundingBox:
    if not polygon:
        return BoundingBox(0, 0, 0, 0)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x = int(np.floor(min(xs)))
    y = int(np.floor(min(ys)))
    x2 = int(np.ceil(max(xs)))
    y2 = int(np.ceil(max(ys)))
    return BoundingBox(x, y, max(0, x2 - x), max(0, y2 - y))


def _normalize_detection(
    raw: RawDetection,
    image_width: int,
    image_height: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    minimum_side: int = 1,
) -> tuple[TextRegion | None, str | None]:
    """Restore, validate, and clip one detector polygon."""

    polygon_value = getattr(raw, "polygon", None)
    if not isinstance(polygon_value, (list, tuple)) or len(polygon_value) < 3:
        return None, "malformed_polygon"
    points: list[tuple[float, float]] = []
    for point in polygon_value:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
            return None, "malformed_point"
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None, "non_numeric_point"
        if not math.isfinite(x) or not math.isfinite(y):
            return None, "non_finite_point"
        points.append((x, y))
    try:
        restored = restore_polygon_coordinates(points, scale_x, scale_y)
    except ValueError:
        return None, "invalid_coordinate_transform"
    box = _box_from_polygon(restored)
    raw_x2, raw_y2 = box.x2, box.y2
    if raw_x2 <= 0 or raw_y2 <= 0 or box.x >= image_width or box.y >= image_height:
        return None, "outside_image"
    x1 = max(0, min(image_width, box.x))
    y1 = max(0, min(image_height, box.y))
    x2 = max(x1, min(image_width, raw_x2))
    y2 = max(y1, min(image_height, raw_y2))
    clipped = BoundingBox(x1, y1, x2 - x1, y2 - y1)
    if clipped.width <= 0 or clipped.height <= 0:
        return None, "empty_after_clipping"
    if clipped.width < minimum_side or clipped.height < minimum_side:
        return None, "too_small"
    clipped_polygon = [
        (
            max(0, min(image_width, int(round(x)))),
            max(0, min(image_height, int(round(y)))),
        )
        for x, y in restored
    ]
    confidence = getattr(raw, "confidence", None)
    try:
        confidence = (
            float(confidence)
            if confidence is not None and math.isfinite(float(confidence))
            else None
        )
    except (TypeError, ValueError):
        confidence = None
    return (
        TextRegion(
            region_id="",
            bbox=clipped,
            polygon=clipped_polygon,
            confidence=confidence,
        ),
        None,
    )


def _call_detector(detector: Any, image_rgb: np.ndarray, config: PilotConfig) -> list[RawDetection]:
    """Call current and backwards-compatible detector adapters."""

    detect = detector.detect
    try:
        parameters = inspect.signature(detect).parameters
    except (TypeError, ValueError):
        parameters = {}
    if len(parameters) >= 2 or "config" in parameters:
        result = detect(image_rgb, config.detection)
    else:
        result = detect(image_rgb)
    if result is None:
        return []
    return list(result)


def _region_feature_values(gray_crop: np.ndarray, config: FeatureConfig) -> dict[str, FeatureResult]:
    """Compute only explicitly enabled initial experimental features."""

    results: dict[str, FeatureResult] = {}
    enabled = set(config.enabled_features)
    if "sharpness_laplacian_variance" in enabled:
        results["sharpness_laplacian_variance"] = compute_laplacian_variance(
            gray_crop, minimum_size=config.minimum_region_size
        )
    if "edge_density" in enabled:
        results["edge_density"] = compute_edge_density(
            gray_crop,
            low_threshold=config.edge_low_threshold,
            high_threshold=config.edge_high_threshold,
            minimum_size=config.minimum_region_size,
        )
    if "intensity_std" in enabled:
        results["intensity_std"] = compute_intensity_std(gray_crop)
    return results


def _unavailable_region_features(config: FeatureConfig, reason: str) -> dict[str, FeatureResult]:
    return {
        name: FeatureResult(name, FEATURE_GROUPS[name], "unavailable", reason=reason)
        for name in config.enabled_features
        if name in FEATURE_GROUPS
    }


def _unavailable_image_features(config: PilotConfig, reason: str) -> dict[str, FeatureResult]:
    result: dict[str, FeatureResult] = {}
    for name in expected_image_feature_names(config):
        if name == "text_region_area_ratio":
            source, group, statistic = name, "density", None
        else:
            source, statistic = name.rsplit("_", 1)
            group = FEATURE_GROUPS.get(source, "experimental")
        result[name] = FeatureResult(
            name,
            group,
            "unavailable",
            reason=reason,
            metadata={
                "source_feature": source,
                "statistic": statistic,
                "valid_region_count": 0,
            },
        )
    return result


def _initialize_analysis(
    image_id: str,
    width: int,
    height: int,
    channels: int,
    file_type: str | None,
    label: str | None,
    detector: Any,
    config: PilotConfig,
) -> ImageAnalysis:
    settings = {
        **config.to_dict(),
        "feature_registry": feature_registry(),
        "model_feature_names": expected_model_feature_names(config),
        "image_feature_names": expected_image_feature_names(config),
    }
    return ImageAnalysis(
        image_id=image_id,
        width=width,
        height=height,
        channels=channels,
        file_type=file_type,
        label=label,
        detector=getattr(detector, "name", config.detector_name),
        detector_status=getattr(detector, "status", config.detector_status),
        detection_status="pending",
        settings=settings,
    )


def make_unavailable_analysis(
    image_id: str,
    config: PilotConfig,
    reason: str,
    label: str | None = None,
    file_type: str | None = None,
    detector: Any | None = None,
) -> ImageAnalysis:
    """Create a fixed-schema failure record for decode or batch errors."""

    analysis = _initialize_analysis(
        image_id,
        width=0,
        height=0,
        channels=0,
        file_type=file_type,
        label=label,
        detector=detector or object(),
        config=config,
    )
    analysis.detection_status = "failed"
    analysis.image_features.update(_unavailable_image_features(config, reason))
    analysis.image_features["detected_region_count"] = FeatureResult(
        "detected_region_count", "density", "unavailable", reason=reason, metadata={"role": "diagnostic"}
    )
    analysis.warnings.append(reason)
    analysis.diagnostics.update(
        {
            "stage": "image_load",
            "raw_detection_count": 0,
            "accepted_detection_count": 0,
            "rejected_detection_count": 0,
            "rejected_detections": [],
            "valid_region_count": 0,
            "failed_features": expected_image_feature_names(config),
        }
    )
    return analysis


def analyze_image(
    image: Image.Image,
    detector: Any,
    config: PilotConfig | None = None,
    image_id: str = "uploaded_image",
    label: str | None = None,
    file_type: str | None = None,
) -> tuple[ImageAnalysis, AnalysisArtifacts]:
    """Run detection, crops, features, aggregation, and diagnostics.

    Features always use native-resolution crops from the original RGB image.
    Detector-only preprocessing and scaling are retained in ``artifacts`` and
    mapped back before region normalization.
    """

    config = config or PilotConfig()
    rgb = image_to_rgb_array(image)
    height, width = rgb.shape[:2]
    analysis = _initialize_analysis(image_id, width, height, 3, file_type, label, detector, config)
    artifacts = AnalysisArtifacts()
    try:
        preparation = prepare_detection_image(rgb, config.preprocessing)
    except Exception as exc:
        analysis.detection_status = "failed"
        reason = f"Detection preprocessing failed: {exc}"
        analysis.image_features.update(_unavailable_image_features(config, reason))
        analysis.image_features["detected_region_count"] = FeatureResult(
            "detected_region_count", "density", "unavailable", reason=reason, metadata={"role": "diagnostic"}
        )
        analysis.warnings.append(reason)
        analysis.diagnostics.update(
            {
                "stage": "detection_preprocessing",
                "raw_detection_count": 0,
                "accepted_detection_count": 0,
                "rejected_detection_count": 0,
                "rejected_detections": [],
                "valid_region_count": 0,
                "failed_features": expected_image_feature_names(config),
            }
        )
        log_analysis_summary(analysis)
        return analysis, artifacts
    artifacts.detection_input = preparation.image_rgb
    analysis.diagnostics["preprocessing"] = preparation.to_dict()
    try:
        raw_detections = _call_detector(detector, preparation.image_rgb, config)
    except Exception as exc:
        analysis.detection_status = "failed"
        reason = f"Text detection failed: {exc}"
        analysis.image_features.update(_unavailable_image_features(config, reason))
        analysis.image_features["detected_region_count"] = FeatureResult(
            "detected_region_count", "density", "unavailable", reason=reason, metadata={"role": "diagnostic"}
        )
        analysis.warnings.append(reason)
        analysis.diagnostics.update(
            {
                "stage": "detection",
                "raw_detection_count": 0,
                "accepted_detection_count": 0,
                "rejected_detection_count": 0,
                "rejected_detections": [],
                "valid_region_count": 0,
                "failed_features": expected_image_feature_names(config),
                "detector_settings": config.to_dict()["detector"],
            }
        )
        log_analysis_summary(analysis)
        return analysis, artifacts

    analysis.detection_status = "ok"
    normalized: list[TextRegion] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_detections, start=1):
        region, reason = _normalize_detection(
            raw,
            width,
            height,
            scale_x=preparation.scale_x,
            scale_y=preparation.scale_y,
            minimum_side=config.feature_config.minimum_region_size,
        )
        if region is None:
            rejected.append({"index": index, "reason": reason or "rejected"})
            continue
        region.region_id = f"region_{len(normalized) + 1:03d}"
        normalized.append(region)
    analysis.regions = normalized
    if rejected:
        analysis.warnings.append(f"{len(rejected)} detector region(s) were rejected by pilot validation filters.")
    analysis.diagnostics.update(
        {
            "stage": "detection",
            "raw_detection_count": len(raw_detections),
            "accepted_detection_count": len(normalized),
            "rejected_detection_count": len(rejected),
            "rejected_detections": rejected,
            "valid_region_count": 0,
            "detector_settings": config.to_dict()["detector"],
        }
    )

    region_feature_maps: list[dict[str, FeatureResult]] = []
    for region in analysis.regions:
        crop, crop_bbox = crop_region(rgb, region.bbox, config.crop_padding)
        region.crop_bbox = crop_bbox
        if crop is None:
            region.warnings.append("Invalid or empty crop after clipping.")
            region.features = _unavailable_region_features(config.feature_config, "Crop is empty.")
            region_feature_maps.append(region.features)
            continue
        artifacts.crops[region.region_id] = crop
        try:
            gray = to_grayscale(crop)
            artifacts.grayscale[region.region_id] = gray
            edge = edge_map(
                gray,
                config.feature_config.edge_low_threshold,
                config.feature_config.edge_high_threshold,
            )
            if edge is not None:
                artifacts.edges[region.region_id] = edge
            region.features = _region_feature_values(gray, config.feature_config)
        except Exception as exc:
            region.features = _unavailable_region_features(
                config.feature_config, f"Feature preprocessing failed: {exc}"
            )
        if any(result.status != "ok" for result in region.features.values()):
            region.warnings.append("One or more experimental features were unavailable for this crop.")
        region_feature_maps.append(region.features)

    aggregation_statistics = inspection_aggregation_statistics(config)
    aggregated = aggregate_region_features(region_feature_maps, aggregation_statistics)
    for feature_name in config.feature_config.enabled_features:
        if feature_name not in FEATURE_GROUPS:
            continue
        for statistic in aggregation_statistics:
            key = f"{feature_name}_{statistic}"
            aggregated.setdefault(
                key,
                FeatureResult(
                    key,
                    FEATURE_GROUPS[feature_name],
                    "unavailable",
                    reason="No valid detected-region measurement was available.",
                    metadata={
                        "source_feature": feature_name,
                        "statistic": statistic,
                        "valid_region_count": 0,
                    },
                ),
            )
    analysis.image_features.update(aggregated)
    analysis.image_features["detected_region_count"] = FeatureResult(
        "detected_region_count",
        "density",
        value=float(len(analysis.regions)),
        metadata={"role": "diagnostic"},
    )
    if config.feature_config.density_enabled:
        analysis.image_features["text_region_area_ratio"] = compute_text_region_area_ratio(
            [region.bbox for region in analysis.regions], width, height
        )
    if not analysis.regions:
        analysis.warnings.append("No visible text regions were detected. Region-based features are unavailable.")
    analysis.diagnostics["valid_region_count"] = len(artifacts.crops)
    failed_features = {
        name
        for region in analysis.regions
        for name, result in region.features.items()
        if result.status != "ok"
    }
    failed_features.update(
        name for name, result in analysis.image_features.items() if result.status != "ok"
    )
    analysis.diagnostics["failed_features"] = sorted(failed_features)
    log_analysis_summary(analysis)
    return analysis, artifacts
