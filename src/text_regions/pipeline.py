"""Geometry validation, deterministic IDs, crops, and localization artifacts."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..contracts import DetectionResult, TextRegion
from ..prototype_config import canonical_json
from .adapters import RawGeometry


@dataclass
class LocalizationArtifacts:
    padded_crops: dict[str, np.ndarray] = field(default_factory=dict)
    unpadded_crops: dict[str, np.ndarray] = field(default_factory=dict)
    detection_input: np.ndarray | None = None


def detector_config_hash(config: dict[str, Any]) -> str:
    detector = config.get("detector", config)
    return hashlib.sha256(canonical_json(detector).encode("utf-8")).hexdigest()


def localize_image(image: Image.Image, sample_id: str, detector: Any, config: dict[str, Any]) -> tuple[DetectionResult, LocalizationArtifacts]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    started = time.perf_counter()
    result = DetectionResult(
        schema_version="localization-1",
        sample_id=sample_id,
        image_width=width,
        image_height=height,
        detector_name=getattr(detector, "name", "unknown-detector"),
        detector_version=str(getattr(detector, "version", "unknown")),
        detector_config_sha256=detector_config_hash(config),
    )
    artifacts = LocalizationArtifacts(detection_input=rgb)
    try:
        raw_values = detector.detect(rgb, config)
    except Exception as exc:
        result.status = "detector_unavailable"
        result.warnings.append(str(exc))
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result, artifacts
    normalized: list[tuple[tuple[int, int, int, int], RawGeometry, list[tuple[float, float]]]] = []
    minimum_side = int(config.get("preprocessing", {}).get("minimum_region_side", 3))
    rejected: list[str] = []
    for raw in raw_values or []:
        polygon = getattr(raw, "polygon", None)
        if polygon is None and isinstance(raw, dict):
            polygon = raw.get("polygon")
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            rejected.append("malformed_polygon")
            continue
        points: list[tuple[float, float]] = []
        valid = True
        for point in polygon:
            try:
                x, y = float(point[0]), float(point[1])
                if not math.isfinite(x) or not math.isfinite(y):
                    valid = False
                    break
                points.append((x, y))
            except (TypeError, ValueError, IndexError):
                valid = False
                break
        if not valid or len(points) < 3:
            rejected.append("malformed_point")
            continue
        x1 = max(0, min(width, int(math.floor(min(point[0] for point in points)))))
        y1 = max(0, min(height, int(math.floor(min(point[1] for point in points)))))
        x2 = max(0, min(width, int(math.ceil(max(point[0] for point in points)))))
        y2 = max(0, min(height, int(math.ceil(max(point[1] for point in points)))))
        if x2 <= x1 or y2 <= y1:
            rejected.append("empty_after_clipping")
            continue
        if x2 - x1 < minimum_side or y2 - y1 < minimum_side:
            rejected.append("too_small")
            continue
        clipped_points = [(max(0.0, min(float(width), x)), max(0.0, min(float(height), y))) for x, y in points]
        bbox = (x1, y1, x2 - x1, y2 - y1)
        normalized.append((bbox, raw, clipped_points))
    normalized.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    padding_ratio = float(config.get("preprocessing", {}).get("crop_padding_ratio", 0.05))
    for index, (bbox, raw, polygon) in enumerate(normalized):
        x, y, w, h = bbox
        crop_x = max(0, int(round(x - w * padding_ratio)))
        crop_y = max(0, int(round(y - h * padding_ratio)))
        crop_x2 = min(width, int(round(x + w + w * padding_ratio)))
        crop_y2 = min(height, int(round(y + h + h * padding_ratio)))
        region_id = f"{sample_id}-r{index:03d}"
        confidence = getattr(raw, "confidence", None)
        if confidence is None and isinstance(raw, dict):
            confidence = raw.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None and math.isfinite(float(confidence)) else None
        except (TypeError, ValueError):
            confidence = None
        original_polygon = getattr(raw, "polygon", None)
        if original_polygon is None and isinstance(raw, dict):
            original_polygon = raw.get("polygon")
        clipped = original_polygon is not None and any(
            float(px) != float(clipped_x) or float(py) != float(clipped_y)
            for (px, py), (clipped_x, clipped_y) in zip(original_polygon, polygon)
        )
        region = TextRegion(
            region_id=region_id,
            bbox_px=bbox,
            bbox_norm=(x / width, y / height, w / width, h / height),
            polygon_px=polygon,
            polygon_norm=[(px / width, py / height) for px, py in polygon],
            confidence=confidence,
            quality_flags=["clipped_geometry"] if clipped else [],
            crop_bbox_px=(crop_x, crop_y, crop_x2 - crop_x, crop_y2 - crop_y),
        )
        result.regions.append(region)
        artifacts.unpadded_crops[region_id] = rgb[y:y + h, x:x + w].copy()
        artifacts.padded_crops[region_id] = rgb[crop_y:crop_y2, crop_x:crop_x2].copy()
    result.status = "no_detection" if not result.regions else "ok"
    if result.status == "no_detection":
        result.warnings.append("No valid text-like regions were detected.")
    if rejected:
        result.warnings.append(f"Rejected {len(rejected)} malformed or too-small detection(s).")
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result, artifacts


def localization_json(result: DetectionResult) -> dict[str, Any]:
    return result.to_dict()


def write_localization(path: str | Path, result: DetectionResult) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(localization_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
