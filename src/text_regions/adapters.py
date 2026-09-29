"""Geometry-only detector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..detection.detector import EasyOCRTextDetector


@dataclass
class RawGeometry:
    polygon: list[tuple[float, float]]
    confidence: float | None = None


class EasyOCRDetectorAdapter:
    """Use the existing detection-only EasyOCR integration without recognition."""

    name = "EasyOCR CRAFT (detection-only)"
    version = "1.7.2"

    def __init__(self, config: dict[str, Any]):
        detector = config.get("detector", config)
        self.version = str(detector.get("version", self.version))
        self.weights_identifier = str(detector.get("weights_identifier", "unknown"))
        self.weights_sha256 = detector.get("weights_sha256")
        self._detector = EasyOCRTextDetector(
            languages=tuple(detector.get("languages", ["en"])),
            gpu=bool(detector.get("gpu", False)),
            model_storage_directory=str(detector.get("model_directory", "easyocr-models")),
            user_network_directory=str(detector.get("model_directory", "easyocr-models")) + "/user_network",
            download_enabled=True,
        )

    def detect(self, image_rgb: np.ndarray, config: dict[str, Any] | None = None) -> list[RawGeometry]:
        detector = (config or {}).get("detector", config or {})
        from ..types import DetectionConfig

        detection_config = DetectionConfig(
            text_threshold=float(detector.get("text_threshold", 0.7)),
            low_text=float(detector.get("low_text", 0.4)),
            link_threshold=float(detector.get("link_threshold", 0.4)),
            bbox_min_score=float(detector.get("bbox_min_score", 0.2)),
            min_size=int(detector.get("min_size", 20)),
            canvas_size=int(detector.get("canvas_size", 2560)),
            mag_ratio=float(detector.get("mag_ratio", 1.0)),
        )
        raw = self._detector.detect(image_rgb, detection_config)
        return [RawGeometry(polygon=list(item.polygon), confidence=getattr(item, "confidence", None)) for item in raw]


class FixtureTextDetector:
    """Deterministic detector for generated geometric fixture images."""

    name = "fixture-geometry-detector"
    version = "1.0"

    def __init__(self, threshold: int = 180):
        self.threshold = threshold

    def detect(self, image_rgb: np.ndarray, config: dict[str, Any] | None = None) -> list[RawGeometry]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        mask = (gray < self.threshold).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results: list[RawGeometry] = []
        for index in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            if area < 4 or width < 2 or height < 2:
                continue
            results.append(RawGeometry([(x, y), (x + width, y), (x + width, y + height), (x, y + height)], None))
        return sorted(results, key=lambda item: (item.polygon[0][1], item.polygon[0][0]))
