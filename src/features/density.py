"""Exploratory image-level text density measurement."""

from __future__ import annotations

import numpy as np

from src.types import BoundingBox, FeatureResult


def compute_text_region_area_ratio(
    boxes: list[BoundingBox], image_width: int, image_height: int
) -> FeatureResult:
    """Compute raster union area of detector boxes divided by image area."""

    name = "text_region_area_ratio"
    if image_width <= 0 or image_height <= 0:
        return FeatureResult(name, "density", "unavailable", reason="Image dimensions are invalid.")
    try:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for box in boxes:
            x1 = max(0, min(image_width, int(box.x)))
            y1 = max(0, min(image_height, int(box.y)))
            x2 = max(x1, min(image_width, int(box.x2)))
            y2 = max(y1, min(image_height, int(box.y2)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        value = float(mask.mean())
        return FeatureResult(name, "density", value=value, metadata={"area_method": "union_of_boxes"})
    except Exception as exc:
        return FeatureResult(name, "density", "unavailable", reason=f"Text-density calculation failed: {exc}")
