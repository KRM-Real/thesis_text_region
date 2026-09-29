"""Exploratory edge and rendering features."""

from __future__ import annotations

import numpy as np

from src.types import FeatureResult


def edge_map(gray_crop: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray | None:
    """Return a uint8 Canny edge map, or ``None`` for an empty crop."""

    if gray_crop.ndim != 2 or gray_crop.size == 0:
        return None
    try:
        import cv2

        return cv2.Canny(gray_crop, int(low_threshold), int(high_threshold))
    except Exception:
        return None


def compute_edge_density(
    gray_crop: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    minimum_size: int = 3,
) -> FeatureResult:
    """Compute the fraction of Canny edge pixels in a crop."""

    name = "edge_density"
    if gray_crop.ndim != 2 or min(gray_crop.shape, default=0) < minimum_size:
        return FeatureResult(name, "edge_rendering", "unavailable", reason="Region is too small for edge calculation.")
    if low_threshold < 0 or high_threshold <= low_threshold:
        return FeatureResult(name, "edge_rendering", "unavailable", reason="Canny thresholds are invalid.")
    try:
        edges = edge_map(gray_crop, low_threshold, high_threshold)
        if edges is None or edges.size == 0:
            return FeatureResult(name, "edge_rendering", "unavailable", reason="Edge map is empty.")
        value = float(np.count_nonzero(edges) / edges.size)
        return FeatureResult(
            name,
            "edge_rendering",
            value=value,
            metadata={"method": "canny", "low_threshold": low_threshold, "high_threshold": high_threshold},
        )
    except Exception as exc:
        return FeatureResult(name, "edge_rendering", "unavailable", reason=f"Edge calculation failed: {exc}")
