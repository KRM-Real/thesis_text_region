"""Exploratory sharpness feature."""

from __future__ import annotations

import numpy as np

from src.types import FeatureResult


def compute_laplacian_variance(gray_crop: np.ndarray, minimum_size: int = 3) -> FeatureResult:
    """Compute variance of a 3x3 Laplacian on a native-resolution crop."""

    name = "sharpness_laplacian_variance"
    if gray_crop.ndim != 2 or min(gray_crop.shape, default=0) < minimum_size:
        return FeatureResult(name, "sharpness", "unavailable", reason="Region is too small for a Laplacian calculation.")
    try:
        import cv2

        laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F, ksize=3)
        value = float(np.var(laplacian))
        if not np.isfinite(value):
            raise ValueError("calculated value is not finite")
        return FeatureResult(name, "sharpness", value=value, metadata={"kernel_size": 3})
    except Exception as exc:
        return FeatureResult(name, "sharpness", "unavailable", reason=f"Laplacian calculation failed: {exc}")
