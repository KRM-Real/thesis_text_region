"""Exploratory grayscale contrast feature."""

from __future__ import annotations

import numpy as np

from src.types import FeatureResult


def compute_intensity_std(gray_crop: np.ndarray) -> FeatureResult:
    """Compute population standard deviation of grayscale crop intensities."""

    name = "intensity_std"
    if gray_crop.ndim != 2 or gray_crop.size == 0:
        return FeatureResult(name, "contrast", "unavailable", reason="Region contains no grayscale pixels.")
    try:
        value = float(np.std(gray_crop.astype(np.float64), ddof=0))
        if not np.isfinite(value):
            raise ValueError("calculated value is not finite")
        return FeatureResult(name, "contrast", value=value, metadata={"statistic": "population_std"})
    except Exception as exc:
        return FeatureResult(name, "contrast", "unavailable", reason=f"Intensity calculation failed: {exc}")
