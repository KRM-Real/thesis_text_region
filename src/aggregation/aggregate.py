"""Transparent aggregation of region-level visual measurements."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from src.types import FeatureResult


def aggregate_region_features(
    region_features: Iterable[dict[str, FeatureResult]],
    statistics: tuple[str, ...] = ("mean", "std"),
) -> dict[str, FeatureResult]:
    """Aggregate valid region values using explicitly selected statistics.

    Standard deviation is population standard deviation. A single valid region
    therefore has ``std == 0``. If no region has a valid value, all requested
    outputs are unavailable with an explicit reason.
    """

    allowed_statistics = {"mean", "std", "min", "max", "median"}
    unknown = set(statistics) - allowed_statistics
    if unknown:
        raise ValueError(f"Unsupported aggregation statistic(s): {', '.join(sorted(unknown))}")

    # Materialize once because callers commonly pass a generator built while
    # processing crops. This also lets us preserve a stable unavailable schema
    # when every region measurement fails.
    region_feature_list = list(region_features)
    grouped: dict[str, list[tuple[str, float]]] = {}
    for features in region_feature_list:
        for name, result in features.items():
            if result.status != "ok" or result.value is None:
                continue
            try:
                numeric_value = float(result.value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                grouped.setdefault(name, []).append((result.group, numeric_value))

    aggregated: dict[str, FeatureResult] = {}
    for name, values in grouped.items():
        group = values[0][0]
        array = np.asarray([value for _, value in values], dtype=float)
        for statistic in statistics:
            if statistic == "mean":
                value = float(np.mean(array))
            elif statistic == "std":
                value = float(np.std(array, ddof=0))
            elif statistic == "min":
                value = float(np.min(array))
            elif statistic == "max":
                value = float(np.max(array))
            elif statistic == "median":
                value = float(np.median(array))
            else:
                raise ValueError(f"Unsupported aggregation statistic: {statistic}")
            aggregated[f"{name}_{statistic}"] = FeatureResult(
                f"{name}_{statistic}",
                group,
                value=value,
                metadata={"source_feature": name, "statistic": statistic, "valid_region_count": len(values)},
            )

    # Preserve a stable schema even when there are no detections or every
    # measurement failed. The registry is intentionally kept local to avoid a
    # circular import and makes this helper useful with arbitrary features.
    feature_names = sorted({name for features in region_feature_list for name in features})
    for name in feature_names:
        if name not in grouped:
            source = next(features for features in region_feature_list if name in features)
            group = source[name].group
            for statistic in statistics:
                key = f"{name}_{statistic}"
                aggregated[key] = FeatureResult(
                    key,
                    group,
                    "unavailable",
                    reason="No valid region measurements were available.",
                    metadata={"source_feature": name, "statistic": statistic, "valid_region_count": 0},
                )
    return aggregated
