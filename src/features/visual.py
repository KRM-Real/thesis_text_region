"""Pixel and geometry-only feature extractors.

All functions in this module accept image pixels and validated geometry.  No
function accepts OCR strings, language, or detector recognition confidence.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import skeletonize

from ..contracts import FeatureSchemaEntry, RegionFeatureResult, TextRegion


REGION_STATISTICS = ("mean", "median", "std", "iqr", "min", "max", "p10", "p90")


def _result(name: str, group: str, value: float | None, formula: str, reason: str | None = None, **metadata: Any) -> RegionFeatureResult:
    status = "ok" if value is not None and math.isfinite(float(value)) else "unavailable"
    return RegionFeatureResult(name, group, None if value is None else float(value), status=status, formula_id=formula, reason=reason, metadata=metadata)


def _missing(name: str, group: str, formula: str, reason: str) -> RegionFeatureResult:
    return RegionFeatureResult(name, group, None, status="unavailable", formula_id=formula, reason=reason)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _entropy(values: np.ndarray, bins: int) -> float | None:
    if values.size == 0:
        return None
    counts, _ = np.histogram(values, bins=bins, range=(0, bins))
    probabilities = counts.astype(float) / max(1, counts.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else None


def _binary_mask(gray: np.ndarray, fraction_range: tuple[float, float]) -> np.ndarray | None:
    if gray.size == 0 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return None
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = [(thresholded > 0), (thresholded == 0)]
    valid = [mask for mask in candidates if fraction_range[0] <= float(mask.mean()) <= fraction_range[1]]
    if not valid:
        return None
    return min(valid, key=lambda mask: float(mask.mean())).astype(np.uint8)


def _orientation(region: TextRegion) -> float | None:
    if not region.polygon_px or len(region.polygon_px) < 3:
        return None
    points = np.asarray(region.polygon_px, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    angle = float(rect[2])
    if angle < -45:
        angle += 90
    return angle


def _component_stats(mask: np.ndarray, min_area: int) -> tuple[list[dict[str, float]], int]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, float]] = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        component_mask = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(contours[0]))) if contours else 0.0
        components.append({
            "area": area,
            "aspect": width / height if height > 0 else float("nan"),
            "compactness": perimeter * perimeter / area if area > 0 else float("nan"),
            "solidity": area / hull_area if hull_area > 0 else float("nan"),
            "x": float(centroids[index][0]),
            "y": float(centroids[index][1]),
        })
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = int(sum(1 for item in hierarchy[0] if item[3] >= 0))
    return components, holes


def _stroke_values(mask: np.ndarray | None) -> tuple[np.ndarray, float] | None:
    if mask is None or not mask.any():
        return None
    distance = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
    skeleton = skeletonize(mask > 0)
    widths = (distance[skeleton] * 2.0).astype(float)
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size == 0:
        return None
    return widths, float(widths.size / mask.size)


def _glcm_features(gray: np.ndarray, levels: int, distances: list[int], angles: list[float]) -> tuple[float | None, float | None]:
    if gray.size == 0 or min(gray.shape) < 2:
        return None, None
    quantized = np.floor(gray.astype(np.float32) * levels / 256.0).clip(0, levels - 1).astype(np.uint8)
    matrix = graycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    return float(np.mean(graycoprops(matrix, "contrast"))), float(np.mean(graycoprops(matrix, "homogeneity")))


def compute_region_features(
    gray: np.ndarray,
    region: TextRegion,
    image_shape: tuple[int, int],
    *,
    padded_rgb: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, RegionFeatureResult]:
    """Compute the configured experimental visual features for one region."""

    config = config or {}
    height, width = image_shape
    groups = set(config.get("enabled_groups", ["geometry_density", "spacing_alignment", "stroke_shape", "edge_gradient", "contrast_sharpness", "texture"]))
    results: dict[str, RegionFeatureResult] = {}
    x, y, w, h = region.bbox_px
    area_ratio = (w * h) / max(1, width * height)
    if "geometry_density" in groups:
        geometry = {
            "geometry.area_ratio": (area_ratio, "geometry-area-ratio-v1"),
            "geometry.width_ratio": (w / max(1, width), "geometry-width-ratio-v1"),
            "geometry.height_ratio": (h / max(1, height), "geometry-height-ratio-v1"),
            "geometry.aspect_ratio": (w / h if h > 0 else None, "geometry-aspect-ratio-v1"),
            "geometry.center_x": ((x + w / 2) / max(1, width), "geometry-center-x-v1"),
            "geometry.center_y": ((y + h / 2) / max(1, height), "geometry-center-y-v1"),
            "geometry.orientation": (_orientation(region), "geometry-orientation-v1"),
        }
        for name, (value, formula) in geometry.items():
            results[name] = _result(name, "geometry_density", value, formula, "detector supplied no polygon orientation" if name.endswith("orientation") and value is None else None)

    if gray.size == 0:
        for name, group, formula in _pixel_feature_names():
            if group in groups:
                results.setdefault(name, _missing(name, group, formula, "empty crop"))
        return results

    if "edge_gradient" in groups:
        low = int(config.get("canny_low", 100))
        high = int(config.get("canny_high", 200))
        edges = cv2.Canny(gray, low, high)
        edge_pixels = float(np.count_nonzero(edges))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        orientation = (np.arctan2(gy, gx) + np.pi) % np.pi
        nonzero = magnitude >= float(config.get("gradient_min_magnitude", 1.0))
        histogram, _ = np.histogram(orientation[nonzero], bins=int(config.get("gradient_orientation_bins", 8)), range=(0, np.pi))
        probabilities = histogram.astype(float) / max(1, histogram.sum())
        probabilities = probabilities[probabilities > 0]
        orientation_entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else None
        continuity = _boundary_continuity(edges)
        results.update({
            "edge.canny_density": _result("edge.canny_density", "edge_gradient", edge_pixels / gray.size, "edge-canny-density-v1"),
            "gradient.magnitude_mean": _result("gradient.magnitude_mean", "edge_gradient", float(magnitude.mean()), "gradient-sobel-mean-v1"),
            "gradient.magnitude_std": _result("gradient.magnitude_std", "edge_gradient", float(magnitude.std()), "gradient-sobel-std-v1"),
            "gradient.magnitude_p90": _result("gradient.magnitude_p90", "edge_gradient", float(np.percentile(magnitude, 90)), "gradient-sobel-p90-v1"),
            "gradient.orientation_entropy": _result("gradient.orientation_entropy", "edge_gradient", orientation_entropy, "gradient-orientation-entropy-v1", "no gradients above threshold" if orientation_entropy is None else None),
            "edge.boundary_continuity": _result("edge.boundary_continuity", "edge_gradient", continuity, "edge-boundary-component-ratio-v1", "no boundary-touching edge component" if continuity is None else None),
        })

    if "contrast_sharpness" in groups:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        tenengrad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        tenengrad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = tenengrad_x ** 2 + tenengrad_y ** 2
        ring_delta = _ring_delta(gray, padded_rgb, region)
        results.update({
            "contrast.gray_std": _result("contrast.gray_std", "contrast_sharpness", float(gray.std()), "contrast-gray-std-v1"),
            "contrast.local_ring_delta": _result("contrast.local_ring_delta", "contrast_sharpness", ring_delta, "contrast-region-ring-delta-v1", "padding ring unavailable" if ring_delta is None else None),
            "contrast.p10_p90_range": _result("contrast.p10_p90_range", "contrast_sharpness", float(np.percentile(gray, 90) - np.percentile(gray, 10)), "contrast-p10-p90-v1"),
            "sharpness.laplacian_variance": _result("sharpness.laplacian_variance", "contrast_sharpness", float(laplacian.var()) if min(gray.shape) >= 3 else None, "sharpness-laplacian-variance-v1", "crop smaller than 3 pixels" if min(gray.shape) < 3 else None),
            "sharpness.tenengrad_mean": _result("sharpness.tenengrad_mean", "contrast_sharpness", float(tenengrad.mean()), "sharpness-tenengrad-mean-v1"),
        })

    needs_mask = "stroke_shape" in groups or "texture" in groups
    mask = _binary_mask(gray, tuple(config.get("mask_foreground_fraction", [0.01, 0.60]))) if needs_mask else None
    if "stroke_shape" in groups:
        stroke = _stroke_values(mask)
        if stroke is None:
            for name, formula in (("stroke.width_median", "stroke-distance-median-v1"), ("stroke.width_std", "stroke-distance-std-v1"), ("stroke.width_iqr", "stroke-distance-iqr-v1"), ("stroke.width_p90", "stroke-distance-p90-v1"), ("stroke.valid_fraction", "stroke-valid-fraction-v1")):
                results[name] = _missing(name, "stroke_shape", formula, "no stable foreground skeleton")
        else:
            widths, valid_fraction = stroke
            results.update({
                "stroke.width_median": _result("stroke.width_median", "stroke_shape", float(np.median(widths)), "stroke-distance-median-v1"),
                "stroke.width_std": _result("stroke.width_std", "stroke_shape", float(widths.std()), "stroke-distance-std-v1"),
                "stroke.width_iqr": _result("stroke.width_iqr", "stroke_shape", float(np.percentile(widths, 75) - np.percentile(widths, 25)), "stroke-distance-iqr-v1"),
                "stroke.width_p90": _result("stroke.width_p90", "stroke_shape", float(np.percentile(widths, 90)), "stroke-distance-p90-v1"),
                "stroke.valid_fraction": _result("stroke.valid_fraction", "stroke_shape", valid_fraction, "stroke-valid-fraction-v1"),
            })
        components, holes = _component_stats(mask, int(config.get("component_min_area", 3))) if mask is not None else ([], 0)
        if components:
            crop_area = float(gray.size)
            results.update({
                "shape.component_count": _result("shape.component_count", "stroke_shape", len(components), "shape-component-count-v1"),
                "shape.component_area_median": _result("shape.component_area_median", "stroke_shape", float(np.median([item["area"] for item in components]) / crop_area), "shape-component-area-median-v1"),
                "shape.component_aspect_median": _result("shape.component_aspect_median", "stroke_shape", float(np.nanmedian([item["aspect"] for item in components])), "shape-component-aspect-median-v1"),
                "shape.compactness_mean": _result("shape.compactness_mean", "stroke_shape", float(np.nanmean([item["compactness"] for item in components])), "shape-compactness-mean-v1"),
                "shape.solidity_mean": _result("shape.solidity_mean", "stroke_shape", float(np.nanmean([item["solidity"] for item in components])), "shape-solidity-mean-v1"),
                "shape.hole_count": _result("shape.hole_count", "stroke_shape", holes, "shape-hole-count-v1"),
            })
        else:
            for name, formula in (("shape.component_count", "shape-component-count-v1"), ("shape.component_area_median", "shape-component-area-median-v1"), ("shape.component_aspect_median", "shape-component-aspect-median-v1"), ("shape.compactness_mean", "shape-compactness-mean-v1"), ("shape.solidity_mean", "shape-solidity-mean-v1"), ("shape.hole_count", "shape-hole-count-v1")):
                results[name] = _missing(name, "stroke_shape", formula, "no valid connected components")

    if "texture" in groups:
        lbp = local_binary_pattern(gray, int(config.get("lbp_points", 8)), float(config.get("lbp_radius", 1)), method="uniform")
        bins = int(config.get("lbp_points", 8)) + 2
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
        probabilities = hist.astype(float) / max(1, hist.sum())
        positive = probabilities[probabilities > 0]
        lbp_entropy = float(-(positive * np.log2(positive)).sum()) if positive.size else None
        glcm_contrast, glcm_homogeneity = _glcm_features(gray, int(config.get("glcm_levels", 16)), list(config.get("glcm_distances", [1])), list(config.get("glcm_angles", [0.0])))
        results.update({
            "texture.gray_entropy": _result("texture.gray_entropy", "texture", _entropy((gray.astype(np.float32) * 256 / 256).astype(np.uint8), 256), "texture-gray-entropy-v1"),
            "texture.lbp_uniform_fraction": _result("texture.lbp_uniform_fraction", "texture", float(np.mean(lbp < bins - 1)), "texture-lbp-uniform-fraction-v1"),
            "texture.lbp_entropy": _result("texture.lbp_entropy", "texture", lbp_entropy, "texture-lbp-entropy-v1"),
            "texture.glcm_contrast": _result("texture.glcm_contrast", "texture", glcm_contrast, "texture-glcm-contrast-v1", "GLCM unavailable for crop" if glcm_contrast is None else None),
            "texture.glcm_homogeneity": _result("texture.glcm_homogeneity", "texture", glcm_homogeneity, "texture-glcm-homogeneity-v1", "GLCM unavailable for crop" if glcm_homogeneity is None else None),
        })
    return results


def _pixel_feature_names() -> list[tuple[str, str, str]]:
    return [
        ("edge.canny_density", "edge_gradient", "edge-canny-density-v1"),
        ("gradient.magnitude_mean", "edge_gradient", "gradient-sobel-mean-v1"),
        ("gradient.magnitude_std", "edge_gradient", "gradient-sobel-std-v1"),
        ("gradient.magnitude_p90", "edge_gradient", "gradient-sobel-p90-v1"),
        ("gradient.orientation_entropy", "edge_gradient", "gradient-orientation-entropy-v1"),
        ("edge.boundary_continuity", "edge_gradient", "edge-boundary-component-ratio-v1"),
        ("contrast.gray_std", "contrast_sharpness", "contrast-gray-std-v1"),
        ("contrast.local_ring_delta", "contrast_sharpness", "contrast-region-ring-delta-v1"),
        ("contrast.p10_p90_range", "contrast_sharpness", "contrast-p10-p90-v1"),
        ("sharpness.laplacian_variance", "contrast_sharpness", "sharpness-laplacian-variance-v1"),
        ("sharpness.tenengrad_mean", "contrast_sharpness", "sharpness-tenengrad-mean-v1"),
    ]


def _boundary_continuity(edges: np.ndarray) -> float | None:
    if not np.any(edges):
        return None
    components, labels = cv2.connectedComponents((edges > 0).astype(np.uint8), connectivity=8)
    sizes = []
    total = float(np.count_nonzero(edges))
    h, w = edges.shape
    boundary_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist()) - {0}
    for label in boundary_labels:
        sizes.append(int(np.count_nonzero(labels == label)))
    return max(sizes) / total if sizes else None


def _ring_delta(gray: np.ndarray, padded_rgb: np.ndarray | None, region: TextRegion | None = None) -> float | None:
    if padded_rgb is None or padded_rgb.shape[:2] == gray.shape:
        return None
    padded_gray = cv2.cvtColor(padded_rgb, cv2.COLOR_RGB2GRAY)
    if padded_gray.size <= gray.size:
        return None
    center_h, center_w = gray.shape
    if region is not None and region.crop_bbox_px is not None:
        crop_x, crop_y, _, _ = region.crop_bbox_px
        region_x, region_y, _, _ = region.bbox_px
        x = max(0, region_x - crop_x)
        y = max(0, region_y - crop_y)
    else:
        y = max(0, (padded_gray.shape[0] - center_h) // 2)
        x = max(0, (padded_gray.shape[1] - center_w) // 2)
    center = padded_gray[y:y + center_h, x:x + center_w]
    ring_mask = np.ones(padded_gray.shape, dtype=bool)
    ring_mask[y:y + center_h, x:x + center_w] = False
    ring_values = padded_gray[ring_mask]
    return float(center.mean() - ring_values.mean()) if ring_values.size else None


def image_geometry_features(
    regions: list[TextRegion],
    image_shape: tuple[int, int],
    row_tolerance_ratio: float = 0.02,
    enabled_groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, RegionFeatureResult]:
    """Compute image-level spatial features from region geometry only."""

    height, width = image_shape
    groups = set(enabled_groups or {"geometry_density", "spacing_alignment"})
    result: dict[str, RegionFeatureResult] = {}
    if "geometry_density" in groups:
        result.update({
            "density.region_count": _result("density.region_count", "geometry_density", len(regions), "density-region-count-v1"),
            "density.union_coverage": _result("density.union_coverage", "geometry_density", _union_coverage(regions, width, height), "density-union-coverage-v1"),
            "density.area_sum_ratio": _result("density.area_sum_ratio", "geometry_density", sum(region.bbox_px[2] * region.bbox_px[3] for region in regions) / max(1, width * height), "density-area-sum-ratio-v1"),
        })
    if "spacing_alignment" not in groups:
        return result
    if not regions:
        for name, group, formula in (
            ("layout.centroid_dispersion_x", "spacing_alignment", "layout-centroid-dispersion-x-v1"),
            ("layout.centroid_dispersion_y", "spacing_alignment", "layout-centroid-dispersion-y-v1"),
            ("layout.row_alignment_error", "spacing_alignment", "layout-row-alignment-error-v1"),
            ("layout.overlap_ratio", "spacing_alignment", "layout-overlap-ratio-v1"),
            ("layout.nearest_horizontal_gap", "spacing_alignment", "layout-nearest-horizontal-gap-v1"),
            ("layout.nearest_vertical_gap", "spacing_alignment", "layout-nearest-vertical-gap-v1"),
        ):
            result[name] = _missing(name, group, formula, "no valid regions")
        return result
    centers = np.asarray([((x + w) / 2 / max(1, width), (y + h) / 2 / max(1, height)) for x, y, w, h in (region.bbox_px for region in regions)], dtype=float)
    result["layout.centroid_dispersion_x"] = _result("layout.centroid_dispersion_x", "spacing_alignment", float(centers[:, 0].std()), "layout-centroid-dispersion-x-v1")
    result["layout.centroid_dispersion_y"] = _result("layout.centroid_dispersion_y", "spacing_alignment", float(centers[:, 1].std()), "layout-centroid-dispersion-y-v1")
    result["layout.row_alignment_error"] = _result("layout.row_alignment_error", "spacing_alignment", _row_error(centers[:, 1], row_tolerance_ratio), "layout-row-alignment-error-v1")
    ious = [_iou(left.bbox_px, right.bbox_px) for index, left in enumerate(regions) for right in regions[index + 1:]]
    result["layout.overlap_ratio"] = _result("layout.overlap_ratio", "spacing_alignment", float(np.mean(ious)) if ious else 0.0, "layout-overlap-ratio-v1")
    result["layout.nearest_horizontal_gap"] = _result("layout.nearest_horizontal_gap", "spacing_alignment", _nearest_gap(regions, width, height, horizontal=True), "layout-nearest-horizontal-gap-v1", "no horizontally related pair" if _nearest_gap(regions, width, height, horizontal=True) is None else None)
    result["layout.nearest_vertical_gap"] = _result("layout.nearest_vertical_gap", "spacing_alignment", _nearest_gap(regions, width, height, horizontal=False), "layout-nearest-vertical-gap-v1", "no vertically related pair" if _nearest_gap(regions, width, height, horizontal=False) is None else None)
    return result


def _union_coverage(regions: list[TextRegion], width: int, height: int) -> float:
    if not regions:
        return 0.0
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if region.polygon_px and len(region.polygon_px) >= 3:
            points = np.asarray(region.polygon_px, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        else:
            x, y, w, h = region.bbox_px
            mask[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = 1
    return float(mask.mean())


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - inter
    return inter / union if union else 0.0


def _nearest_gap(regions: list[TextRegion], width: int, height: int, horizontal: bool) -> float | None:
    values: list[float] = []
    for index, left in enumerate(regions):
        lx, ly, lw, lh = left.bbox_px
        for right in regions[index + 1:]:
            rx, ry, rw, rh = right.bbox_px
            if horizontal:
                if min(ly + lh, ry + rh) <= max(ly, ry):
                    continue
                gap = max(rx - (lx + lw), lx - (rx + rw), 0)
                if gap > 0:
                    values.append(gap / max(1, width))
            else:
                if min(lx + lw, rx + rw) <= max(lx, rx):
                    continue
                gap = max(ry - (ly + lh), ly - (ry + rh), 0)
                if gap > 0:
                    values.append(gap / max(1, height))
    return min(values) if values else None


def _row_error(values: np.ndarray, tolerance: float) -> float:
    rows: list[list[float]] = []
    for value in sorted(values):
        target = next((row for row in rows if abs(value - float(np.mean(row))) <= tolerance), None)
        if target is None:
            rows.append([float(value)])
        else:
            target.append(float(value))
    errors = [abs(value - float(np.mean(row))) for row in rows for value in row]
    return float(np.mean(errors)) if errors else 0.0


def aggregate_region_features(
    region_results: list[dict[str, RegionFeatureResult]],
    image_features: dict[str, RegionFeatureResult],
    sample_id: str,
    label: str | None,
    schema_version: str = "features-1",
    enabled_groups: list[str] | tuple[str, ...] | None = None,
    statistics: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, float | None], list[FeatureSchemaEntry], list[str]]:
    """Aggregate region results into one fixed-length image-level vector."""

    specs = all_region_feature_specs(enabled_groups)
    statistics = tuple(statistics or REGION_STATISTICS)
    all_names = sorted({name for result in region_results for name in result} | {item[0] for item in specs})
    values: dict[str, float | None] = {}
    schema: list[FeatureSchemaEntry] = []
    quality: list[str] = []
    for name in all_names:
        available = [float(result[name].value) for result in region_results if result.get(name) and result[name].value is not None and result[name].status == "ok"]
        source = next((result[name] for result in region_results if name in result), None)
        if source is None:
            spec = next(item for item in specs if item[0] == name)
            source = RegionFeatureResult(name, spec[1], None, status="unavailable", formula_id=spec[2], reason="no valid region measurements")
        for statistic in statistics:
            column = f"{name}.{statistic}"
            value = _aggregate_value(available, statistic)
            values[column] = value
            if value is None:
                quality.append(f"missing:{column}")
            schema.append(FeatureSchemaEntry(column, source.group, "image", f"{source.formula_id}+{statistic}", schema_version, (name,), expected_range=(None, None)))
    for name, result in sorted(image_features.items()):
        values[name] = result.value
        schema.append(FeatureSchemaEntry(name, result.group, "image", result.formula_id, schema_version, (), expected_range=(None, None)))
        if result.value is None:
            quality.append(f"missing:{name}")
    values["quality.no_detection"] = 1 if not region_results else 0
    schema.append(FeatureSchemaEntry("quality.no_detection", "quality", "image", "quality-no-detection-v1", schema_version, (), expected_range=(0, 1), diagnostic_only=False))
    return values, schema, sorted(set(quality))


def all_region_feature_specs(enabled_groups: list[str] | tuple[str, ...] | None = None) -> list[tuple[str, str, str]]:
    """Return the stable region-level schema, including unavailable candidates."""

    specs = [
        ("geometry.area_ratio", "geometry_density", "geometry-area-ratio-v1"),
        ("geometry.width_ratio", "geometry_density", "geometry-width-ratio-v1"),
        ("geometry.height_ratio", "geometry_density", "geometry-height-ratio-v1"),
        ("geometry.aspect_ratio", "geometry_density", "geometry-aspect-ratio-v1"),
        ("geometry.center_x", "geometry_density", "geometry-center-x-v1"),
        ("geometry.center_y", "geometry_density", "geometry-center-y-v1"),
        ("geometry.orientation", "geometry_density", "geometry-orientation-v1"),
    ]
    specs.extend(_pixel_feature_names())
    specs.extend([
        ("stroke.width_median", "stroke_shape", "stroke-distance-median-v1"),
        ("stroke.width_std", "stroke_shape", "stroke-distance-std-v1"),
        ("stroke.width_iqr", "stroke_shape", "stroke-distance-iqr-v1"),
        ("stroke.width_p90", "stroke_shape", "stroke-distance-p90-v1"),
        ("stroke.valid_fraction", "stroke_shape", "stroke-valid-fraction-v1"),
        ("shape.component_count", "stroke_shape", "shape-component-count-v1"),
        ("shape.component_area_median", "stroke_shape", "shape-component-area-median-v1"),
        ("shape.component_aspect_median", "stroke_shape", "shape-component-aspect-median-v1"),
        ("shape.compactness_mean", "stroke_shape", "shape-compactness-mean-v1"),
        ("shape.solidity_mean", "stroke_shape", "shape-solidity-mean-v1"),
        ("shape.hole_count", "stroke_shape", "shape-hole-count-v1"),
        ("texture.gray_entropy", "texture", "texture-gray-entropy-v1"),
        ("texture.lbp_uniform_fraction", "texture", "texture-lbp-uniform-fraction-v1"),
        ("texture.lbp_entropy", "texture", "texture-lbp-entropy-v1"),
        ("texture.glcm_contrast", "texture", "texture-glcm-contrast-v1"),
        ("texture.glcm_homogeneity", "texture", "texture-glcm-homogeneity-v1"),
    ])
    if enabled_groups is None:
        return specs
    enabled = set(enabled_groups)
    return [item for item in specs if item[1] in enabled]


def _aggregate_value(values: list[float], statistic: str) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if statistic == "mean":
        return float(array.mean())
    if statistic == "median":
        return float(np.median(array))
    if statistic == "std":
        return float(array.std(ddof=0))
    if statistic == "iqr":
        return float(np.percentile(array, 75) - np.percentile(array, 25))
    if statistic == "min":
        return float(array.min())
    if statistic == "max":
        return float(array.max())
    if statistic == "p10":
        return float(np.percentile(array, 10))
    if statistic == "p90":
        return float(np.percentile(array, 90))
    raise ValueError(f"unsupported aggregation statistic: {statistic}")
