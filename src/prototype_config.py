"""Configuration loader for the fixture-first prototype pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "prototype-config-1",
    "dataset": {
        "root": "data",
        "manifest": "data/manifests/audited_manifest.csv",
        "category": "receipt",
        "labels": ["non_ai_generated", "ai_generated"],
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42},
    },
    "detector": {
        "backend": "easyocr_craft_detection_only",
        "status": "PILOT_DECISION",
        "version": "1.7.2",
        "weights_identifier": "EasyOCR CRAFT default detector weights",
        "weights_sha256": None,
        "languages": ["en"],
        "gpu": False,
        "recognizer": False,
        "text_threshold": 0.7,
        "low_text": 0.4,
        "link_threshold": 0.4,
        "bbox_min_score": 0.2,
        "min_size": 20,
        "canvas_size": 2560,
        "mag_ratio": 1.0,
        "model_directory": "easyocr-models",
    },
    "preprocessing": {
        "feature_grayscale": "opencv_rgb2gray",
        "detector_resize": "preserve_aspect_ratio",
        "detector_scale": 1.0,
        "crop_padding_ratio": 0.05,
        "minimum_region_side": 3,
        "mask_policy": "polygon_when_available_else_box",
        "resize_feature_crops": False,
    },
    "features": {
        "status": "EXPERIMENTAL_FEATURE",
        "schema_version": "features-1",
        "enabled_groups": [
            "geometry_density",
            "spacing_alignment",
            "stroke_shape",
            "edge_gradient",
            "contrast_sharpness",
            "texture",
        ],
        "canny_low": 100,
        "canny_high": 200,
        "gradient_orientation_bins": 8,
        "gradient_min_magnitude": 1.0,
        "row_tolerance_ratio": 0.02,
        "component_min_area": 3,
        "glcm_levels": 16,
        "glcm_distances": [1],
        "glcm_angles": [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345],
        "lbp_points": 8,
        "lbp_radius": 1,
        "mask_foreground_fraction": [0.01, 0.60],
    },
    "aggregation": {
        "status": "PILOT_DECISION",
        "schema_version": "aggregation-1",
        "statistics": ["mean", "median", "std", "iqr", "min", "max", "p10", "p90"],
        "std_ddof": 0,
        "no_detection_policy": "retain_with_missing_and_indicator",
    },
    "models": {
        "positive_label": "ai_generated",
        "selection_metric": "f1_positive",
        "threshold_method": "validation_f1",
        "random_seed": 42,
        "logistic_regression": {"C": [0.1, 1.0, 10.0], "max_iter": 2000},
        "svm": {"kernels": ["linear", "rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
        "random_forest": {"n_estimators": [200, 500], "max_depth": [None, 10], "min_samples_leaf": [1, 2, 5]},
    },
    "artifacts": {"root": "artifacts/runs"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_prototype_config(path: str | Path = "configs/prototype.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Prototype configuration root must be a mapping")
    else:
        loaded = {}
    config = _deep_merge(DEFAULT_CONFIG, loaded)
    config["_path"] = str(config_path)
    config["_sha256"] = config_hash({k: v for k, v in config.items() if not k.startswith("_")})
    return config


def write_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    Path(path).write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
