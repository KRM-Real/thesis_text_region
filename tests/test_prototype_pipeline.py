from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.manifest import ManifestError, load_manifest, normalize_label
from src.data.splits import create_split_manifest, validate_split_leakage
from src.features.pipeline import extract_features
from src.fixtures import generate_fixture_dataset
from src.prototype_config import load_prototype_config
from src.text_regions.adapters import FixtureTextDetector


def test_label_alias_is_normalized_without_semantic_features():
    assert normalize_label("real") == "non_ai_generated"
    assert normalize_label("AI-GENERATED") == "ai_generated"


def test_fixture_localization_and_features_are_geometry_only(tmp_path: Path):
    manifest_path = generate_fixture_dataset(tmp_path, count_per_label=1)
    records = load_manifest(manifest_path, tmp_path)
    config = load_prototype_config(Path("configs/prototype.yaml"))
    with Image.open(tmp_path / records[0].relative_path) as image:
        result = extract_features(image, records[0].sample_id, FixtureTextDetector(), config, records[0].label)

    assert result.detection.status == "ok"
    assert result.detection.regions
    assert result.image_feature_record is not None
    assert all(token not in key.lower() for key in result.image_feature_record.values for token in ("recognized_text", "ocr", "word", "semantic"))
    assert all(token not in key.lower() for key in result.detection.to_dict() for token in ("recognized_text", "ocr", "word", "semantic"))
    assert "geometry.area_ratio.mean" in result.image_feature_record.values
    assert "density.union_coverage" in result.image_feature_record.values


def test_no_detection_keeps_fixed_schema_and_quality_flag():
    config = load_prototype_config(Path("configs/prototype.yaml"))
    image = Image.fromarray(np.full((80, 120, 3), 255, dtype=np.uint8), mode="RGB")

    class EmptyDetector:
        name = "empty-test-detector"
        version = "test"

        def detect(self, image_rgb, config=None):
            return []

    result = extract_features(image, "blank", EmptyDetector(), config)
    assert result.image_feature_record is not None
    assert result.image_feature_record.no_detection is True
    assert result.image_feature_record.values["quality.no_detection"] == 1
    assert result.image_feature_record.values["geometry.area_ratio.mean"] is None
    assert "no_detection" in result.image_feature_record.quality_flags


def test_detector_geometry_is_clipped_and_flagged_without_text_fields():
    config = load_prototype_config(Path("configs/prototype.yaml"))
    image = Image.new("RGB", (40, 40), "white")

    class OutOfBoundsDetector:
        name = "out-of-bounds-test-detector"
        version = "test"

        def detect(self, image_rgb, config=None):
            return [{"polygon": [(-4, -3), (18, -3), (18, 12), (-4, 12)], "text": "must be discarded", "confidence": 0.8}]

    result = extract_features(image, "clipped", OutOfBoundsDetector(), config)
    assert result.detection.status == "ok"
    assert result.detection.regions[0].quality_flags == ["clipped_geometry"]
    assert result.detection.regions[0].confidence == 0.8
    assert "text" not in result.detection.regions[0].to_dict()


def test_group_aware_split_and_cross_split_leakage_guard(tmp_path: Path):
    manifest_path = generate_fixture_dataset(tmp_path, count_per_label=6)
    records = load_manifest(manifest_path, tmp_path)
    split_records = create_split_manifest(records, seed=42)
    assert {record.split for record in split_records} == {"train", "validation", "test"}
    groups = {}
    for record in split_records:
        groups.setdefault(record.group_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in groups.values())

    first = split_records[0]
    conflicting = replace(split_records[1], sha256=first.sha256, split="test" if first.split != "test" else "train")
    with pytest.raises(ManifestError, match="exact duplicate"):
        validate_split_leakage([first, conflicting])


def test_manifest_rejects_path_traversal(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    fields = ["sample_id", "relative_path", "label", "category", "source_id", "group_id", "provenance_type", "acquisition_type", "width", "height", "format", "sha256"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "sample_id": "escape",
            "relative_path": "../outside.png",
            "label": "real",
            "category": "receipt",
            "source_id": "test",
            "group_id": "test",
            "provenance_type": "fixture",
            "acquisition_type": "synthetic",
            "width": "1",
            "height": "1",
            "format": "PNG",
            "sha256": "0" * 64,
        })
    with pytest.raises(ManifestError, match="relative_path"):
        load_manifest(manifest, tmp_path)
