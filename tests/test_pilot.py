from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.aggregation.aggregate import aggregate_region_features
from src.batch import audit_dataset, dataset_inventory, discover_dataset, run_batch
from src.config import load_pilot_config
from src.detection.detector import EasyOCRTextDetector
from src.detection.types import RawDetection
from src.export.feature_export import (
    build_analysis_zip,
    feature_manifest,
    image_features_csv,
    json_bytes,
    region_features_csv,
)
from src.features.contrast import compute_intensity_std
from src.features.density import compute_text_region_area_ratio
from src.features.edges import compute_edge_density
from src.features.sharpness import compute_laplacian_variance
from src.pipeline import analyze_image
from src.preprocessing.image_ops import (
    ImageInputError,
    clip_bbox,
    crop_region,
    load_image,
    prepare_detection_image,
)
from src.types import BoundingBox, FeatureResult, FeatureConfig, PilotConfig
from src.visualization.overlays import draw_detection_overlay


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_streamlit_initial_screen_renders_without_loading_detector():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["Receipt text-region prototype"]
    assert len(app.file_uploader) == 1
    assert any(button.label == "Analyze image" for button in app.button)


class FakeDetector:
    name = "fake detector"
    status = "TEST_ONLY"

    def __init__(self, detections: list[RawDetection]):
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        return self.detections


class FailingDetector:
    name = "failing detector"
    status = "TEST_ONLY"

    def detect(self, image: np.ndarray):
        raise RuntimeError("test detector failure")


class RecordingDetector:
    name = "recording detector"
    status = "TEST_ONLY"

    def __init__(self, detections: list[RawDetection]):
        self.detections = detections
        self.shapes: list[tuple[int, int]] = []

    def detect(self, image: np.ndarray, config=None) -> list[RawDetection]:
        self.shapes.append(image.shape[:2])
        return self.detections


class NestedEasyOCRReader:
    """Mimic EasyOCR's per-image nesting for boxes and free polygons."""

    def detect(self, image: np.ndarray):
        return (
            [[[np.int32(1), np.int32(8), np.int32(2), np.int32(7)]]],
            [
                [
                    [
                        [np.float64(10.5), np.float64(3.0)],
                        [np.float64(16.0), np.float64(4.0)],
                        [np.float64(15.0), np.float64(9.0)],
                        [np.float64(9.5), np.float64(8.0)],
                    ]
                ]
            ],
        )


class ConfigRecordingReader:
    def __init__(self):
        self.kwargs = None

    def detect(self, image, **kwargs):
        self.kwargs = kwargs
        return ([], [])


def test_load_image_rejects_invalid_bytes_and_composites_transparency():
    transparent = Image.new("RGBA", (8, 6), (10, 20, 30, 0))
    loaded = load_image(_png_bytes(transparent))
    assert loaded.mode == "RGB"
    assert loaded.size == (8, 6)
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    try:
        load_image(b"not an image")
    except ImageInputError as exc:
        assert "decode" in str(exc).lower()
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("invalid bytes should fail")


def test_easyocr_adapter_flattens_per_image_boxes_and_free_polygons():
    detector = object.__new__(EasyOCRTextDetector)
    detector.reader = NestedEasyOCRReader()

    detections = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].polygon == [(1.0, 2.0), (8.0, 2.0), (8.0, 7.0), (1.0, 7.0)]
    assert detections[1].polygon == [(10.5, 3.0), (16.0, 4.0), (15.0, 9.0), (9.5, 8.0)]


def test_easyocr_adapter_forwards_configured_localization_thresholds():
    detector = object.__new__(EasyOCRTextDetector)
    detector.reader = ConfigRecordingReader()
    config = load_pilot_config().detection
    detector.detect(np.zeros((20, 20, 3), dtype=np.uint8), config)
    assert detector.reader.kwargs["text_threshold"] == config.text_threshold
    assert detector.reader.kwargs["low_text"] == config.low_text
    assert detector.reader.kwargs["bbox_min_score"] == config.bbox_min_score


def test_crop_clips_out_of_bounds_and_applies_padding():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    clipped = clip_bbox(BoundingBox(-4, 7, 20, 8), 12, 10)
    assert clipped == BoundingBox(0, 7, 12, 3)
    crop, crop_box = crop_region(rgb, BoundingBox(2, 2, 3, 3), padding=4)
    assert crop is not None
    assert crop.shape == (9, 9, 3)
    assert crop_box == BoundingBox(0, 0, 9, 9)


def test_detection_preparation_and_coordinate_restore_are_explicit():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    prepared = prepare_detection_image(rgb, load_pilot_config().preprocessing)
    assert prepared.image_rgb.shape[:2] == (10, 12)
    scaled = prepare_detection_image(
        rgb,
        load_pilot_config().preprocessing.__class__(mode="clahe", scale=2.0),
    )
    assert scaled.image_rgb.shape[:2] == (20, 24)
    assert scaled.to_dict()["operations"] == ["clahe_grayscale", "resize_2x"]


def test_scaled_detection_coordinates_return_to_native_image():
    image = Image.new("RGB", (20, 20), "white")
    detector = RecordingDetector([RawDetection([(4, 4), (16, 4), (16, 12), (4, 12)])])
    config = load_pilot_config()
    config = config.__class__(
        aggregation=config.aggregation,
        feature_config=config.feature_config,
        detection=config.detection,
        preprocessing=config.preprocessing.__class__(mode="none", scale=2.0),
        batch=config.batch,
    )
    analysis, artifacts = analyze_image(image, detector, config=config)
    assert detector.shapes == [(40, 40)]
    assert analysis.regions[0].bbox == BoundingBox(2, 2, 6, 4)
    assert artifacts.crops["region_001"].shape[:2] == (4, 6)


def test_initial_features_are_deterministic_and_explicit():
    gray = np.zeros((12, 20), dtype=np.uint8)
    gray[4:8, 3:17] = 255
    sharpness = compute_laplacian_variance(gray)
    edges = compute_edge_density(gray)
    contrast = compute_intensity_std(gray)
    assert sharpness.status == "ok" and sharpness.value is not None and sharpness.value > 0
    assert edges.status == "ok" and 0 <= edges.value <= 1
    assert contrast.status == "ok" and contrast.value > 0
    tiny = np.zeros((2, 2), dtype=np.uint8)
    assert compute_laplacian_variance(tiny).status == "unavailable"
    assert compute_edge_density(tiny).status == "unavailable"
    assert compute_intensity_std(np.empty((0, 0), dtype=np.uint8)).status == "unavailable"


def test_density_uses_union_area_for_overlapping_boxes():
    result = compute_text_region_area_ratio(
        [BoundingBox(0, 0, 5, 5), BoundingBox(2, 2, 5, 5)], 10, 10
    )
    assert result.status == "ok"
    assert result.value == pytest.approx(0.41)  # 25 + 25 - 9 pixels, divided by 100


def test_aggregation_materializes_generators_and_handles_single_region():
    first = {"edge_density": FeatureResult("edge_density", "edge_rendering", value=0.2)}
    second = {"edge_density": FeatureResult("edge_density", "edge_rendering", value=0.6)}
    aggregate = aggregate_region_features((item for item in [first, second]))
    assert aggregate["edge_density_mean"].value == 0.4
    assert aggregate["edge_density_std"].value == pytest.approx(0.2)
    one = aggregate_region_features([first])
    assert one["edge_density_std"].value == 0
    unavailable = aggregate_region_features(
        [{"edge_density": FeatureResult("edge_density", "edge_rendering", "unavailable", reason="tiny")}]
    )
    assert unavailable["edge_density_mean"].status == "unavailable"
    assert unavailable["edge_density_mean"].value is None


def test_pipeline_preserves_regions_but_excludes_diagnostics_from_model_record():
    image = Image.fromarray(np.full((32, 40, 3), 180, dtype=np.uint8))
    image_array = np.asarray(image).copy()
    image_array[8:18, 5:26] = 20
    image = Image.fromarray(image_array)
    detector = FakeDetector(
        [
            RawDetection([(5, 8), (26, 8), (26, 18), (5, 18)]),
            RawDetection([(-20, -20), (-10, -20), (-10, -10), (-20, -10)]),
        ]
    )
    analysis, crops = analyze_image(
        image,
        detector,
        PilotConfig(feature_config=FeatureConfig()),
        image_id="synthetic",
        label="REAL",
        file_type="image/png",
    )
    assert analysis.detection_status == "ok"
    assert len(analysis.regions) == 1
    assert analysis.regions[0].region_id == "region_001"
    assert crops["region_001"].shape[:2] == (10, 21)
    assert "detected_region_count" not in analysis.model_feature_names
    assert analysis.image_features["text_region_area_ratio"].value == 21 * 10 / (40 * 32)
    json_text = json_bytes(analysis).decode("utf-8").lower()
    assert "recognized_words" not in json_text
    assert "hello" not in json_text


def test_pipeline_empty_and_failed_detection_are_distinct():
    image = Image.new("RGB", (16, 16), "white")
    empty, _ = analyze_image(image, FakeDetector([]), image_id="empty")
    assert empty.detection_status == "ok"
    assert empty.image_features["text_region_area_ratio"].value == 0
    assert empty.image_features["sharpness_laplacian_variance_mean"].status == "unavailable"
    assert "sharpness_laplacian_variance" in region_features_csv(empty).decode("utf-8").splitlines()[0]
    assert any("No visible text" in warning for warning in empty.warnings)
    failed, _ = analyze_image(image, FailingDetector(), image_id="failed")
    assert failed.detection_status == "failed"
    assert failed.image_features["text_region_area_ratio"].status == "unavailable"
    disabled, _ = analyze_image(
        image,
        FakeDetector([]),
        PilotConfig(feature_config=FeatureConfig(density_enabled=False)),
        image_id="density_disabled",
    )
    assert "text_region_area_ratio" not in disabled.image_features


def test_failed_detection_keeps_the_same_model_schema_as_success():
    image = Image.new("RGB", (16, 16), "white")
    success, _ = analyze_image(image, FakeDetector([]), image_id="success")
    failed, _ = analyze_image(image, FailingDetector(), image_id="failed")
    assert failed.model_feature_names == success.model_feature_names
    assert list(failed.model_feature_record()) == list(success.model_feature_record())
    assert all(value is None for value in failed.model_feature_record().values())


def test_rejection_diagnostics_capture_malformed_and_tiny_regions():
    image = Image.new("RGB", (20, 20), "white")
    detector = FakeDetector(
        [
            RawDetection([(1, 1), (2, 1), (2, 2), (1, 2)]),
            RawDetection([(30, 30), (31, 30), (31, 31), (30, 31)]),
            RawDetection([]),
        ]
    )
    analysis, _ = analyze_image(image, detector)
    assert not analysis.regions
    assert analysis.diagnostics["rejected_detection_count"] == 3
    assert {item["reason"] for item in analysis.diagnostics["rejected_detections"]} == {
        "too_small",
        "outside_image",
        "malformed_polygon",
    }


def test_exports_have_stable_ocr_free_schema_and_zip_artifacts():
    image = Image.new("RGB", (20, 20), "white")
    analysis, crops = analyze_image(
        image,
        FakeDetector([RawDetection([(2, 2), (10, 2), (10, 8), (2, 8)])]),
        image_id="export_case",
    )
    image_csv = image_features_csv(analysis).decode("utf-8")
    header = next(csv.reader(io.StringIO(image_csv)))
    assert header[:3] == ["image_id", "label", "sharpness_laplacian_variance_mean"]
    assert "detected_region_count" not in header
    assert "recognized_text" not in image_csv.lower()
    region_csv = region_features_csv(analysis).decode("utf-8")
    assert "region_001" in region_csv
    payload = json.loads(json_bytes(analysis))
    assert payload["model_feature_names"]
    assert "detected_region_count" not in payload["model_feature_names"]
    zip_payload = build_analysis_zip(
        analysis,
        draw_detection_overlay(image, analysis.regions),
        crops,
        config=load_pilot_config(),
    )
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        assert {
            "analysis.json",
            "image_features.csv",
            "image_diagnostics.csv",
            "region_features.csv",
            "feature_manifest.json",
            "feature_manifest.csv",
            "overlay.png",
            "detection_input.png",
        }.issubset(set(archive.namelist()))
        assert "crops/raw/region_001.png" in archive.namelist()
        assert "crops/grayscale/region_001.png" in archive.namelist()
        assert "crops/edges/region_001.png" in archive.namelist()


def test_config_drives_batch_and_resume_outputs(tmp_path):
    root = tmp_path / "dataset"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    for directory, color in (("real", "white"), ("fake", "gray")):
        for index in range(2):
            Image.new("RGB", (12, 12), color).save(root / directory / f"{index}.jpg")
    config = load_pilot_config()
    config = config.__class__(
        aggregation=config.aggregation,
        feature_config=config.feature_config,
        detection=config.detection,
        preprocessing=config.preprocessing,
        batch=config.batch.__class__(expected_real_count=2, expected_ai_generated_count=2),
    )
    items = discover_dataset(root)
    inventory = dataset_inventory(items)
    audit = audit_dataset(items, inventory, config)
    assert audit["counts"] == {"REAL": 2, "AI_GENERATED": 2}
    output = tmp_path / "batch"
    fake_detector = FakeDetector([RawDetection([(2, 2), (9, 2), (9, 8), (2, 8)])])
    first = run_batch(root, config=config, output_dir=output, detector=fake_detector, resume=True)
    assert first.total == 4 and first.succeeded == 4 and first.skipped == 0
    second = run_batch(root, config=config, output_dir=output, detector=fake_detector, resume=True)
    assert second.total == 4 and second.skipped == 4
    rows = list(csv.DictReader((output / "image_features.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    assert list(rows[0]) == ["image_id", "label", *success_names(config)]
    assert "recognized_text" not in (output / "image_features.csv").read_text(encoding="utf-8").lower()
    assert (output / "batch_artifacts.zip").is_file()


def test_feature_manifest_documents_scope_and_model_inclusion():
    manifest = feature_manifest(load_pilot_config())
    names = {entry["name"] for entry in manifest}
    assert "sharpness_laplacian_variance_mean" in names
    assert "text_region_area_ratio" in names
    assert "detected_region_count" in names
    assert all(
        {"output_type", "interpretation", "failure_behavior", "included_in_model_vector"}.issubset(entry)
        for entry in manifest
    )
    assert next(entry for entry in manifest if entry["name"] == "detected_region_count")["included_in_model_vector"] is False


def test_optional_aggregation_statistics_stay_out_of_model_vector():
    config = PilotConfig(aggregation=("mean", "std", "min"))
    image = Image.new("RGB", (20, 20), "white")
    analysis, _ = analyze_image(
        image,
        FakeDetector([RawDetection([(2, 2), (10, 2), (10, 8), (2, 8)])]),
        config=config,
        image_id="optional_stats",
    )
    assert "sharpness_laplacian_variance_min" in analysis.image_features
    assert "sharpness_laplacian_variance_min" not in analysis.model_feature_names
    assert "sharpness_laplacian_variance_min" not in image_features_csv(analysis).decode("utf-8").splitlines()[0]
    manifest = feature_manifest(config)
    minimum = next(entry for entry in manifest if entry["name"] == "sharpness_laplacian_variance_min")
    assert minimum["included_in_model_vector"] is False


def success_names(config):
    from src.pipeline import expected_model_feature_names

    return expected_model_feature_names(config)
