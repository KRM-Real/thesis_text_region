"""Stable, OCR-free CSV/JSON/ZIP export helpers."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image

from src.features.base import DEFERRED_FEATURE_GROUPS, feature_registry
from src.pipeline import AnalysisArtifacts, inspection_aggregation_statistics
from src.types import ImageAnalysis, PilotConfig


def _csv_bytes(fieldnames: list[str], rows: Iterable[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def image_feature_row(analysis: ImageAnalysis) -> dict[str, object]:
    """Return one model-facing row without diagnostic columns."""

    return {
        "image_id": analysis.image_id,
        "label": analysis.label,
        **analysis.model_feature_record(),
    }


def image_features_csv(analysis: ImageAnalysis) -> bytes:
    """Export one fixed-length, visual-only image-level row."""

    names = analysis.model_feature_names
    return _csv_bytes(["image_id", "label", *names], [image_feature_row(analysis)])


def region_feature_rows(analysis: ImageAnalysis) -> list[dict[str, object]]:
    """Return region rows containing values and explicit status/reason columns."""

    configured = list(analysis.settings.get("feature_config", {}).get("enabled_features", []))
    observed = sorted({name for region in analysis.regions for name in region.features})
    feature_names = configured + [name for name in observed if name not in configured]
    rows: list[dict[str, object]] = []
    for region in analysis.regions:
        row: dict[str, object] = {
            "image_id": analysis.image_id,
            "label": analysis.label,
            "region_id": region.region_id,
            **region.bbox.to_dict(),
            "confidence": region.confidence,
            "crop_x": region.crop_bbox.x if region.crop_bbox else None,
            "crop_y": region.crop_bbox.y if region.crop_bbox else None,
            "crop_width": region.crop_bbox.width if region.crop_bbox else None,
            "crop_height": region.crop_bbox.height if region.crop_bbox else None,
        }
        for name in feature_names:
            result = region.features.get(name)
            row[name] = result.value if result is not None and result.status == "ok" else None
            row[f"{name}__status"] = result.status if result is not None else "unavailable"
            row[f"{name}__reason"] = result.reason if result is not None else "Feature was not computed."
        rows.append(row)
    return rows


def region_features_csv(analysis: ImageAnalysis) -> bytes:
    """Export region geometry, visual values, and failure diagnostics."""

    configured = list(analysis.settings.get("feature_config", {}).get("enabled_features", []))
    observed = sorted({name for region in analysis.regions for name in region.features})
    feature_names = configured + [name for name in observed if name not in configured]
    fields = [
        "image_id",
        "label",
        "region_id",
        "x",
        "y",
        "width",
        "height",
        "confidence",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
    ]
    fields.extend(column for name in feature_names for column in (name, f"{name}__status", f"{name}__reason"))
    return _csv_bytes(fields, region_feature_rows(analysis))


def image_diagnostics_row(analysis: ImageAnalysis) -> dict[str, object]:
    """Flatten non-feature pipeline state for batch auditing."""

    diagnostics = analysis.diagnostics
    return {
        "image_id": analysis.image_id,
        "label": analysis.label,
        "detection_status": analysis.detection_status,
        "detector": analysis.detector,
        "detector_status": analysis.detector_status,
        "width": analysis.width,
        "height": analysis.height,
        "channels": analysis.channels,
        "file_type": analysis.file_type,
        "source_group": diagnostics.get("source_group"),
        "design_category": diagnostics.get("design_category"),
        "conversion_pipeline": diagnostics.get("conversion_pipeline"),
        "raw_detection_count": diagnostics.get("raw_detection_count", 0),
        "accepted_detection_count": diagnostics.get("accepted_detection_count", len(analysis.regions)),
        "rejected_detection_count": diagnostics.get("rejected_detection_count", 0),
        "rejected_detections": json.dumps(diagnostics.get("rejected_detections", []), sort_keys=True),
        "valid_region_count": diagnostics.get("valid_region_count", 0),
        "failed_features": json.dumps(diagnostics.get("failed_features", [])),
        "warnings": json.dumps(analysis.warnings),
        "settings_json": json.dumps(analysis.settings, sort_keys=True),
        "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
    }


def image_diagnostics_csv(analysis: ImageAnalysis) -> bytes:
    """Export diagnostic metadata separately from the model feature row."""

    row = image_diagnostics_row(analysis)
    return _csv_bytes(list(row), [row])


def feature_manifest(config: PilotConfig | None = None) -> list[dict[str, Any]]:
    """Build metadata for source and generated image-level features."""

    config = config or PilotConfig()
    entries: list[dict[str, Any]] = []
    definitions = {item["name"]: item for item in feature_registry()}

    def with_calculation(entry: dict[str, Any]) -> dict[str, Any]:
        """Expose the formula under the plain-language ``calculation`` key too."""

        result = dict(entry)
        result.setdefault("calculation", result.get("formula"))
        return result

    for name in config.feature_config.enabled_features:
        definition = definitions.get(name)
        if definition is None:
            continue
        source = with_calculation({**definition, "included_in_model_vector": False})
        entries.append(source)
        for statistic in inspection_aggregation_statistics(config):
            entries.append(
                with_calculation({
                    **definition,
                    "name": f"{name}_{statistic}",
                    "scope": "image",
                    "formula": f"{statistic}({name} across valid regions)",
                    "description": f"{statistic} aggregation of the region-level {name} measurement.",
                    "interpretation": "Exploratory image-level summary; not an authenticity verdict.",
                    "included_in_model_vector": statistic in {"mean", "std"},
                    "output_type": "float",
                })
            )
    if config.feature_config.density_enabled and "text_region_area_ratio" in definitions:
        entries.append(with_calculation(definitions["text_region_area_ratio"]))
    entries.extend(with_calculation(dict(entry)) for entry in DEFERRED_FEATURE_GROUPS)
    entries.append(
        with_calculation({
            "name": "detected_region_count",
            "group": "density",
            "status": "DIAGNOSTIC_ONLY",
            "scope": "image",
            "description": "Number of accepted detector regions.",
            "formula": "count(accepted regions)",
            "preprocessing": "Clipped detector polygons; no semantic OCR data.",
            "expected_range": "0 to unbounded integer.",
            "failure_behavior": "Unavailable when detection fails.",
            "caveats": "Diagnostic metadata and excluded from the model feature vector.",
            "output_type": "integer",
            "interpretation": "Used to audit detector coverage and aggregation behavior.",
            "included_in_model_vector": False,
        })
    )
    return entries


def feature_manifest_json(config: PilotConfig | None = None) -> bytes:
    return json.dumps(feature_manifest(config), indent=2, allow_nan=False).encode("utf-8")


def feature_manifest_csv(config: PilotConfig | None = None) -> bytes:
    rows = feature_manifest(config)
    fields = [
        "name",
        "group",
        "status",
        "scope",
        "description",
        "formula",
        "calculation",
        "preprocessing",
        "expected_range",
        "failure_behavior",
        "interpretation",
        "caveats",
        "output_type",
        "included_in_model_vector",
    ]
    return _csv_bytes(fields, rows)


def json_bytes(analysis: ImageAnalysis) -> bytes:
    """Export strict JSON including diagnostics and registry settings."""

    return json.dumps(analysis.to_dict(), indent=2, allow_nan=False).encode("utf-8")


def _png_bytes(image: Image.Image | np.ndarray) -> bytes:
    stream = io.BytesIO()
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = Image.fromarray(image.astype(np.uint8), mode="L")
        else:
            image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    image.save(stream, format="PNG")
    return stream.getvalue()


def _artifact_mapping(crops: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None]:
    if isinstance(crops, AnalysisArtifacts):
        return crops.crops, crops.grayscale, crops.edges, crops.detection_input
    return dict(crops), {}, {}, None


def build_analysis_zip(
    analysis: ImageAnalysis,
    overlay: Image.Image,
    crops: Iterable[tuple[str, Image.Image]] | AnalysisArtifacts,
    config: PilotConfig | None = None,
) -> bytes:
    """Bundle analysis, separate diagnostics, manifest, overlay, and crops."""

    crop_map, gray_map, edge_map_values, detection_input = _artifact_mapping(crops)
    if isinstance(crops, AnalysisArtifacts):
        crop_items = crop_map.items()
    else:
        crop_items = crop_map.items()
    config = config or PilotConfig()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("analysis.json", json_bytes(analysis))
        archive.writestr("image_features.csv", image_features_csv(analysis))
        archive.writestr("image_diagnostics.csv", image_diagnostics_csv(analysis))
        archive.writestr("region_features.csv", region_features_csv(analysis))
        archive.writestr("feature_manifest.json", feature_manifest_json(config))
        archive.writestr("feature_manifest.csv", feature_manifest_csv(config))
        archive.writestr("overlay.png", _png_bytes(overlay))
        if detection_input is not None:
            archive.writestr("detection_input.png", _png_bytes(detection_input))
        for region_id, crop in crop_items:
            archive.writestr(f"crops/raw/{region_id}.png", _png_bytes(crop))
            if region_id in gray_map:
                archive.writestr(f"crops/grayscale/{region_id}.png", _png_bytes(gray_map[region_id]))
            if region_id in edge_map_values:
                archive.writestr(f"crops/edges/{region_id}.png", _png_bytes(edge_map_values[region_id]))
    return stream.getvalue()
