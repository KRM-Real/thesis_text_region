"""Parallel, resumable-friendly per-image feature extraction workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import ImageRecord
from ..features.pipeline import extract_features
from ..reproducibility import write_json
from ..text_regions.adapters import EasyOCRDetectorAdapter, FixtureTextDetector
from ..text_regions.visualization import render_overlay


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_DATA_ROOT: Path | None = None
_WORKER_TARGET: Path | None = None
_WORKER_DETECTOR: Any = None


def initialize_feature_worker(
    config: dict[str, Any],
    data_root: str,
    target: str,
    fixture_detector: bool,
    workers: int,
) -> None:
    """Initialize one detector per worker process rather than per image."""

    global _WORKER_CONFIG, _WORKER_DATA_ROOT, _WORKER_TARGET, _WORKER_DETECTOR
    # Prevent each EasyOCR/PyTorch process from trying to use every CPU thread.
    thread_count = str(max(1, 16 // max(1, int(workers))))
    import os

    os.environ.setdefault("OMP_NUM_THREADS", thread_count)
    os.environ.setdefault("MKL_NUM_THREADS", thread_count)
    try:
        import torch

        torch.set_num_threads(max(1, int(thread_count)))
    except Exception:  # pragma: no cover - depends on the installed torch build
        pass
    _WORKER_CONFIG = config
    _WORKER_DATA_ROOT = Path(data_root)
    _WORKER_TARGET = Path(target)
    _WORKER_DETECTOR = FixtureTextDetector() if fixture_detector else EasyOCRDetectorAdapter(config)


def _failed_result(record: ImageRecord, error: Exception) -> dict[str, Any]:
    return {
        "row": {
            "sample_id": record.sample_id,
            "label": record.label,
            "split": record.split or "",
            "processing_status": "failed",
            "processing_error": str(error),
        },
        "image_record": {
            "sample_id": record.sample_id,
            "label": record.label,
            "processing_status": "failed",
            "processing_error": str(error),
        },
        "region_records": [],
        "quality_flags": ["processing_failed"],
        "schemas": {},
    }


def extract_one_record(record_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract and persist artifacts for one image in the current worker."""

    if _WORKER_CONFIG is None or _WORKER_DATA_ROOT is None or _WORKER_TARGET is None or _WORKER_DETECTOR is None:
        raise RuntimeError("feature worker has not been initialized")
    record = ImageRecord(**record_payload)
    try:
        image_path = _WORKER_DATA_ROOT / record.relative_path
        with Image.open(image_path) as image:
            image.load()
            input_image = image.convert("RGB").copy()
            extraction = extract_features(image, record.sample_id, _WORKER_DETECTOR, _WORKER_CONFIG, record.label)
        localization_dir = _WORKER_TARGET / "localization" / "per_image"
        overlay_dir = _WORKER_TARGET / "localization" / "overlays"
        crop_dir = _WORKER_TARGET / "localization" / "crops"
        localization_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        localization_payload = extraction.detection.to_dict()
        localization_payload.update({"image_sha256": record.sha256, "relative_path": record.relative_path})
        render_overlay(input_image, extraction).save(overlay_dir / f"{record.sample_id}.png", format="PNG")
        image_crop_dir = crop_dir / record.sample_id
        image_crop_dir.mkdir(parents=True, exist_ok=True)
        for region_id, crop in extraction.artifacts.padded_crops.items():
            Image.fromarray(crop).save(image_crop_dir / f"{region_id}.png", format="PNG")
        for region_payload in localization_payload.get("regions", []):
            region_payload["crop_path"] = str(Path("localization") / "crops" / record.sample_id / f"{region_payload['region_id']}.png").replace("\\", "/")
        write_json(localization_dir / f"{record.sample_id}.json", localization_payload)
        region_records = []
        for region_id, features in extraction.region_features.items():
            for name, feature in sorted(features.items()):
                region_records.append(
                    {
                        "sample_id": record.sample_id,
                        "label": record.label,
                        "region_id": region_id,
                        "feature": name,
                        "group": feature.group,
                        "value": feature.value,
                        "status": feature.status,
                        "formula_id": feature.formula_id,
                        "reason": feature.reason or "",
                    }
                )
        values = dict(extraction.image_feature_record.values if extraction.image_feature_record else {})
        row = {"sample_id": record.sample_id, "label": record.label, "split": record.split or "", **values}
        image_record = extraction.image_feature_record.to_dict() if extraction.image_feature_record else {"sample_id": record.sample_id, "label": record.label, "processing_status": "failed"}
        schemas = {entry.name: entry.to_dict() for entry in extraction.feature_schema}
        return {
            "row": row,
            "image_record": image_record,
            "region_records": region_records,
            "quality_flags": list(extraction.image_feature_record.quality_flags if extraction.image_feature_record else []),
            "schemas": schemas,
        }
    except Exception as exc:  # noqa: BLE001 - one bad image becomes an auditable failed row
        return _failed_result(record, exc)
