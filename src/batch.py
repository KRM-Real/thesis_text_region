"""Resumable, OCR-free batch feature extraction for the local pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from src.config import config_to_yaml, load_pilot_config
from src.detection.detector import EasyOCRTextDetector
from src.export.feature_export import (
    feature_manifest,
    feature_manifest_csv,
    feature_manifest_json,
    image_diagnostics_row,
    image_feature_row,
    region_feature_rows,
    json_bytes,
)
from src.pipeline import AnalysisArtifacts, analyze_image, expected_model_feature_names, make_unavailable_analysis
from src.preprocessing.image_ops import ImageInputError, load_image
from src.types import ImageAnalysis, PilotConfig
from src.utils.logging import configure_logging, log_analysis_summary
from src.visualization.overlays import draw_detection_overlay


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LABEL_ALIASES = {
    "REAL": "REAL",
    "AI": "AI_GENERATED",
    "FAKE": "AI_GENERATED",
    "AI_GENERATED": "AI_GENERATED",
    "AI-GENERATED": "AI_GENERATED",
}


@dataclass(frozen=True)
class BatchItem:
    """One explicitly labeled image and optional non-feature provenance fields."""

    path: Path
    label: str
    image_id: str
    source_group: str | None = None
    design_category: str | None = None
    conversion_pipeline: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": str(self.path),
            "label": self.label,
            "image_id": self.image_id,
            "source_group": self.source_group,
            "design_category": self.design_category,
            "conversion_pipeline": self.conversion_pipeline,
        }


@dataclass
class BatchSummary:
    """Counts and artifact paths returned by :func:`run_batch`."""

    output_dir: str
    total: int
    succeeded: int
    failed: int
    skipped: int
    image_features_path: str
    region_features_path: str
    diagnostics_path: str
    audit_path: str
    manifest_path: str
    warnings: list[str] = field(default_factory=list)
    artifact_zip_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def normalize_label(value: str) -> str:
    """Normalize accepted aliases to the two thesis labels."""

    normalized = LABEL_ALIASES.get(str(value).strip().upper())
    if normalized is None:
        raise ValueError("labels must be REAL or AI_GENERATED")
    return normalized


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "image"


def _unique_image_id(label: str, path: Path, seen: set[str]) -> str:
    base = _slug(f"{label.lower()}__{path.stem}")
    candidate = base
    index = 2
    while candidate in seen:
        candidate = f"{base}_{index}"
        index += 1
    seen.add(candidate)
    return candidate


def discover_dataset(
    dataset_root: str | Path,
    real_dir: str = "real",
    ai_dir: str = "fake",
) -> list[BatchItem]:
    """Discover supported images from the two configured class directories."""

    root = Path(dataset_root).resolve()
    items: list[BatchItem] = []
    seen: set[str] = set()
    for label, directory in (("REAL", real_dir), ("AI_GENERATED", ai_dir)):
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Dataset class directory does not exist: {folder}")
        for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append(BatchItem(path, label, _unique_image_id(label, path, seen)))
    if not items:
        raise ValueError(f"No supported images found below {root}")
    return items


def read_manifest(manifest_path: str | Path) -> list[BatchItem]:
    """Read ``image_path,label`` and optional diagnostic provenance columns."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ValueError(f"Manifest does not exist: {path}")
    items: list[BatchItem] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Manifest must contain image_path and label columns")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            if not raw_path:
                raise ValueError(f"Manifest row {row_number} has an empty image_path")
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            label = normalize_label(row.get("label", ""))
            image_id = _slug(row.get("image_id", "")) if row.get("image_id") else _unique_image_id(label, image_path, seen)
            if image_id in seen:
                image_id = _unique_image_id(label, image_path, seen)
            else:
                seen.add(image_id)
            items.append(
                BatchItem(
                    image_path,
                    label,
                    image_id,
                    row.get("source_group") or None,
                    row.get("design_category") or None,
                    row.get("conversion_pipeline") or None,
                )
            )
    if not items:
        raise ValueError(f"Manifest contains no rows: {path}")
    return items


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def dataset_inventory(items: Iterable[BatchItem]) -> list[dict[str, object]]:
    """Collect read-only file/image metadata for dataset confound auditing."""

    inventory: list[dict[str, object]] = []
    for item in items:
        record: dict[str, object] = {
            **item.to_dict(),
            "extension": item.path.suffix.lower(),
            "file_size_bytes": None,
            "sha256": _sha256(item.path),
            "actual_format": None,
            "compression_signature": None,
            "mode": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "channels": None,
            "has_exif": None,
            "decode_status": "failed",
            "decode_error": None,
        }
        try:
            record["file_size_bytes"] = item.path.stat().st_size
            with Image.open(item.path) as image:
                record["actual_format"] = image.format
                quantization = getattr(image, "quantization", None)
                if quantization:
                    record["compression_signature"] = hashlib.sha256(
                        json.dumps(quantization, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()[:16]
                record["mode"] = image.mode
                record["width"], record["height"] = image.size
                if image.height:
                    record["aspect_ratio"] = round(float(image.width) / float(image.height), 6)
                record["channels"] = len(image.getbands())
                record["has_exif"] = bool(getattr(image, "getexif", lambda: {})())
                image.verify()
            record["decode_status"] = "ok"
        except Exception as exc:
            record["decode_error"] = str(exc)
        inventory.append(record)
    return inventory


def _class_value_sets(inventory: list[dict[str, object]], field: str) -> dict[str, set[object]]:
    result: dict[str, set[object]] = defaultdict(set)
    for record in inventory:
        value = record.get(field)
        if value is not None:
            result[str(record["label"])].add(value)
    return result


def _class_numeric_summaries(inventory: list[dict[str, object]], fields: tuple[str, ...]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize numeric metadata without using it as a model input."""

    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for metadata_name in fields:
        by_label: dict[str, dict[str, float | int]] = {}
        for label in ("REAL", "AI_GENERATED"):
            values = [
                float(record[metadata_name])
                for record in inventory
                if record.get("label") == label and isinstance(record.get(metadata_name), (int, float))
            ]
            if values:
                array = np.asarray(values, dtype=float)
                by_label[label] = {
                    "count": int(array.size),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "mean": float(np.mean(array)),
                    "median": float(np.median(array)),
                }
        summaries[metadata_name] = by_label
    return summaries


def audit_dataset(
    items: list[BatchItem], inventory: list[dict[str, object]], config: PilotConfig
) -> dict[str, object]:
    """Summarize class balance and possible preprocessing/source confounds."""

    counts = Counter(item.label for item in items)
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for record in inventory:
        digest = record.get("sha256")
        if digest:
            hash_groups[str(digest)].append(str(record["image_id"]))
    duplicates = [ids for ids in hash_groups.values() if len(ids) > 1]
    warnings: list[str] = []
    expected = {
        "REAL": config.batch.expected_real_count,
        "AI_GENERATED": config.batch.expected_ai_generated_count,
    }
    for label, target in expected.items():
        if counts.get(label, 0) != target:
            warnings.append(f"Expected {target} {label} images but found {counts.get(label, 0)}.")
    if duplicates:
        warnings.append(f"Found {len(duplicates)} exact duplicate hash group(s).")
    for metadata_field in (
        "actual_format",
        "compression_signature",
        "mode",
        "width",
        "height",
        "aspect_ratio",
        "channels",
        "has_exif",
    ):
        value_sets = _class_value_sets(inventory, metadata_field)
        if len(value_sets) == 2 and value_sets["REAL"] and value_sets["AI_GENERATED"] and not (
            value_sets["REAL"] & value_sets["AI_GENERATED"]
        ):
            warnings.append(f"{metadata_field} values are class-separated; investigate possible dataset leakage.")
    if any(record.get("decode_status") != "ok" for record in inventory):
        warnings.append("One or more files failed metadata decoding and will be retained as batch failures.")
    return {
        "schema_version": "1.0",
        "counts": dict(counts),
        "expected_counts": expected,
        "inventory_count": len(inventory),
        "duplicate_hash_groups": duplicates,
        "distributions": {
            field: {label: sorted(str(value) for value in values) for label, values in _class_value_sets(inventory, field).items()}
            for field in (
                "actual_format",
                "compression_signature",
                "mode",
                "width",
                "height",
                "aspect_ratio",
                "channels",
                "has_exif",
            )
        },
        "numeric_summaries": _class_numeric_summaries(
            inventory, ("file_size_bytes", "width", "height", "aspect_ratio")
        ),
        "warnings": warnings,
        "notes": [
            "Folder names, filenames, formats, sizes, file bytes, EXIF, and source metadata are diagnostics only and are excluded from the model vector.",
            "JPEG conversion does not prove that prior compression artifacts were equal across classes.",
        ],
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _write_batch_zip(output: Path) -> Path:
    """Bundle consolidated files and per-image artifacts into one ZIP."""

    archive_path = output / "batch_artifacts.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path == archive_path:
                continue
            archive.write(path, path.relative_to(output).as_posix())
    return archive_path


def _region_fields(config: PilotConfig) -> list[str]:
    fields = [
        "image_id", "label", "region_id", "x", "y", "width", "height", "confidence",
        "crop_x", "crop_y", "crop_width", "crop_height",
    ]
    for name in config.feature_config.enabled_features:
        fields.extend((name, f"{name}__status", f"{name}__reason"))
    return fields


def _save_artifacts(
    image_dir: Path,
    analysis: ImageAnalysis,
    artifacts: AnalysisArtifacts,
    config: PilotConfig,
    image: Image.Image | None,
) -> None:
    from src.export.feature_export import _png_bytes  # local import keeps public API focused

    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "analysis.json").write_bytes(json_bytes(analysis))
    if image is not None:
        (image_dir / "overlay.png").write_bytes(_png_bytes(draw_detection_overlay(image, analysis.regions)))
    if artifacts.detection_input is not None:
        (image_dir / "detection_input.png").write_bytes(_png_bytes(artifacts.detection_input))
    for region_id, crop in artifacts.crops.items():
        for folder, values in (
            ("crops/raw", artifacts.crops),
            ("crops/grayscale", artifacts.grayscale),
            ("crops/edges", artifacts.edges),
        ):
            value = values.get(region_id)
            if value is None:
                continue
            target = image_dir / folder
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{region_id}.png").write_bytes(_png_bytes(value))


def _load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if record.get("image_id"):
                    records[str(record["image_id"])] = record
            except json.JSONDecodeError:
                continue
    return records


def _config_fingerprint(config: PilotConfig) -> str:
    """Hash reproducibility-affecting settings while tolerating old run files."""

    payload = config.to_dict()
    # Language/GPU/mode are construction metadata rather than extraction
    # parameters. Omitting them keeps checkpoints made immediately before a
    # metadata-only schema extension safely resumable.
    detector = payload.get("detector")
    if isinstance(detector, dict):
        for key in ("languages", "gpu", "mode"):
            detector.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _post_extraction_audit(
    analyses: list[dict[str, object]], image_rows: list[dict[str, object]], config: PilotConfig
) -> dict[str, object]:
    statuses = Counter(str(row.get("detection_status")) for row in analyses)
    failed_features: Counter[str] = Counter()
    for row in analyses:
        raw_failed = row.get("failed_features", [])
        if isinstance(raw_failed, str):
            try:
                raw_failed = json.loads(raw_failed)
            except json.JSONDecodeError:
                raw_failed = []
        for name in raw_failed:
            failed_features[str(name)] += 1
    density_values = [
        float(row["text_region_area_ratio"])
        for row in image_rows
        if row.get("text_region_area_ratio") not in (None, "")
    ]
    terciles: dict[str, float] = {}
    density_bins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if density_values:
        q1, q2 = np.quantile(np.asarray(density_values, dtype=float), [1 / 3, 2 / 3])
        terciles = {"low_upper": float(q1), "medium_upper": float(q2)}
        for row in image_rows:
            value = row.get("text_region_area_ratio")
            if value in (None, ""):
                bucket = "unavailable"
            elif float(value) <= q1:
                bucket = "low"
            elif float(value) <= q2:
                bucket = "medium"
            else:
                bucket = "high"
            density_bins[str(row.get("label") or "unlabeled")][bucket] += 1
    return {
        "detection_status_counts": dict(statuses),
        "failed_feature_counts": dict(failed_features),
        "region_count_distribution": {
            label: Counter(
                int(row.get("accepted_detection_count", 0))
                for row in analyses
                if row.get("label") == label
            )
            for label in ("REAL", "AI_GENERATED")
        },
        "text_density_terciles": terciles,
        "text_density_bins_by_class": {label: dict(values) for label, values in density_bins.items()},
        "model_feature_names": expected_model_feature_names(config),
    }


def run_batch(
    source: str | Path | list[BatchItem],
    config: PilotConfig | None = None,
    output_dir: str | Path = "outputs/batches/pilot_600",
    resume: bool = True,
    detector: Any | None = None,
    real_dir: str = "real",
    ai_dir: str = "fake",
    progress_callback: Callable[[int, int, BatchItem, ImageAnalysis], None] | None = None,
) -> BatchSummary:
    """Extract every image sequentially and checkpoint after each image."""

    config = config or PilotConfig()
    if isinstance(source, list):
        items = source
    elif Path(source).suffix.lower() == ".csv":
        items = read_manifest(source)
    else:
        items = discover_dataset(source, real_dir=real_dir, ai_dir=ai_dir)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_logging(output / "batch.log")
    metadata_path = output / "run_metadata.json"
    fingerprint = _config_fingerprint(config)
    if resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Cannot resume: output directory was created with a different configuration.")
    metadata_path.write_text(
        json.dumps({"config_fingerprint": fingerprint, "items": [item.to_dict() for item in items]}, indent=2),
        encoding="utf-8",
    )
    inventory = dataset_inventory(items)
    preflight = audit_dataset(items, inventory, config)
    _write_csv(output / "dataset_inventory.csv", list(inventory[0]) if inventory else [], inventory)
    (output / "feature_manifest.json").write_bytes(feature_manifest_json(config))
    (output / "feature_manifest.csv").write_bytes(feature_manifest_csv(config))
    (output / "resolved_config.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
    checkpoint_path = output / "checkpoint.jsonl"
    if not resume and checkpoint_path.exists():
        checkpoint_path.write_text("", encoding="utf-8")
    completed = _load_checkpoint(checkpoint_path) if resume else {}
    image_records: list[dict[str, object]] = []
    region_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    analysis_diagnostics: list[dict[str, object]] = []
    skipped = 0
    detector_instance = detector
    if detector_instance is None:
        model_dir = Path(__file__).resolve().parents[1] / "easyocr-models"
        detector_instance = EasyOCRTextDetector(
            languages=config.detection.languages,
            gpu=config.detection.gpu,
            model_storage_directory=str(model_dir),
        )
    for index, item in enumerate(items, start=1):
        if item.image_id in completed:
            record = completed[item.image_id]
            image_records.append(record["image_row"])
            region_records.extend(record.get("region_rows", []))
            diagnostic_records.append(record["diagnostics_row"])
            analysis_diagnostics.append(record["diagnostics_row"])
            skipped += 1
            continue
        image: Image.Image | None = None
        try:
            with Image.open(item.path) as source_image:
                file_type = source_image.format or item.path.suffix.lstrip(".").upper()
            image = load_image(item.path)
            analysis, artifacts = analyze_image(
                image,
                detector_instance,
                config=config,
                image_id=item.image_id,
                label=item.label,
                file_type=file_type,
            )
        except (ImageInputError, OSError, ValueError) as exc:
            analysis = make_unavailable_analysis(
                item.image_id,
                config,
                f"Image could not be loaded: {exc}",
                label=item.label,
                file_type=item.path.suffix.lstrip(".").upper() or None,
                detector=detector_instance,
            )
            artifacts = AnalysisArtifacts()
        # Manifest provenance is retained for auditing only and is never
        # copied into the model-facing feature row.
        analysis.diagnostics.update(
            {
                "source_group": item.source_group,
                "design_category": item.design_category,
                "conversion_pipeline": item.conversion_pipeline,
            }
        )
        if config.batch.save_artifacts == "all" or (
            config.batch.save_artifacts == "failures" and analysis.detection_status != "ok"
        ):
            _save_artifacts(output / "images" / item.image_id, analysis, artifacts, config, image)
        image_row = image_feature_row(analysis)
        region_rows = region_feature_rows(analysis)
        diagnostics_row = image_diagnostics_row(analysis)
        record = {
            "image_id": item.image_id,
            "image_row": image_row,
            "region_rows": region_rows,
            "diagnostics_row": diagnostics_row,
            "failed_features": analysis.diagnostics.get("failed_features", []),
        }
        with checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        image_records.append(image_row)
        region_records.extend(region_rows)
        diagnostic_records.append(diagnostics_row)
        analysis_diagnostics.append(diagnostics_row)
        if analysis.detection_status != "ok":
            log_analysis_summary(analysis)
        if progress_callback:
            progress_callback(index, len(items), item, analysis)

    model_fields = ["image_id", "label", *expected_model_feature_names(config)]
    _write_csv(output / "image_features.csv", model_fields, image_records)
    _write_csv(output / "region_features.csv", _region_fields(config), region_records)
    diagnostic_fields = list(image_diagnostics_row(make_unavailable_analysis("schema", config, "schema")))
    _write_csv(output / "image_diagnostics.csv", diagnostic_fields, diagnostic_records)
    postflight = _post_extraction_audit(analysis_diagnostics, image_records, config)
    audit = {"preflight": preflight, "post_extraction": postflight}
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (output / "batch_manifest.json").write_text(
        json.dumps({"items": [item.to_dict() for item in items], "feature_manifest": feature_manifest(config)}, indent=2),
        encoding="utf-8",
    )
    summary = BatchSummary(
        output_dir=str(output),
        total=len(items),
        succeeded=sum(1 for row in diagnostic_records if row.get("detection_status") == "ok"),
        failed=sum(1 for row in diagnostic_records if row.get("detection_status") != "ok"),
        skipped=skipped,
        image_features_path=str(output / "image_features.csv"),
        region_features_path=str(output / "region_features.csv"),
        diagnostics_path=str(output / "image_diagnostics.csv"),
        audit_path=str(output / "dataset_audit.json"),
        manifest_path=str(output / "feature_manifest.json"),
        warnings=list(preflight.get("warnings", [])),
        artifact_zip_path=str(output / "batch_artifacts.zip"),
    )
    # Write the summary before archiving so one ZIP pass can include it.
    (output / "batch_summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    try:
        _write_batch_zip(output)
    except OSError as exc:
        summary.artifact_zip_path = None
        summary.warnings.append(f"Batch ZIP creation failed: {exc}")
        (output / "batch_summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Visual Text-Region Feature Extraction Pilot in batch mode.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", help="Root containing real/ and fake/ directories")
    source.add_argument("--manifest", help="CSV with image_path,label columns")
    parser.add_argument("--real-dir", default="real")
    parser.add_argument("--ai-dir", default="fake")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output-dir", default="outputs/batches/pilot_600")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_pilot_config(args.config)
    source = args.manifest or args.dataset_root
    summary = run_batch(
        source,
        config=config,
        output_dir=args.output_dir,
        resume=args.resume,
        real_dir=args.real_dir,
        ai_dir=args.ai_dir,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
