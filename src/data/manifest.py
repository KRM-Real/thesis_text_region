"""Dataset manifest loading, path safety, checksums, and shortcut audits."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image

from ..contracts import ImageRecord
from ..reproducibility import sha256_file


CANONICAL_LABELS = {"non_ai_generated", "ai_generated"}
ALIASES = {"real": "non_ai_generated", "non-ai-generated": "non_ai_generated", "ai-generated": "ai_generated"}
REQUIRED_FIELDS = {
    "sample_id", "relative_path", "label", "category", "source_id", "group_id",
    "provenance_type", "acquisition_type", "width", "height", "format", "sha256",
}
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
VALID_INCLUSION_STATUSES = {"included", "excluded", "review"}
VALID_SPLITS = {None, "", "train", "validation", "test"}


class ManifestError(ValueError):
    """Raised when a dataset manifest cannot safely enter the pipeline."""


def normalize_label(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_LABELS:
        raise ManifestError(f"Unknown label {value!r}; expected non_ai_generated or ai_generated")
    return normalized


def _safe_path(root: Path, relative_path: str) -> Path:
    raw = str(relative_path).replace("\\", "/")
    if not raw or raw.startswith("/") or PureWindowsPath(raw).is_absolute():
        raise ManifestError(f"relative_path must be project-relative: {relative_path!r}")
    candidate = (root / Path(raw)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"relative_path escapes the data root: {relative_path!r}") from exc
    return candidate


def _format_from_image(image: Image.Image, path: Path) -> str:
    return (image.format or path.suffix.lstrip(".") or "unknown").upper()


def load_manifest(
    manifest_path: str | Path,
    data_root: str | Path = "data",
    *,
    verify_checksums: bool = True,
    require_receipts: bool = True,
) -> list[ImageRecord]:
    path = Path(manifest_path)
    root = Path(data_root)
    if not path.exists():
        raise ManifestError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_FIELDS - fields)
        if missing:
            raise ManifestError(f"Manifest is missing required fields: {', '.join(missing)}")
        records: list[ImageRecord] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                sample_id = str(row["sample_id"]).strip()
                if not sample_id or sample_id in seen_ids:
                    raise ManifestError(f"duplicate or empty sample_id at row {line_number}")
                seen_ids.add(sample_id)
                relative_path = str(row["relative_path"]).strip().replace("\\", "/")
                image_path = _safe_path(root, relative_path)
                if not image_path.is_file():
                    raise ManifestError(f"missing image at row {line_number}: {relative_path}")
                label = normalize_label(row["label"])
                category = str(row["category"]).strip().lower()
                inclusion_status = str(row.get("inclusion_status") or "included").strip().lower()
                if inclusion_status not in VALID_INCLUSION_STATUSES:
                    raise ManifestError(f"invalid inclusion_status at row {line_number}: {inclusion_status!r}")
                if inclusion_status == "included" and require_receipts and category != "receipt":
                    raise ManifestError(f"included row {line_number} has non-receipt category {category!r}")
                if inclusion_status != "included" and not str(row.get("exclusion_reason") or "").strip():
                    raise ManifestError(f"non-included row {line_number} requires exclusion_reason")
                required_text_fields = {"source_id": row.get("source_id"), "group_id": row.get("group_id"), "provenance_type": row.get("provenance_type"), "acquisition_type": row.get("acquisition_type")}
                missing_text = [name for name, value in required_text_fields.items() if not str(value or "").strip()]
                if missing_text:
                    raise ManifestError(f"missing required provenance fields at row {line_number}: {', '.join(missing_text)}")
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    image_format = _format_from_image(image, image_path)
                if image_format not in SUPPORTED_FORMATS:
                    raise ManifestError(f"unsupported decoded image format at row {line_number}: {image_format}")
                declared_width = int(row["width"])
                declared_height = int(row["height"])
                if declared_width <= 0 or declared_height <= 0:
                    raise ManifestError(f"dimensions must be positive at row {line_number}")
                if (declared_width, declared_height) != (width, height):
                    raise ManifestError(f"dimension mismatch at row {line_number}: declared {(declared_width, declared_height)}, decoded {(width, height)}")
                checksum = sha256_file(image_path)
                if verify_checksums and str(row["sha256"]).strip().lower() != checksum:
                    raise ManifestError(f"checksum mismatch at row {line_number}: {relative_path}")
                split_value = str(row.get("split") or "").strip() or None
                if split_value not in VALID_SPLITS:
                    raise ManifestError(f"invalid split at row {line_number}: {split_value!r}")
                records.append(ImageRecord(
                    sample_id=sample_id,
                    relative_path=relative_path,
                    label=label,
                    category=category,
                    source_id=str(row["source_id"]).strip(),
                    group_id=str(row["group_id"]).strip(),
                    provenance_type=str(row["provenance_type"]).strip(),
                    generator_id=(str(row.get("generator_id") or "unknown").strip() or "unknown"),
                    acquisition_type=str(row["acquisition_type"]).strip(),
                    width=width,
                    height=height,
                    format=image_format,
                    sha256=checksum,
                    split=split_value,
                    inclusion_status=inclusion_status,
                    exclusion_reason=str(row.get("exclusion_reason") or "").strip(),
                ))
            except (ValueError, OSError, KeyError) as exc:
                if isinstance(exc, ManifestError):
                    raise
                raise ManifestError(f"invalid row {line_number}: {exc}") from exc
    if not records:
        raise ManifestError("Manifest contains no usable rows")
    return records


def _average_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((8, 8))
        values = np.asarray(gray, dtype=np.float32)
    mean = float(values.mean())
    result = 0
    for value in values.ravel():
        result = (result << 1) | int(value >= mean)
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def audit_manifest(records: list[ImageRecord], data_root: str | Path = "data", near_duplicate_distance: int = 4) -> dict[str, Any]:
    root = Path(data_root)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, int] = {}
    for record in records:
        by_hash[record.sha256].append(record.sample_id)
        by_group[record.group_id].append(record.sample_id)
        hashes[record.sample_id] = _average_hash(_safe_path(root, record.relative_path))
    near_pairs: list[dict[str, Any]] = []
    ids = sorted(hashes)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            distance = _hamming(hashes[left_id], hashes[right_id])
            if distance <= near_duplicate_distance and left_id not in {right_id}:
                near_pairs.append({"left_sample_id": left_id, "right_sample_id": right_id, "hamming_distance": distance})
    return {
        "sample_count": len(records),
        "label_counts": dict(Counter(record.label for record in records)),
        "category_counts": dict(Counter(record.category for record in records)),
        "format_counts": dict(Counter(record.format for record in records)),
        "dimension_counts": dict(Counter(f"{record.width}x{record.height}" for record in records)),
        "exact_duplicate_groups": [ids for ids in by_hash.values() if len(ids) > 1],
        "group_counts": {group: len(ids) for group, ids in by_group.items()},
        "near_duplicate_candidates": near_pairs,
        "shortcut_diagnostics": {
            "label_by_format": _label_cross_tab(records, lambda record: record.format),
            "label_by_dimensions": _label_cross_tab(records, lambda record: f"{record.width}x{record.height}"),
            "label_by_acquisition": _label_cross_tab(records, lambda record: record.acquisition_type),
            "label_by_source": _label_cross_tab(records, lambda record: record.source_id),
        },
    }


def _label_cross_tab(records: list[ImageRecord], key_fn) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        result[str(key_fn(record))][record.label or "unknown"] += 1
    return {key: dict(value) for key, value in result.items()}


def write_manifest(path: str | Path, records: list[ImageRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "relative_path", "label", "category", "source_id", "group_id", "provenance_type", "generator_id", "acquisition_type", "width", "height", "format", "sha256", "split", "inclusion_status", "exclusion_reason"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def write_audit(path: str | Path, audit: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
