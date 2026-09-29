"""Prepare a user-supplied receipt dataset without changing the raw files.

This module is deliberately separate from model training.  It creates a
normalized image copy, a manifest with source and transformed-file hashes, a
shortcut/duplicate audit, and a deterministic balanced manifest.  Filenames
and extensions are retained only as provenance diagnostics; they are never
feature inputs.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts import ImageRecord
from ..reproducibility import sha256_file, write_json
from .manifest import write_manifest


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROCESSING_SCHEMA_VERSION = "dataset-processing-1"


_DCT_BASIS: np.ndarray | None = None


def _perceptual_hash(image: Image.Image) -> int:
    """Return a DCT-based perceptual hash for duplicate review.

    A plain average hash over receipt images is overly sensitive to their
    shared white-paper background.  The low-frequency DCT signature is still
    only a review aid, but it is less likely to group unrelated receipts that
    merely share a page layout.
    """

    global _DCT_BASIS
    if _DCT_BASIS is None:
        size = 32
        indices = np.arange(size, dtype=np.float32)
        basis = np.cos(np.pi * (2.0 * indices[:, None] + 1.0) * indices[None, :] / (2.0 * size))
        basis[:, 0] *= 1.0 / np.sqrt(size)
        basis[:, 1:] *= np.sqrt(2.0 / size)
        _DCT_BASIS = basis
    gray = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.float32) / 255.0
    coefficients = _DCT_BASIS.T @ values @ _DCT_BASIS
    low_frequency = coefficients[:8, :8].ravel()
    median = float(np.median(low_frequency[1:]))
    result = 0
    for value in low_frequency:
        result = (result << 1) | int(value >= median)
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _safe_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _sample_id(label: str, source_sha256: str, occurrence: int) -> str:
    suffix = "" if occurrence == 0 else f"-dup{occurrence:02d}"
    return f"textfake-{label}-{source_sha256[:16]}{suffix}"


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable extended CSV while preserving fields unknown to ImageRecord."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_rows(
    input_root: Path,
    output_relative_root: Path,
    project_root: Path,
    label_dirs: dict[str, str],
    *,
    source_id: str,
    category: str,
    target_width: int,
    target_height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_hash_occurrences: Counter[str] = Counter()
    decode_failures: list[dict[str, str]] = []
    seen_files: set[Path] = set()

    for label, directory_name in label_dirs.items():
        class_root = input_root / directory_name
        if not class_root.is_dir():
            raise ValueError(f"label directory does not exist: {class_root}")
        for source_path in sorted(class_root.rglob("*"), key=lambda path: str(path).lower()):
            if not source_path.is_file():
                continue
            seen_files.add(source_path.resolve())
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append({"path": _safe_relative(source_path, project_root), "reason": "unsupported_or_non_image_file"})
                continue
            source_relative = _safe_relative(source_path, project_root)
            try:
                source_sha = sha256_file(source_path)
                with Image.open(source_path) as opened:
                    opened.load()
                    source_format = (opened.format or source_path.suffix.lstrip(".") or "unknown").upper()
                    source_mode = opened.mode
                    source_width, source_height = opened.size
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    source_hash_value = _perceptual_hash(image)
                    transformed = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    occurrence = source_hash_occurrences[source_sha]
                    source_hash_occurrences[source_sha] += 1
                    sample_id = _sample_id(label, source_sha, occurrence)
                    relative_output = output_relative_root / label / f"{sample_id}.png"
                    output_path = project_root / "data" / output_relative_root / label / f"{sample_id}.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if output_path.is_file():
                        try:
                            with Image.open(output_path) as existing:
                                existing.load()
                                reusable = existing.format == "PNG" and existing.mode == "RGB" and existing.size == (target_width, target_height)
                        except Exception:  # noqa: BLE001 - a bad prior artifact is replaced below
                            reusable = False
                    else:
                        reusable = False
                    if not reusable:
                        transformed.save(output_path, format="PNG", compress_level=6)
                    output_sha = sha256_file(output_path)
            except Exception as exc:  # noqa: BLE001 - source audit must retain the exact failure reason
                decode_failures.append({"path": source_relative, "reason": str(exc)})
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "relative_path": str(relative_output).replace("\\", "/"),
                    "label": label,
                    "category": category,
                    "source_id": source_id,
                    "group_id": f"source-{source_sha[:16]}",
                    "provenance_type": "user_supplied_textfake_dataset",
                    "generator_id": "unknown" if label == "ai_generated" else "not_applicable",
                    "acquisition_type": "unknown",
                    "width": target_width,
                    "height": target_height,
                    "format": "PNG",
                    "sha256": output_sha,
                    "split": "",
                    "inclusion_status": "included",
                    "exclusion_reason": "",
                    "source_relative_path": source_relative,
                    "source_sha256": source_sha,
                    "source_width": source_width,
                    "source_height": source_height,
                    "source_format": source_format,
                    "source_mode": source_mode,
                    "source_bytes": source_path.stat().st_size,
                    "normalized_width": target_width,
                    "normalized_height": target_height,
                    "normalization_policy": f"RGB+fixed_canvas_{target_width}x{target_height}+PNG",
                    "source_perceptual_hash": str(source_hash_value),
                    "balance_status": "eligible",
                }
            )

    for extra_path in sorted(input_root.rglob("*"), key=lambda path: str(path).lower()):
        if extra_path.is_file() and extra_path.resolve() not in seen_files:
            skipped.append({"path": _safe_relative(extra_path, project_root), "reason": "outside_explicit_label_directories"})

    audit = {
        "source_file_count": len(rows) + len(decode_failures) + len(skipped),
        "decoded_image_count": len(rows),
        "skipped_file_count": len(skipped),
        "decode_failure_count": len(decode_failures),
        "skipped_files": skipped,
        "decode_failures": decode_failures,
        "source_label_counts": dict(Counter(row["label"] for row in rows)),
        "source_format_counts": dict(Counter(row["source_format"] for row in rows)),
        "source_mode_counts": dict(Counter(row["source_mode"] for row in rows)),
        "source_dimension_counts_by_label": {
            label: dict(Counter(f"{row['source_width']}x{row['source_height']}" for row in rows if row["label"] == label))
            for label in sorted(label_dirs)
        },
        "normalization": {
            "output_format": "PNG",
            "output_color_mode": "RGB",
            "target_width": target_width,
            "target_height": target_height,
            "resampling": "LANCZOS",
            "exif_orientation_applied": True,
            "metadata_policy": "no source metadata copied to normalized PNG",
        },
    }
    return rows, skipped, audit


def _mark_duplicate_reviews(rows: list[dict[str, Any]], near_duplicate_distance: int) -> dict[str, Any]:
    """Mark exact duplicates and perceptual-hash candidates before balancing."""

    by_source_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source_hash[str(row["source_sha256"])].append(row)

    exact_groups: list[list[str]] = []
    exact_cross_label: list[list[str]] = []
    for grouped in by_source_hash.values():
        if len(grouped) <= 1:
            continue
        ids = sorted(str(row["sample_id"]) for row in grouped)
        exact_groups.append(ids)
        labels = {str(row["label"]) for row in grouped}
        if len(labels) > 1:
            exact_cross_label.append(ids)
            for row in grouped:
                row["inclusion_status"] = "review"
                row["exclusion_reason"] = "exact_duplicate_crosses_labels"
                row["balance_status"] = "review"
        else:
            keep = min(grouped, key=lambda row: str(row["source_relative_path"]))
            for row in grouped:
                if row is not keep:
                    row["inclusion_status"] = "excluded"
                    row["exclusion_reason"] = "exact_duplicate_redundant"
                    row["balance_status"] = "excluded_duplicate"

    hash_values = {str(row["sample_id"]): int(row["source_perceptual_hash"]) for row in rows}
    labels = {str(row["sample_id"]): str(row["label"]) for row in rows}
    ids = sorted(hash_values)
    near_pairs: list[dict[str, Any]] = []
    near_ids: set[str] = set()
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            distance = _hamming(hash_values[left_id], hash_values[right_id])
            if distance <= near_duplicate_distance and left_id != right_id:
                near_pairs.append(
                    {
                        "left_sample_id": left_id,
                        "right_sample_id": right_id,
                        "left_label": labels[left_id],
                        "right_label": labels[right_id],
                        "hamming_distance": distance,
                    }
                )
                near_ids.update((left_id, right_id))

    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in near_ids:
            continue
        if row["inclusion_status"] == "included":
            row["inclusion_status"] = "review"
            row["exclusion_reason"] = "perceptual_near_duplicate_candidate_requires_review"
            row["balance_status"] = "review"
        row["near_duplicate_candidate"] = "true"

    return {
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_cross_label_groups": exact_cross_label,
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_candidate_count": len(near_pairs),
        "near_duplicate_sample_count": len(near_ids),
        "near_duplicate_candidates": near_pairs,
    }


def _select_balanced(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["inclusion_status"] == "included":
            eligible_by_label[str(row["label"])].append(row)
    labels = sorted(eligible_by_label)
    if labels != ["ai_generated", "non_ai_generated"]:
        raise ValueError(f"both canonical labels are required after audit; found {labels}")
    counts = {label: len(eligible_by_label[label]) for label in labels}
    target = min(counts.values())
    if target == 0:
        raise ValueError("no eligible samples remain for at least one label")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for label in labels:
        candidates = sorted(eligible_by_label[label], key=lambda row: str(row["sample_id"]))
        rng = random.Random(seed + (1 if label == "ai_generated" else 0))
        rng.shuffle(candidates)
        chosen_ids = {str(row["sample_id"]) for row in candidates[:target]}
        for row in candidates:
            if str(row["sample_id"]) in chosen_ids:
                row["balance_status"] = "selected"
                selected.append(row)
            else:
                row["balance_status"] = "excluded_majority_for_balance"
                row["exclusion_reason"] = "deterministic_downsample_to_equal_class_count"
                excluded.append(row)
    selected.sort(key=lambda row: str(row["sample_id"]))
    excluded.sort(key=lambda row: str(row["sample_id"]))
    return selected, excluded, {
        "eligible_counts_before_balance": counts,
        "selected_counts": dict(Counter(str(row["label"]) for row in selected)),
        "balance_strategy": "deterministic_downsample_majority_without_oversampling",
        "seed": seed,
    }


def prepare_dataset(
    input_root: str | Path,
    output_root: str | Path = "data/processed/textfake_receipts_v1",
    manifest_root: str | Path = "data/manifests",
    *,
    real_dir: str = "0_real",
    ai_dir: str = "1_fake",
    source_id: str = "textfake",
    category: str = "receipt",
    target_width: int = 1024,
    target_height: int = 1536,
    seed: int = 42,
    near_duplicate_distance: int = 0,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Create normalized copies and a deterministic balanced manifest.

    The source tree is only read.  Normalized images are written to
    ``output_root`` and manifests/reports are written below ``manifest_root``.
    The balanced manifest references normalized images and is the only manifest
    intended for later split creation or training.
    """

    if target_width < 16 or target_height < 16:
        raise ValueError("target dimensions must both be at least 16 pixels")
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance must be between 0 and 64")
    root = Path(project_root).resolve()
    input_path = (root / input_root).resolve()
    output_path = (root / output_root).resolve()
    manifest_path = (root / manifest_root).resolve()
    if not input_path.is_dir():
        raise ValueError(f"input dataset directory does not exist: {input_path}")
    if output_path == input_path or input_path in output_path.parents:
        raise ValueError("output_root must not be inside the raw input dataset")
    data_root = (root / "data").resolve()
    try:
        output_relative_root = output_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("output_root must be inside the project data directory") from exc
    output_path.mkdir(parents=True, exist_ok=True)

    labels = {"non_ai_generated": real_dir, "ai_generated": ai_dir}
    rows, skipped, source_audit = _source_rows(
        input_path,
        output_relative_root,
        root,
        labels,
        source_id=source_id,
        category=category,
        target_width=target_width,
        target_height=target_height,
    )
    if not rows:
        raise ValueError("no supported, decodable images were found")

    duplicate_audit = _mark_duplicate_reviews(rows, near_duplicate_distance)
    selected, balance_exclusions, balance_audit = _select_balanced(rows, seed)

    for row in rows:
        row["source_root"] = _safe_relative(input_path, root)
    for row in selected:
        row["split"] = ""

    source_manifest_path = manifest_path / "textfake_source_manifest.csv"
    audited_manifest_path = manifest_path / "textfake_audited_manifest.csv"
    balanced_manifest_path = manifest_path / "textfake_balanced_manifest.csv"
    balance_exclusions_path = manifest_path / "textfake_balance_exclusions.csv"
    _write_rows(source_manifest_path, rows)
    _write_rows(audited_manifest_path, rows)
    _write_rows(balance_exclusions_path, balance_exclusions)
    write_manifest(balanced_manifest_path, [ImageRecord(**{key: row[key] for key in ImageRecord.__dataclass_fields__}) for row in selected])

    audit = {
        "schema_version": PROCESSING_SCHEMA_VERSION,
        "input_root": _safe_relative(input_path, root),
        "output_root": _safe_relative(output_path, root),
        "source_id": source_id,
        "category": category,
        "label_directories": labels,
        "source_audit": source_audit,
        "duplicate_audit": duplicate_audit,
        "balance_audit": balance_audit,
        "selected_count": len(selected),
        "balance_exclusion_count": len(balance_exclusions),
        "review_count": sum(row["inclusion_status"] == "review" for row in rows),
        "excluded_count": sum(row["inclusion_status"] == "excluded" for row in rows),
        "skipped_files": skipped,
        "warnings": [
            "Acquisition type is recorded as unknown because it was not supplied in the input metadata.",
            "AI generator identity is recorded as unknown because it was not supplied in the input metadata.",
            "Perceptual-hash candidates are marked review and excluded from the balanced manifest until reviewed.",
            "Fixed-canvas resizing can affect sharpness, stroke, and texture measurements; this is a documented prototype decision.",
        ],
    }
    audit_path = manifest_path / "textfake_dataset_audit.json"
    shortcut_path = manifest_path / "textfake_shortcut_audit.json"
    duplicate_path = manifest_path / "textfake_duplicate_review.csv"
    write_json(audit_path, audit)
    write_json(
        shortcut_path,
        {
            "source_label_by_format": {
                fmt: dict(Counter(row["label"] for row in rows if row["source_format"] == fmt))
                for fmt in sorted({str(row["source_format"]) for row in rows})
            },
            "source_label_by_dimensions": {
                dimension: dict(Counter(row["label"] for row in rows if f"{row['source_width']}x{row['source_height']}" == dimension))
                for dimension in sorted({f"{row['source_width']}x{row['source_height']}" for row in rows})
            },
            "normalized_label_by_format": {"PNG": dict(Counter(row["label"] for row in selected))},
            "normalized_label_by_dimensions": {f"{target_width}x{target_height}": dict(Counter(row["label"] for row in selected))},
            "model_feature_policy": "file names, extensions, paths, source bytes, and metadata are excluded from the feature matrix",
        },
    )
    duplicate_rows = duplicate_audit["near_duplicate_candidates"]
    _write_rows(duplicate_path, duplicate_rows)

    manifest_hashes = {
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "audited_manifest_sha256": sha256_file(audited_manifest_path),
        "balanced_manifest_sha256": sha256_file(balanced_manifest_path),
    }
    (manifest_path / "textfake_manifest_sha256.json").write_text(json.dumps(manifest_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_lines = [
        "# TextFake receipt dataset processing summary",
        "",
        f"- Input: `{_safe_relative(input_path, root)}`",
        f"- Normalized output: `{_safe_relative(output_path, root)}`",
        f"- Source images decoded: **{len(rows)}**",
        f"- Selected balanced images: **{len(selected)}** ({json.dumps(balance_audit['selected_counts'], sort_keys=True)})",
        f"- Review rows: **{audit['review_count']}**",
        f"- Excluded rows: **{audit['excluded_count'] + len(balance_exclusions)}**",
        f"- Target canvas: **{target_width}x{target_height} RGB PNG**",
        "",
        "The raw dataset was not modified. The balanced manifest is the intended input for later split creation.",
        "No model training was performed by this processing run.",
        "",
        "## Important limitations",
        "",
        "- The input format and source dimensions were class-correlated; the normalized copies remove the direct format and dimension shortcut.",
        "- Original compression and acquisition differences may still be present in pixel values and require interpretation through the shortcut audit.",
        "- Generator and acquisition metadata were not supplied and are recorded as explicit unknown values.",
        "- Near-duplicate candidates were not silently included in the balanced manifest.",
    ]
    summary_path = manifest_path / "textfake_processing_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "schema_version": PROCESSING_SCHEMA_VERSION,
        "source_manifest": str(source_manifest_path),
        "audited_manifest": str(audited_manifest_path),
        "balanced_manifest": str(balanced_manifest_path),
        "audit": str(audit_path),
        "shortcut_audit": str(shortcut_path),
        "duplicate_review": str(duplicate_path),
        "summary": str(summary_path),
        "manifest_hashes": manifest_hashes,
        "source_count": len(rows),
        "selected_count": len(selected),
        "selected_counts": balance_audit["selected_counts"],
        "review_count": audit["review_count"],
        "excluded_count": audit["excluded_count"],
        "skipped_count": len(skipped),
    }
