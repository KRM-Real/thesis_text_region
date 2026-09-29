"""Deterministic image-level, group-aware train/validation/test splitting."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from ..contracts import ImageRecord
from .manifest import ManifestError


def validate_split_leakage(records: list[ImageRecord]) -> None:
    """Fail closed when exact files or related groups cross saved partitions."""

    by_hash: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.split not in {"train", "validation", "test"}:
            continue
        by_hash[record.sha256].add(record.split)
        by_group[record.group_id].add(record.split)
    duplicate_crossings = [key for key, splits in by_hash.items() if len(splits) > 1]
    group_crossings = [key for key, splits in by_group.items() if len(splits) > 1]
    if duplicate_crossings:
        raise ManifestError(f"exact duplicate checksum crosses splits: {duplicate_crossings[:5]}")
    if group_crossings:
        raise ManifestError(f"related group crosses splits: {group_crossings[:5]}")


def _choose_fold(records: list[ImageRecord], folds: list[tuple[list[int], list[int]]], target: float, seed: int) -> tuple[list[int], list[int]]:
    total = len(records)
    overall = {label: sum(record.label == label for record in records) for label in {record.label for record in records}}
    best: tuple[float, int, tuple[list[int], list[int]]] | None = None
    for index, (train_indices, test_indices) in enumerate(folds):
        test_records = [records[i] for i in test_indices]
        label_error = sum(abs(sum(record.label == label for record in test_records) / max(1, overall[label]) - target) for label in overall)
        size_error = abs(len(test_records) / max(1, total) - target)
        score = size_error + label_error
        candidate = (score, index, (train_indices, test_indices))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2]


def create_split_manifest(records: list[ImageRecord], seed: int = 42) -> list[ImageRecord]:
    if any(record.inclusion_status != "included" for record in records):
        raise ManifestError("split creation accepts included rows only; resolve excluded/review rows first")
    validate_split_leakage(records)
    checksum_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        checksum_groups[record.sha256].append(record.sample_id)
    duplicate_ids = [sample_ids for sample_ids in checksum_groups.values() if len(sample_ids) > 1]
    if duplicate_ids:
        raise ManifestError(f"exact duplicate images must be resolved before splitting: {duplicate_ids[:5]}")
    if len(records) < 12:
        raise ManifestError("At least 12 included images are required for a three-way split")
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record.group_id].add(record.label or "")
    mixed = {group: labels for group, labels in groups.items() if len(labels) > 1}
    if mixed:
        raise ManifestError(f"group_id crosses labels and cannot be split safely: {sorted(mixed)}")
    labels = [record.label for record in records]
    group_values = [record.group_id for record in records]
    min_class_count = min(Counter(labels).values())
    n_test_splits = min(7, max(2, len(set(group_values)), min_class_count))
    if n_test_splits > min_class_count:
        n_test_splits = min_class_count
    if n_test_splits < 2:
        raise ManifestError("Not enough samples per class for a stratified test split")
    if n_test_splits < 3:
        raise ManifestError("Not enough groups for a group-aware test split")
    first = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    first_folds = list(first.split(records, labels, group_values))
    train_val_indices, test_indices = _choose_fold(records, first_folds, 0.15, seed)
    remaining = [records[i] for i in train_val_indices]
    remaining_labels = [record.label for record in remaining]
    remaining_groups = [record.group_id for record in remaining]
    remaining_class_count = min(Counter(remaining_labels).values())
    n_validation_splits = min(6, max(2, len(set(remaining_groups)), remaining_class_count))
    if n_validation_splits > remaining_class_count:
        n_validation_splits = remaining_class_count
    if n_validation_splits < 2:
        raise ManifestError("Not enough remaining samples per class for a stratified validation split")
    second = StratifiedGroupKFold(n_splits=n_validation_splits, shuffle=True, random_state=seed + 1)
    second_folds = list(second.split(remaining, remaining_labels, remaining_groups))
    _, validation_local = _choose_fold(remaining, second_folds, 0.15 / 0.85, seed + 1)
    validation_indices = [train_val_indices[i] for i in validation_local]
    validation_set = set(validation_indices) | set(test_indices)
    train_indices = [index for index in range(len(records)) if index not in validation_set]
    split_by_index = {index: "train" for index in train_indices}
    split_by_index.update({index: "validation" for index in validation_indices})
    split_by_index.update({index: "test" for index in test_indices})
    split_records = []
    for index, record in enumerate(records):
        split_records.append(ImageRecord(**{**record.to_dict(), "split": split_by_index[index]}))
    for split in ("train", "validation", "test"):
        split_labels = {record.label for record in split_records if record.split == split}
        if len(split_labels) < 2:
            raise ManifestError(f"split {split} does not contain both labels; record an approved alternative before changing the policy")
    return split_records


def write_split_manifest(path: str | Path, records: list[ImageRecord]) -> None:
    from .manifest import write_manifest

    write_manifest(path, records)
