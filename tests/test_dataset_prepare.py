from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.data.manifest import load_manifest
from src.data.prepare import prepare_dataset


def _write_image(path: Path, offset: int, fmt: str) -> None:
    image = Image.new("RGB", (80, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10 + offset, 20, 65, 25), fill="black")
    draw.rectangle((15, 45 + offset, 60, 50 + offset), fill="black")
    image.save(path, format=fmt)


def test_prepare_dataset_preserves_raw_files_and_creates_balanced_normalized_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "dataset" / "Receipts"
    real_root = input_root / "0_real"
    fake_root = input_root / "1_fake"
    real_root.mkdir(parents=True)
    fake_root.mkdir(parents=True)
    _write_image(real_root / "same-name.jpg", 0, "JPEG")
    _write_image(real_root / "real-two.jpg", 1, "JPEG")
    _write_image(fake_root / "same-name.png", 2, "PNG")
    _write_image(fake_root / "fake-two.png", 3, "PNG")
    (input_root / "density_results.json").write_text("[]\n", encoding="utf-8")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_root.rglob("*") if path.is_file()}

    result = prepare_dataset(
        input_root,
        tmp_path / "data" / "processed" / "textfake",
        tmp_path / "data" / "manifests",
        target_width=64,
        target_height=96,
        project_root=tmp_path,
        near_duplicate_distance=0,
    )

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_root.rglob("*") if path.is_file()}
    assert before == after
    assert result["selected_count"] == 4
    assert result["selected_counts"] == {"ai_generated": 2, "non_ai_generated": 2}

    records = load_manifest(result["balanced_manifest"], tmp_path / "data")
    assert len(records) == 4
    assert {record.format for record in records} == {"PNG"}
    assert {(record.width, record.height) for record in records} == {(64, 96)}
    assert {record.label for record in records} == {"ai_generated", "non_ai_generated"}

    manifest_rows = list(csv.DictReader(Path(result["source_manifest"]).open(encoding="utf-8", newline="")))
    assert len(manifest_rows) == 4
    assert all(row["source_format"] in {"JPEG", "PNG"} for row in manifest_rows)
    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    assert audit["source_audit"]["skipped_file_count"] == 1
    assert audit["source_audit"]["decode_failure_count"] == 0
