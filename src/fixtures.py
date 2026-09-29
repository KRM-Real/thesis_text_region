"""Deterministic geometric-image fixtures for tests and software demos only."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .contracts import ImageRecord
from .data.manifest import write_manifest
from .reproducibility import sha256_file


def generate_fixture_dataset(root: str | Path = "data", count_per_label: int = 12, seed: int = 42) -> Path:
    root_path = Path(root)
    fixture_root = root_path / "fixtures"
    fixture_root.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    for label in ("non_ai_generated", "ai_generated"):
        directory = fixture_root / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_label):
            rng = random.Random(seed + index + (0 if label == "non_ai_generated" else 10_000))
            width, height = 320 + rng.randrange(0, 4) * 16, 220 + rng.randrange(0, 3) * 12
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            line_count = 3 + index % 3
            top = 35 + rng.randrange(0, 20)
            for line in range(line_count):
                y = top + line * (18 + rng.randrange(0, 5))
                x = 24 + rng.randrange(0, 20)
                line_width = width - 50 - rng.randrange(0, 40)
                thickness = 3 + (index + line) % 3
                draw.rectangle((x, y, min(width - 10, x + line_width), y + thickness), fill=(25, 25, 25))
                if line % 2 == 0:
                    draw.rectangle((x + 6, y - 8, x + 20 + rng.randrange(0, 16), y + thickness + 8), fill=(25, 25, 25))
            if label == "ai_generated":
                image = image.rotate((index % 3) - 1, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="white")
                if index % 4 == 0:
                    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
            path = directory / f"fixture_{index:03d}.png"
            image.save(path, format="PNG")
            records.append(ImageRecord(
                sample_id=f"fixture-{label}-{index:03d}",
                relative_path=str(path.relative_to(root_path)).replace("\\", "/"),
                label=label,
                category="receipt",
                source_id="generated-fixture",
                group_id=f"fixture-group-{label}-{index:03d}",
                provenance_type="generated_fixture",
                generator_id="fixture-generator",
                acquisition_type="synthetic_fixture",
                width=image.width,
                height=image.height,
                format="PNG",
                sha256=sha256_file(path),
            ))
    manifest_path = root_path / "manifests" / "fixture_manifest.csv"
    write_manifest(manifest_path, records)
    return manifest_path
