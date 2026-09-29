"""Small representative runtime checks for the exploratory pilot.

This is intentionally separate from the automated unit suite: it exercises the
real EasyOCR detector on a few images and records resolution/padding sensitivity
without treating the measurements as thesis findings.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_pilot_config  # noqa: E402
from src.detection.detector import EasyOCRTextDetector  # noqa: E402
from src.pipeline import analyze_image  # noqa: E402
from src.preprocessing.image_ops import load_image  # noqa: E402


def _record(name: str, image: Image.Image, detector: EasyOCRTextDetector, config, padding: int) -> dict[str, object]:
    analysis, _ = analyze_image(
        image,
        detector,
        config=replace(config, crop_padding=padding),
        image_id=name,
    )
    return {
        "name": name,
        "padding": padding,
        "width": image.width,
        "height": image.height,
        "detection_status": analysis.detection_status,
        "accepted_detection_count": analysis.diagnostics.get("accepted_detection_count", 0),
        "rejected_detection_count": analysis.diagnostics.get("rejected_detection_count", 0),
        "failed_features": analysis.diagnostics.get("failed_features", []),
        "model_feature_record": analysis.model_feature_record(),
    }


def main() -> None:
    config = load_pilot_config(ROOT / "configs" / "pilot.yaml")
    detector = EasyOCRTextDetector(
        languages=config.detection.languages,
        gpu=config.detection.gpu,
        model_storage_directory=str(ROOT / "easyocr-models"),
    )
    examples = {
        "plain_poster": ROOT / "dataset" / "real" / "588990a995a7a863ddcc33a3_preview.jpg",
        "photographic_background": ROOT / "dataset" / "real" / "5914265f95a7a863ddcd7b69_preview.jpg",
        "graphic_poster": ROOT / "dataset" / "fake" / "commercial_poster_0716.jpg",
    }
    results: list[dict[str, object]] = []
    for name, path in examples.items():
        image = load_image(path)
        results.append(_record(name, image, detector, config, padding=0))
        results.append(_record(f"{name}_padding4", image, detector, config, padding=4))

    repeated_image = load_image(examples["plain_poster"])
    first = _record("repeat_a", repeated_image, detector, config, padding=0)
    second = _record("repeat_b", repeated_image, detector, config, padding=0)
    resized = repeated_image.resize(
        (max(1, repeated_image.width // 2), max(1, repeated_image.height // 2)),
        Image.Resampling.LANCZOS,
    )
    resized_record = _record("plain_poster_half_resolution", resized, detector, config, padding=0)
    output = ROOT / "outputs" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "Operational sensitivity checks only; not thesis findings.",
        "repeated_run_model_features_equal": first["model_feature_record"] == second["model_feature_record"],
        "runs": results + [first, second, resized_record],
    }
    target = output / "representative_runtime_checks.json"
    target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
