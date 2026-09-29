"""Research-safe analysis logging.

Logs describe pipeline state and failures without logging OCR-recognized text
or any other semantic content.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.types import ImageAnalysis


LOGGER = logging.getLogger("visual_text_region_pilot")


def configure_logging(log_path: str | Path | None = None) -> None:
    """Configure a predictable console/file logger once per process."""

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(console)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        matching_file = None
        for handler in list(LOGGER.handlers):
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == resolved:
                    matching_file = handler
                else:
                    LOGGER.removeHandler(handler)
                    handler.close()
        if matching_file is None:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOGGER.addHandler(file_handler)


def log_analysis_summary(analysis: ImageAnalysis, export_paths: tuple[str, ...] = ()) -> None:
    """Log one concise, OCR-free summary for an analyzed image."""

    failed_features = [
        name
        for region in analysis.regions
        for name, result in region.features.items()
        if result.status != "ok"
    ]
    failed_features.extend(
        name for name, result in analysis.image_features.items() if result.status != "ok"
    )
    aggregation = analysis.settings.get("aggregation", [])
    diagnostics = analysis.diagnostics
    rejection_reasons = sorted(
        {str(item.get("reason")) for item in diagnostics.get("rejected_detections", []) if item.get("reason")}
    )
    LOGGER.info(
        "analysis_id=%s detection_status=%s raw_detections=%s accepted=%s rejected=%s rejection_reasons=%s failed_features=%s aggregation=%s exports=%s warnings=%d",
        analysis.image_id,
        analysis.detection_status,
        diagnostics.get("raw_detection_count", len(analysis.regions)),
        diagnostics.get("accepted_detection_count", len(analysis.regions)),
        diagnostics.get("rejected_detection_count", 0),
        rejection_reasons,
        sorted(set(failed_features)),
        aggregation,
        export_paths,
        len(analysis.warnings),
    )
