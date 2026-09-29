"""Detection-only EasyOCR adapter.

EasyOCR is used as a practical ``PILOT_DECISION`` for locating visible text.
The recognizer is disabled and this module never asks for or stores text
strings.  Detector failures are raised so the UI can report them explicitly.
"""

from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from src.types import DetectionConfig

from .types import RawDetection


class DetectionError(RuntimeError):
    """Raised when the configured detector cannot initialize or run."""


def _is_sequence(value: Any) -> bool:
    """Return whether ``value`` can contain nested detector coordinates."""

    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def _is_horizontal_box(value: Any) -> bool:
    """Identify an EasyOCR ``[x_min, x_max, y_min, y_max]`` box."""

    return (
        _is_sequence(value)
        and len(value) >= 4
        and all(np.isscalar(coordinate) for coordinate in value[:4])
    )


def _is_point(value: Any) -> bool:
    """Identify a two-coordinate point without accepting nested lists."""

    return (
        _is_sequence(value)
        and len(value) >= 2
        and all(np.isscalar(coordinate) for coordinate in value[:2])
    )


def _is_polygon(value: Any) -> bool:
    """Identify a free-form polygon returned by EasyOCR."""

    return _is_sequence(value) and len(value) >= 3 and all(_is_point(point) for point in value)


def _iter_nested_matches(value: Any, predicate: Callable[[Any], bool]) -> Iterator[Any]:
    """Yield coordinate records through EasyOCR's optional batch nesting.

    ``Reader.detect`` returns lists grouped per input image, even for a single
    image. Empty/malformed branches are ignored, while valid boxes or polygons
    are yielded exactly once.
    """

    if predicate(value):
        yield value
        return
    if _is_sequence(value):
        for nested in value:
            yield from _iter_nested_matches(nested, predicate)


class EasyOCRTextDetector:
    """Wrap EasyOCR's localization API without invoking recognition."""

    name = "EasyOCR CRAFT (detection-only)"
    status = "PILOT_DECISION"

    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        gpu: bool = False,
        model_storage_directory: str | None = None,
        user_network_directory: str | None = None,
        download_enabled: bool = True,
    ) -> None:
        if model_storage_directory is None:
            # EasyOCR defaults to ``%USERPROFILE%\\.EasyOCR``. A project-local
            # cache is more reproducible and works in restricted Windows
            # environments where that profile directory is not writable.
            model_storage_directory = os.environ.get(
                "PILOT_EASYOCR_MODEL_DIR",
                str(Path.cwd() / "easyocr-models"),
            )
        if user_network_directory is None:
            user_network_directory = str(Path(model_storage_directory) / "user_network")

        try:
            import easyocr  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise DetectionError(
                "EasyOCR is not installed. Install requirements.txt before running the pilot."
            ) from exc

        try:
            # recognizer=False is the critical research boundary: only the
            # detector weights are loaded and no words are requested.
            self.reader = easyocr.Reader(
                list(languages),
                gpu=gpu,
                detector=True,
                recognizer=False,
                model_storage_directory=model_storage_directory,
                user_network_directory=user_network_directory,
                download_enabled=download_enabled,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            raise DetectionError(f"EasyOCR detector initialization failed: {exc}") from exc

    def detect(
        self, image_rgb: np.ndarray, config: DetectionConfig | None = None
    ) -> list[RawDetection]:
        """Return polygons and confidence values without recognized words."""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise DetectionError("Detector input must be an RGB image array.")
        config = config or DetectionConfig()
        try:
            kwargs = {
                "min_size": config.min_size,
                "text_threshold": config.text_threshold,
                "low_text": config.low_text,
                "link_threshold": config.link_threshold,
                "canvas_size": config.canvas_size,
                "mag_ratio": config.mag_ratio,
                "bbox_min_score": config.bbox_min_score,
                "bbox_min_size": max(3, config.min_size),
            }
            # Keep lightweight reader fakes and older EasyOCR adapters usable
            # while forwarding every setting supported by the installed API.
            try:
                parameters = inspect.signature(self.reader.detect).parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                if not accepts_kwargs:
                    kwargs = {key: value for key, value in kwargs.items() if key in parameters}
            except (TypeError, ValueError):
                pass
            horizontal_list, free_list = self.reader.detect(image_rgb, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise DetectionError(f"EasyOCR detection failed: {exc}") from exc

        detections: list[RawDetection] = []
        # EasyOCR groups both output lists per input image. Recursively matching
        # the coordinate shape supports its single-image batch wrapper as well
        # as already-flat values without confusing a polygon point for a box.
        for box in _iter_nested_matches(horizontal_list, _is_horizontal_box):
            x_min, x_max, y_min, y_max = [float(v) for v in box[:4]]
            detections.append(
                RawDetection(
                    polygon=[
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                )
            )
        for polygon in _iter_nested_matches(free_list, _is_polygon):
            points = [(float(point[0]), float(point[1])) for point in polygon if len(point) >= 2]
            detections.append(RawDetection(polygon=points))
        return detections
