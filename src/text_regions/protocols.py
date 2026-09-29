"""Replaceable detector boundary for production and fixture adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .adapters import RawGeometry


class TextRegionDetector(Protocol):
    """Return geometry-only detections for one RGB image."""

    name: str
    version: str

    def detect(self, image_rgb: np.ndarray, config: dict[str, Any] | None = None) -> list[RawGeometry]:
        ...
