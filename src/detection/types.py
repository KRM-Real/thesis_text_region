"""Types used by detector adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDetection:
    """Normalized detector output before clipping to image bounds."""

    polygon: list[tuple[float, float]]
    confidence: float | None = None
