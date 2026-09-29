"""Detector adapters and geometry/crop orchestration."""

from .adapters import EasyOCRDetectorAdapter, FixtureTextDetector
from .pipeline import LocalizationArtifacts, localize_image
from .protocols import TextRegionDetector
from .visualization import render_overlay

__all__ = ["EasyOCRDetectorAdapter", "FixtureTextDetector", "LocalizationArtifacts", "TextRegionDetector", "localize_image", "render_overlay"]
