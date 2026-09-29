"""Visual feature extraction and image-level aggregation."""

from .pipeline import FeatureExtractionResult, extract_features
from .visual import compute_region_features, image_geometry_features

__all__ = ["FeatureExtractionResult", "extract_features", "compute_region_features", "image_geometry_features"]
