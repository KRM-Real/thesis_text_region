"""Feature registry and shared metadata for experimental measurements."""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    status: str
    scope: str
    description: str
    formula: str
    preprocessing: str
    expected_range: str
    failure_behavior: str
    caveats: str
    output_type: str
    interpretation: str
    included_in_model_vector: bool


FEATURE_REGISTRY: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="sharpness_laplacian_variance",
        group="sharpness",
        status="EXPERIMENTAL_FEATURE",
        scope="region",
        description="Variance of the grayscale Laplacian as a high-frequency detail proxy.",
        formula="variance(Laplacian(gray_crop, kernel=3))",
        preprocessing="Native-resolution rectangular crop converted to uint8 grayscale.",
        expected_range="0 to unbounded; image-size and resolution dependent.",
        failure_behavior="Unavailable when the crop is smaller than 3x3 or calculation fails.",
        caveats="Sensitive to crop size, resolution, compression, resizing, and background context.",
        output_type="float",
        interpretation="A larger value indicates more high-frequency variation in this crop; it is not an authenticity verdict.",
        included_in_model_vector=False,
    ),
    FeatureDefinition(
        name="edge_density",
        group="edge_rendering",
        status="EXPERIMENTAL_FEATURE",
        scope="region",
        description="Proportion of pixels classified as Canny edges in the crop.",
        formula="count(Canny(gray_crop, low, high) > 0) / pixel_count",
        preprocessing="Native-resolution crop converted to uint8 grayscale.",
        expected_range="0 to 1.",
        failure_behavior="Unavailable when the crop is smaller than 3x3, has no pixels, or edge calculation fails.",
        caveats="Threshold and font size sensitive; edges can come from the surrounding background.",
        output_type="float",
        interpretation="The fraction of crop pixels identified as Canny edges; it is an exploratory rendering proxy.",
        included_in_model_vector=False,
    ),
    FeatureDefinition(
        name="intensity_std",
        group="contrast",
        status="EXPERIMENTAL_FEATURE",
        scope="region",
        description="Population standard deviation of grayscale intensities in the crop.",
        formula="sqrt(mean((gray_crop - mean(gray_crop))^2))",
        preprocessing="Native-resolution rectangular crop converted to uint8 grayscale.",
        expected_range="0 to approximately 127.5 for uint8 values.",
        failure_behavior="Unavailable when the crop has no pixels.",
        caveats="Includes any background inside the rectangular crop; not a foreground segmentation measure.",
        output_type="float",
        interpretation="The spread of grayscale intensity values in the crop, including the retained context.",
        included_in_model_vector=False,
    ),
    FeatureDefinition(
        name="text_region_area_ratio",
        group="density",
        status="EXPERIMENTAL_FEATURE",
        scope="image",
        description="Union area of clipped detector rectangles divided by full image area.",
        formula="union_area(clipped_detection_boxes) / (image_width * image_height)",
        preprocessing="Uses unpadded clipped detector boxes and raster union to avoid overlap inflation.",
        expected_range="0 to 1.",
        failure_behavior="Zero for a successful empty detection; unavailable when detection fails.",
        caveats="A detector-localization measure, not a semantic text amount measure.",
        output_type="float",
        interpretation="The raster union fraction covered by clipped detector rectangles.",
        included_in_model_vector=True,
    ),
)

DEFERRED_FEATURE_GROUPS: tuple[dict[str, object], ...] = (
    {
        "name": "texture_group",
        "group": "texture",
        "status": "DEFERRED",
        "scope": "pending",
        "description": "Texture descriptors such as GLCM, LBP, or entropy require a separate pilot decision.",
        "formula": "not implemented in this reliability milestone",
        "preprocessing": "pending methodology decision",
        "expected_range": "pending",
        "failure_behavior": "not available",
        "interpretation": "Reserved thesis feature group; not a current model input.",
        "caveats": "Do not infer usefulness from its absence in this pilot.",
        "output_type": "pending",
        "included_in_model_vector": False,
    },
    {
        "name": "shape_stroke_group",
        "group": "shape_stroke",
        "status": "DEFERRED",
        "scope": "pending",
        "description": "Connected-component geometry and stroke appearance are deferred.",
        "formula": "not implemented in this reliability milestone",
        "preprocessing": "pending methodology decision",
        "expected_range": "pending",
        "failure_behavior": "not available",
        "interpretation": "Reserved thesis feature group; not a current model input.",
        "caveats": "Character segmentation and stroke-width stability need separate study.",
        "output_type": "pending",
        "included_in_model_vector": False,
    },
    {
        "name": "spacing_alignment_group",
        "group": "spacing_alignment",
        "status": "DEFERRED",
        "scope": "pending",
        "description": "Spacing and alignment relationships between text regions are deferred.",
        "formula": "not implemented in this reliability milestone",
        "preprocessing": "pending methodology decision",
        "expected_range": "pending",
        "failure_behavior": "not available",
        "interpretation": "Reserved thesis feature group; not a current model input.",
        "caveats": "These are naturally image-level and depend on reliable multi-region detection.",
        "output_type": "pending",
        "included_in_model_vector": False,
    },
)


def feature_registry() -> list[dict[str, object]]:
    """Return registry metadata suitable for JSON export or UI display."""

    entries: list[dict[str, object]] = []
    for definition in FEATURE_REGISTRY:
        entry = definition.__dict__.copy()
        entry["calculation"] = entry["formula"]
        entries.append(entry)
    return entries
