"""Explicit image loading, clipping, and display preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from src.types import BoundingBox, PreprocessingConfig


class ImageInputError(ValueError):
    """Raised when input bytes are not a supported, decodable image."""


def load_image(source: bytes | bytearray | str | Path) -> Image.Image:
    """Decode a PNG/JPEG/WEBP image and return an RGB image.

    The uploaded image is not resized. Transparent images are composited onto
    white solely to produce a three-channel representation for the detector.
    """

    try:
        with Image.open(BytesIO(source) if isinstance(source, (bytes, bytearray)) else source) as image:
            image.load()
            if image.format and image.format.upper() not in {"PNG", "JPEG", "JPG", "WEBP"}:
                raise ImageInputError(f"Unsupported image type: {image.format}")
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                return Image.alpha_composite(background, rgba).convert("RGB")
            return image.convert("RGB")
    except ImageInputError:
        raise
    except Exception as exc:
        raise ImageInputError(f"Could not decode the uploaded image: {exc}") from exc


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return an unresized uint8 RGB array."""

    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


@dataclass(frozen=True)
class DetectionPreparation:
    """Detector input and the transform needed to restore original coordinates."""

    image_rgb: np.ndarray
    scale_x: float
    scale_y: float
    mode: str
    operations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        height, width = self.image_rgb.shape[:2]
        return {
            "mode": self.mode,
            "operations": list(self.operations),
            "input_width": int(width),
            "input_height": int(height),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }


def prepare_detection_image(
    rgb: np.ndarray, config: PreprocessingConfig | None = None
) -> DetectionPreparation:
    """Prepare a detector-only image and retain an explicit coordinate transform.

    Features are never computed from this prepared image. The returned scale
    values map coordinates from the prepared detector input back to the native
    image by division.
    """

    config = config or PreprocessingConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Expected a non-empty RGB image array.")
    prepared = rgb.astype(np.uint8, copy=True)
    operations: list[str] = []
    scale_x = 1.0
    scale_y = 1.0
    if config.mode == "clahe":
        try:
            import cv2

            gray = to_grayscale(prepared)
            clahe = cv2.createCLAHE(
                clipLimit=float(config.clahe_clip_limit),
                tileGridSize=tuple(int(value) for value in config.clahe_grid_size),
            )
            enhanced = clahe.apply(gray)
            prepared = np.repeat(enhanced[..., None], 3, axis=2)
            operations.append("clahe_grayscale")
        except Exception as exc:
            raise ValueError(f"CLAHE preprocessing failed: {exc}") from exc
    if config.scale != 1.0:
        try:
            import cv2

            height, width = prepared.shape[:2]
            new_size = (max(1, int(round(width * config.scale))), max(1, int(round(height * config.scale))))
            interpolation = cv2.INTER_CUBIC if config.scale > 1 else cv2.INTER_AREA
            prepared = cv2.resize(prepared, new_size, interpolation=interpolation)
            # Keep the realized transform (rather than only the requested
            # factor) because rounding a small dimension can make the two
            # values differ slightly.
            scale_x = float(new_size[0] / width)
            scale_y = float(new_size[1] / height)
            operations.append(f"resize_{config.scale:g}x")
        except Exception as exc:
            raise ValueError(f"Detection resize failed: {exc}") from exc
    return DetectionPreparation(
        image_rgb=prepared,
        scale_x=scale_x,
        scale_y=scale_y,
        mode=config.mode,
        operations=tuple(operations) if operations else ("none",),
    )


def restore_polygon_coordinates(
    polygon: list[tuple[float, float]], scale_x: float, scale_y: float
) -> list[tuple[float, float]]:
    """Map detector-input polygon coordinates to native image coordinates."""

    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Coordinate scales must be positive")
    return [(float(x) / scale_x, float(y) / scale_y) for x, y in polygon]


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB pixels to uint8 luminance using OpenCV's standard weights."""

    if rgb.ndim == 2:
        return rgb.astype(np.uint8, copy=False)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB array or a grayscale array.")
    # Avoid importing OpenCV just for this deterministic conversion.
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def clip_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    """Clip a rectangle to valid image coordinates."""

    x1 = max(0, min(image_width, bbox.x))
    y1 = max(0, min(image_height, bbox.y))
    x2 = max(x1, min(image_width, bbox.x2))
    y2 = max(y1, min(image_height, bbox.y2))
    return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def crop_region(rgb: np.ndarray, bbox: BoundingBox, padding: int = 0) -> tuple[np.ndarray | None, BoundingBox]:
    """Clip and crop an RGB array, returning ``None`` for an empty crop."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image array.")
    raw = BoundingBox(
        bbox.x - padding,
        bbox.y - padding,
        bbox.width + 2 * padding,
        bbox.height + 2 * padding,
    )
    clipped = clip_bbox(raw, rgb.shape[1], rgb.shape[0])
    if clipped.width <= 0 or clipped.height <= 0:
        return None, clipped
    return rgb[clipped.y : clipped.y2, clipped.x : clipped.x2].copy(), clipped


def draw_grayscale(gray: np.ndarray) -> Image.Image:
    """Convert a grayscale array to a displayable PIL image."""

    return Image.fromarray(gray.astype(np.uint8), mode="L")
