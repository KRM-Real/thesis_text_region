"""Detection and preprocessing visualizations."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from src.types import TextRegion


def draw_detection_overlay(image: Image.Image, regions: list[TextRegion]) -> Image.Image:
    """Draw numbered polygons/boxes on a copy of the original RGB image."""

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    for index, region in enumerate(regions, start=1):
        points = region.polygon or [
            (region.bbox.x, region.bbox.y),
            (region.bbox.x2, region.bbox.y),
            (region.bbox.x2, region.bbox.y2),
            (region.bbox.x, region.bbox.y2),
        ]
        points = [(int(x), int(y)) for x, y in points]
        draw.line(points + [points[0]], fill=(220, 35, 35), width=3)
        label = region.region_id
        if region.confidence is not None:
            label = f"{label} ({region.confidence:.2f})"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = right - left, bottom - top
        anchor_x, anchor_y = points[0]
        draw.rectangle(
            (anchor_x, max(0, anchor_y - text_height - 4), anchor_x + text_width + 6, anchor_y),
            fill=(220, 35, 35),
        )
        draw.text((anchor_x + 3, max(0, anchor_y - text_height - 2)), label, fill="white", font=font)
    return overlay


def make_edge_image(edge_map) -> Image.Image:
    """Convert a Canny edge array to a display image."""

    return Image.fromarray(edge_map, mode="L")
