"""Nonsemantic localization visualizations shared by CLI and UI clients."""

from __future__ import annotations

from PIL import Image, ImageDraw

def render_overlay(image: Image.Image, extraction: object) -> Image.Image:
    """Draw deterministic region boxes and numeric IDs without OCR text."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    line_width = max(2, output.width // 400)
    for index, region in enumerate(extraction.detection.regions, start=1):
        x, y, width, height = region.bbox_px
        draw.rectangle((x, y, x + width, y + height), outline=(255, 80, 40), width=line_width)
        draw.text((x + 3, max(0, y - 14)), str(index), fill=(255, 40, 20))
    return output
