"""Shared image, path, and logging helpers."""

from __future__ import annotations

import logging
from typing import BinaryIO
from pathlib import Path
from typing import Union

from PIL import Image, ImageOps

from src.config import MAX_IMAGE_SIDE

ImageInput = Union[str, Path, Image.Image, BinaryIO]


def configure_logging() -> None:
    """Configure concise application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def open_rgb_image(
    image_path_or_pil: ImageInput,
    max_side: int = MAX_IMAGE_SIDE,
) -> Image.Image:
    """Open an image as RGB and shrink it if its longest side is too large."""
    if isinstance(image_path_or_pil, Image.Image):
        opened_image = image_path_or_pil.copy()
    else:
        with Image.open(image_path_or_pil) as source:
            opened_image = source.copy()
    image = ImageOps.exif_transpose(opened_image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    if max_side > 0 and max(image.size) > max_side:
        ratio = max_side / max(image.size)
        new_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image
