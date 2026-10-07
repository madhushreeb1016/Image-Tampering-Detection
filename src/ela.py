"""Error Level Analysis (ELA) image generation and visualization."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from src.config import DEFAULT_ELA_QUALITY
from src.utils import ImageInput, open_rgb_image


def generate_ela(
    image_path_or_pil: ImageInput,
    quality: int = DEFAULT_ELA_QUALITY,
    scale: float | None = None,
) -> Image.Image:
    """Create a brightness-amplified JPEG recompression-difference image.

    Images are converted to RGB and limited to the configured maximum side
    length before recompression. With ``scale=None``, the largest difference
    is mapped to full brightness so the ELA remains visible.
    """
    if not 1 <= quality <= 95:
        raise ValueError("ELA quality must be between 1 and 95.")
    original = open_rgb_image(image_path_or_pil)
    buffer = BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    original_pixels = np.asarray(original, dtype=np.int16)
    recompressed_pixels = np.asarray(recompressed, dtype=np.int16)
    difference = np.abs(original_pixels - recompressed_pixels).astype(np.float32)
    largest_difference = float(difference.max())

    if scale is None:
        multiplier = 255.0 / largest_difference if largest_difference else 1.0
    else:
        if scale < 0:
            raise ValueError("ELA scale cannot be negative.")
        multiplier = scale
    amplified = np.clip(difference * multiplier, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified, mode="RGB")


def ela_heatmap(
    ela_image: Image.Image,
    original_image: Image.Image | None = None,
    alpha: float = 0.5,
) -> tuple[Image.Image, Image.Image]:
    """Return a JET-colored ELA heatmap and its blend over the source image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    ela_rgb = ela_image.convert("RGB")
    ela_array = np.asarray(ela_rgb)
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))

    if original_image is None:
        base = ela_rgb
    else:
        base = open_rgb_image(original_image)
        if base.size != heatmap.size:
            base = base.resize(heatmap.size, Image.Resampling.LANCZOS)
    overlay = Image.blend(base, heatmap, alpha)
    return heatmap, overlay
