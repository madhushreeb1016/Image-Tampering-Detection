"""CASIA-style dataset discovery, feature caching, and demo-data generation."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config import (
    AUTHENTIC_DIR,
    DEFAULT_ELA_QUALITY,
    FEATURES_CSV,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
    TAMPERED_DIR,
)
from src.ela import generate_ela
from src.features import extract_features

LOGGER = logging.getLogger(__name__)


def discover_images(
    authentic_dir: Path = AUTHENTIC_DIR,
    tampered_dir: Path = TAMPERED_DIR,
    limit: int | None = None,
) -> list[tuple[Path, int]]:
    """Find supported image files and return a balanced list of path/label pairs."""
    if limit is not None and limit < 1:
        raise ValueError("Limit must be a positive integer.")
    class_paths: dict[int, list[Path]] = {}
    for label, directory in ((0, authentic_dir), (1, tampered_dir)):
        if not directory.exists():
            class_paths[label] = []
            continue
        class_paths[label] = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not class_paths[0] or not class_paths[1]:
        return []

    # Balance classes by undersampling the larger class, then apply the optional cap.
    balanced_count = min(len(class_paths[0]), len(class_paths[1]))
    if limit is not None:
        balanced_count = min(balanced_count, limit)
    rng = random.Random(42)
    selected: list[tuple[Path, int]] = []
    for label, paths in class_paths.items():
        chosen = sorted(rng.sample(paths, balanced_count))
        selected.extend((path, label) for path in chosen)
    return selected


def build_features_csv(
    limit: int | None = None,
    quality: int = DEFAULT_ELA_QUALITY,
    output_csv: Path = FEATURES_CSV,
    force: bool = False,
) -> pd.DataFrame:
    """Extract and cache balanced ELA features for the discovered dataset."""
    image_items = discover_images(limit=limit)
    if not image_items:
        raise FileNotFoundError(
            "CASIA-style images were not found in data/raw/Au and data/raw/Tp. "
            "Download CASIA v2 (see README.md), then place authentic images in "
            "data/raw/Au and tampered images in data/raw/Tp."
        )

    expected_names = [str(path.relative_to(PROJECT_ROOT)) for path, _ in image_items]
    existing = None
    if output_csv.exists() and not force:
        try:
            candidate = pd.read_csv(output_csv)
            if (
                "filename" in candidate.columns
                and "label" in candidate.columns
                and "ela_quality" in candidate.columns
                and candidate["filename"].tolist() == expected_names
                and candidate["label"].tolist() == [label for _, label in image_items]
                and candidate["ela_quality"].eq(quality).all()
            ):
                existing = candidate
                LOGGER.info("Using cached features from %s", output_csv)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            LOGGER.warning("Feature cache could not be read; recomputing it.")
    if existing is not None:
        return existing

    records: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    for path, label in tqdm(image_items, desc="Extracting ELA features"):
        try:
            ela_image = generate_ela(path, quality=quality)
            feature_values, names = extract_features(ela_image)
            feature_names = names
            records.append(
                {"filename": str(path.relative_to(PROJECT_ROOT)), "label": label,
                 "ela_quality": quality,
                 **dict(zip(names, feature_values))}
            )
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            LOGGER.warning("Skipping unreadable image %s: %s", path, error)

    if not records or feature_names is None:
        raise ValueError("No readable images were available for feature extraction.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        records, columns=["filename", "label", "ela_quality", *feature_names]
    )
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Saved %d image features to %s", len(frame), output_csv)
    return frame


def generate_synthetic_dataset(
    output_root: Path = Path(__file__).resolve().parents[1] / "data" / "raw",
    images_per_class: int = 60,
    seed: int = 42,
) -> None:
    """Create clearly synthetic authentic/spliced images for pipeline smoke tests."""
    if images_per_class < 2:
        raise ValueError("At least two synthetic images per class are required.")
    authentic_dir, tampered_dir = output_root / "Au", output_root / "Tp"
    authentic_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(images_per_class):
        width, height = 256, 192
        base_color = tuple(rng.randint(50, 210) for _ in range(3))
        image = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(image)
        for shape_index in range(10):
            x, y = rng.randint(0, width - 40), rng.randint(0, height - 40)
            color = tuple(rng.randint(0, 255) for _ in range(3))
            draw.rectangle((x, y, x + rng.randint(10, 45), y + rng.randint(10, 45)), fill=color)
        authentic_path = authentic_dir / f"synthetic_au_{index:04d}.jpg"
        image.save(authentic_path, "JPEG", quality=rng.randint(82, 96))

        tampered = image.copy()
        patch = Image.new(
            "RGB",
            (64, 64),
            tuple(rng.randint(0, 255) for _ in range(3)),
        )
        patch_draw = ImageDraw.Draw(patch)
        patch_draw.ellipse((5, 5, 58, 58), fill=tuple(rng.randint(0, 255) for _ in range(3)))
        x, y = rng.randint(0, width - 64), rng.randint(0, height - 64)
        tampered.paste(patch, (x, y))
        tampered_path = tampered_dir / f"synthetic_tp_{index:04d}.jpg"
        tampered.save(tampered_path, "JPEG", quality=rng.randint(82, 96))
