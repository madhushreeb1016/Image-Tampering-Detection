"""Command-line image forgery prediction."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from src.config import DEFAULT_ELA_QUALITY, OUTPUTS_DIR
from src.predictor import predict_image
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse the image prediction command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image file to analyze.")
    parser.add_argument("--quality", type=int, default=DEFAULT_ELA_QUALITY)
    parser.add_argument("--sensitivity", type=int, default=95, help="ELA percentile from 70 to 99.")
    return parser.parse_args()


def main() -> None:
    """Predict an image and save the main visualizations."""
    configure_logging()
    args = parse_args()
    try:
        result = predict_image(args.image, quality=args.quality, sensitivity=args.sensitivity)
    except (OSError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Could not analyze image: {error}") from error

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", args.image.stem)
    result.ela_image.save(OUTPUTS_DIR / f"{safe_stem}_ela.png")
    result.heatmap.save(OUTPUTS_DIR / f"{safe_stem}_heatmap.png")
    result.annotated_image.save(OUTPUTS_DIR / f"{safe_stem}_annotated.png")
    print(f"Verdict: {result.verdict}")
    print(f"Tampering probability: {result.tampering_probability:.1f}%")
    print(f"Suspicious regions: {len(result.suspicious_regions)}")
    print(f"Saved ELA, heatmap, and annotated image to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
