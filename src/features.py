"""Numeric ELA feature extraction for the lightweight classifier."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def extract_features(ela_image: Image.Image) -> tuple[list[float], list[str]]:
    """Extract a stable-length numeric feature vector and matching names."""
    rgb = np.asarray(ela_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    values: list[float] = []
    names: list[str] = []

    # Channel means and spreads can expose color-specific recompression residue.
    for index, channel_name in enumerate(("red", "green", "blue")):
        channel = rgb[:, :, index].astype(np.float64)
        values.extend((float(channel.mean()), float(channel.std())))
        names.extend((f"{channel_name}_mean", f"{channel_name}_std"))

    # Global intensity describes the overall amount and range of ELA error.
    values.extend(
        (
            float(gray.mean()),
            float(gray.std()),
            float(gray.max()),
            float(gray.var()),
        )
    )
    names.extend(("gray_mean", "gray_std", "gray_max", "gray_variance"))

    # Upper percentiles measure unusually bright compression-error pixels.
    for percentile in (90, 95, 99):
        values.append(float(np.percentile(gray, percentile)))
        names.append(f"gray_percentile_{percentile}")

    # Standardized moments summarize asymmetry and heavy-tailed error patterns.
    deviation = gray - gray.mean()
    std = float(gray.std())
    if std > 0:
        standardized = deviation / std
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness, kurtosis = 0.0, 0.0
    values.extend((skewness, kurtosis))
    names.extend(("gray_skewness", "gray_excess_kurtosis"))

    # A fixed high-intensity cutoff approximates the fraction of suspicious pixels.
    values.append(float(np.mean(gray > 50.0)))
    names.append("fraction_above_50")

    # A normalized histogram captures how ELA error is distributed by intensity.
    histogram, _ = np.histogram(gray, bins=16, range=(0, 256))
    histogram = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
    values.extend(histogram.tolist())
    names.extend(f"gray_histogram_bin_{index}" for index in range(16))

    # Block means capture local inconsistency, an important cue for splicing.
    block_grid = cv2.resize(gray.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
    block_means = block_grid.reshape(-1)
    values.extend((float(block_means.std()), float(block_means.max())))
    names.extend(("block_mean_std", "block_mean_max"))

    # Canny edge density describes the amount of sharp boundary structure in ELA.
    gray_uint8 = gray.astype(np.uint8)
    if min(gray_uint8.shape) < 3:
        edge_density = 0.0
    else:
        edges = cv2.Canny(gray_uint8, threshold1=50, threshold2=150)
        edge_density = float(np.count_nonzero(edges) / edges.size)
    values.append(edge_density)
    names.append("edge_density")

    return values, names
