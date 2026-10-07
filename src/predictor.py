"""Image-level prediction and localization of high-error ELA regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw

from src.config import DEFAULT_ELA_QUALITY, MODELS_DIR
from src.ela import ela_heatmap, generate_ela
from src.features import extract_features
from src.utils import ImageInput, open_rgb_image


@dataclass
class PredictionResult:
    """Images, verdict, confidence, and bounding boxes from one prediction."""

    verdict: str
    tampering_probability: float
    original: Image.Image
    ela_image: Image.Image
    heatmap: Image.Image
    overlay: Image.Image
    annotated_image: Image.Image
    suspicious_regions: list[tuple[int, int, int, int]]


def _find_suspicious_regions(
    ela_image: Image.Image,
    sensitivity: int,
    minimum_area: int = 12,
    maximum_regions: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Find the largest connected regions above an ELA percentile threshold."""
    gray = cv2.cvtColor(np.asarray(ela_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    threshold = max(float(np.percentile(gray, sensitivity)), 0.0)
    binary = (gray > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < minimum_area:
            continue
        boxes.append(tuple(int(value) for value in cv2.boundingRect(contour)))
        if len(boxes) == maximum_regions:
            break
    return boxes


def predict_image(
    path_or_pil: ImageInput,
    quality: int = DEFAULT_ELA_QUALITY,
    sensitivity: int = 95,
    model_dir: Path = MODELS_DIR,
) -> PredictionResult:
    """Predict authenticity and return ELA visualizations and suspicious boxes."""
    if not 70 <= quality <= 95:
        raise ValueError("Prediction ELA quality must be between 70 and 95.")
    if not 70 <= sensitivity <= 99:
        raise ValueError("Sensitivity percentile must be between 70 and 99.")
    model_path = model_dir / "best_model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    names_path = model_dir / "feature_names.joblib"
    missing = [path.name for path in (model_path, scaler_path, names_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Trained model files are missing ({', '.join(missing)}). Run train.py first."
        )

    original = open_rgb_image(path_or_pil)
    ela_image = generate_ela(original, quality=quality)
    heatmap, overlay = ela_heatmap(ela_image, original)
    feature_values, feature_names = extract_features(ela_image)
    expected_names = joblib.load(names_path)
    if feature_names != expected_names:
        raise ValueError("Extracted features do not match the feature names in the trained model.")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    probability = float(model.predict_proba(scaler.transform([feature_values]))[0][1]) * 100.0

    suspicious_regions = _find_suspicious_regions(ela_image, sensitivity)
    annotated = original.copy()
    drawing = ImageDraw.Draw(annotated)
    for x, y, width, height in suspicious_regions:
        drawing.rectangle((x, y, x + width, y + height), outline=(255, 32, 32), width=3)
    return PredictionResult(
        verdict="Likely Tampered" if probability >= 50.0 else "Authentic",
        tampering_probability=probability,
        original=original,
        ela_image=ela_image,
        heatmap=heatmap,
        overlay=overlay,
        annotated_image=annotated,
        suspicious_regions=suspicious_regions,
    )
