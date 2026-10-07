"""Basic edge-case checks for ELA generation, features, and inference."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.ela import generate_ela
from src.features import extract_features
from src.predictor import predict_image


def _write_model_bundle(model_dir, sample_image: Image.Image) -> None:
    """Write a deterministic tiny model fixture for prediction tests."""
    ela = generate_ela(sample_image)
    values, names = extract_features(ela)
    training_data = np.vstack((values, np.asarray(values) + 1.0, np.asarray(values) + 2.0))
    labels = np.array([0, 1, 1])
    scaler = StandardScaler().fit(training_data)
    classifier = DummyClassifier(strategy="prior").fit(scaler.transform(training_data), labels)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "best_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(names, model_dir / "feature_names.joblib")


def test_ela_shape_matches_input() -> None:
    image = Image.new("RGB", (80, 40), (120, 80, 20))
    assert generate_ela(image).size == image.size


@pytest.mark.parametrize(
    "image",
    (
        Image.new("L", (32, 24), 120),
        Image.new("RGBA", (32, 24), (20, 80, 140, 100)),
        Image.new("RGB", (32, 24), (90, 120, 150)),
    ),
)
def test_ela_supports_grayscale_rgba_and_png_images(image: Image.Image, tmp_path) -> None:
    path = tmp_path / "image.png"
    image.save(path)
    assert generate_ela(path).size == image.size


def test_very_small_image_is_supported() -> None:
    assert generate_ela(Image.new("RGB", (1, 1), (0, 0, 0))).size == (1, 1)


def test_very_large_image_is_resized() -> None:
    assert generate_ela(Image.new("RGB", (1800, 900), (80, 90, 100))).size == (1024, 512)


def test_feature_vector_has_fixed_length() -> None:
    first, first_names = extract_features(generate_ela(Image.new("RGB", (16, 16), "white")))
    second, second_names = extract_features(generate_ela(Image.new("RGB", (80, 45), "black")))
    assert len(first) == len(second) == len(first_names) == len(second_names)
    assert first_names == second_names


def test_prediction_returns_valid_verdict(tmp_path) -> None:
    image = Image.new("RGB", (40, 40), (20, 120, 200))
    _write_model_bundle(tmp_path, image)
    result = predict_image(image, model_dir=tmp_path)
    assert result.verdict in {"Authentic", "Likely Tampered"}
    assert 0.0 <= result.tampering_probability <= 100.0
    assert result.ela_image.size == image.size


def test_corrupt_file_raises_readable_image_error(tmp_path) -> None:
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        generate_ela(corrupt_path)
