"""Shared paths and defaults for the ELA image forgery detector."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AUTHENTIC_DIR = RAW_DATA_DIR / "Au"
TAMPERED_DIR = RAW_DATA_DIR / "Tp"
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"

DEFAULT_ELA_QUALITY = 90
MAX_IMAGE_SIDE = 1024
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

