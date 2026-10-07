"""Train and compare simple classifiers on ELA-derived CASIA features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.config import (
    AUTHENTIC_DIR,
    DEFAULT_ELA_QUALITY,
    FEATURES_CSV,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS,
    TAMPERED_DIR,
)
from src.dataset import build_features_csv, generate_synthetic_dataset
from src.ela import ela_heatmap, generate_ela
from src.utils import configure_logging, open_rgb_image

LOGGER = logging.getLogger(__name__)


def _save_ela_samples() -> None:
    """Save a source/ELA/heatmap grid using up to three images per class."""
    class_paths = tuple(
        sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )[:3]
        for directory in (AUTHENTIC_DIR, TAMPERED_DIR)
    )
    items = [(path, label) for label, paths in enumerate(class_paths) for path in paths]
    if not items:
        return
    figure, axes = plt.subplots(len(items), 3, figsize=(10, max(2.4, len(items) * 2.3)))
    if len(items) == 1:
        axes = [axes]
    for row, (path, label) in enumerate(items):
        original = open_rgb_image(path)
        ela = generate_ela(original)
        heatmap, _ = ela_heatmap(ela, original)
        for column, (image, title) in enumerate(
            ((original, "Original"), (ela, "ELA"), (heatmap, "Heatmap"))
        ):
            axes[row][column].imshow(image)
            axes[row][column].set_title(f"{title} — {'Authentic' if label == 0 else 'Tampered'}")
            axes[row][column].axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "ela_samples.png", dpi=150)
    plt.close(figure)


def train_models(
    features_csv: Path = FEATURES_CSV,
    output_dir: Path = OUTPUTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Fit three models, save comparison plots, and serialize the best model."""
    frame = pd.read_csv(features_csv)
    feature_names = [column for column in frame.columns if column not in {"filename", "label", "ela_quality"}]
    if not feature_names or frame["label"].nunique() != 2:
        raise ValueError("Training data must contain ELA features and both class labels.")
    x = frame[feature_names].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)
    if min(pd.Series(y).value_counts()) < 2:
        raise ValueError("At least two readable images from each class are required.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    scaler = StandardScaler().fit(x_train)
    scaled_train, scaled_test = scaler.transform(x_train), scaler.transform(x_test)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }
    results: list[dict[str, float | str]] = []
    probabilities: dict[str, object] = {}
    fitted_models: dict[str, object] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for name, model in models.items():
        model.fit(scaled_train, y_train)
        predictions = model.predict(scaled_test)
        probability = model.predict_proba(scaled_test)[:, 1]
        probabilities[name] = probability
        fitted_models[name] = model
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
            xticklabels=("Authentic", "Tampered"),
            yticklabels=("Authentic", "Tampered"),
        )
        axis.set(title=f"{name} confusion matrix", xlabel="Predicted", ylabel="Actual")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(figure)

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    synthetic_flags = frame["filename"].map(
        lambda filename: Path(str(filename)).name.startswith("synthetic_")
    )
    if synthetic_flags.all():
        data_source = "Synthetic demo (not real-world)"
    elif synthetic_flags.any():
        data_source = "Mixed dataset (includes synthetic images)"
    else:
        data_source = "User-provided dataset"
    comparison.insert(1, "data_source", data_source)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure, axis = plt.subplots(figsize=(8, 5))
    comparison.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", ax=axis
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Model metric comparison")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for name, probability in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_test, probability)
        axis.plot(false_positive, true_positive, label=name)
    axis.plot((0, 1), (0, 1), linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(figure)

    random_forest = fitted_models["Random Forest"]
    importances = pd.Series(random_forest.feature_importances_, index=feature_names)
    top_features = importances.nlargest(15).sort_values()
    figure, axis = plt.subplots(figsize=(8, 6))
    top_features.plot(kind="barh", ax=axis)
    axis.set(title="Random Forest: top 15 features", xlabel="Importance")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(figure)
    _save_ela_samples()

    best_name = str(comparison.iloc[0]["model"])
    joblib.dump(fitted_models[best_name], models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(feature_names, models_dir / "feature_names.joblib")
    joblib.dump(best_name, models_dir / "model_name.joblib")
    LOGGER.info("Saved best model (%s, by F1) and scaler in %s", best_name, models_dir)
    return comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum images per class.")
    parser.add_argument("--force-features", action="store_true", help="Recompute cached ELA features.")
    parser.add_argument(
        "--synthetic-demo", action="store_true",
        help="Generate a labeled synthetic set in data/raw for pipeline smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Build features and train the comparison models."""
    configure_logging()
    args = parse_args()
    if args.synthetic_demo:
        existing_images = [
            path
            for directory in (AUTHENTIC_DIR, TAMPERED_DIR)
            for path in directory.rglob("*")
            if directory.exists() and path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if existing_images:
            raise FileExistsError(
                "Synthetic generation only writes to empty class folders; "
                "move existing images before using --synthetic-demo."
            )
        generate_synthetic_dataset()
        LOGGER.warning("Training on SYNTHETIC demo data; metrics are not real-world results.")
    try:
        build_features_csv(limit=args.limit, force=args.force_features)
    except FileNotFoundError as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
    train_models()


if __name__ == "__main__":
    main()
