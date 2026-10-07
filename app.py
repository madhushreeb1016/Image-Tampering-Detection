"""Streamlit demo for the ELA image forgery detector."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import OUTPUTS_DIR, SAMPLES_DIR
from src.predictor import predict_image

st.set_page_config(page_title="ELA Image Forgery Detector", page_icon="🔎", layout="wide")
st.title("🔎 ELA Image Forgery Detector")
st.write(
    "Explore whether an image shows signs of editing using Error Level Analysis "
    "(ELA) and a small machine-learning classifier."
)

with st.sidebar:
    st.header("Analysis settings")
    quality = st.slider("ELA JPEG quality", min_value=70, max_value=95, value=90)
    sensitivity = st.slider("Region sensitivity percentile", min_value=70, max_value=99, value=95)
    with st.expander("About / How it works"):
        st.write(
            "ELA saves a copy of the image as a JPEG and compares it with the original. "
            "Pixels that change more during this recompression appear brighter. An edited "
            "patch may have a different compression history, so its error can stand out. "
            "The classifier uses summary measurements of these patterns; this is a clue, "
            "not proof of editing."
        )
    st.caption(
        "ELA works best on JPEG images. PNG or heavily recompressed images may give "
        "less reliable results."
    )

analysis_tab, performance_tab = st.tabs(("Analyze image", "Model Performance"))
with analysis_tab:
    uploaded_file = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
    sample_images = sorted(
        path for path in SAMPLES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if SAMPLES_DIR.exists() else []
    sample_choice = st.selectbox(
        "Or try a sample image",
        options=["Choose a sample..."] + [path.name for path in sample_images],
    )

    selected_image = uploaded_file
    selected_name = uploaded_file.name if uploaded_file else ""
    if not uploaded_file and sample_choice != "Choose a sample...":
        selected_image = SAMPLES_DIR / sample_choice
        selected_name = sample_choice

    if selected_image:
        try:
            result = predict_image(
                selected_image,
                quality=quality,
                sensitivity=sensitivity,
            )
        except FileNotFoundError as error:
            st.error(f"{error} Train a model first with `python train.py --limit 50`.")
        except (OSError, ValueError) as error:
            st.error(f"Could not read or analyze {selected_name}: {error}")
        else:
            st.subheader(f"Analysis: {selected_name}")
            columns = st.columns(4)
            for column, label, image in zip(
                columns,
                ("Original", "ELA image", "Heatmap overlay", "Suspicious regions"),
                (result.original, result.ela_image, result.overlay, result.annotated_image),
            ):
                with column:
                    st.image(image, caption=label, use_column_width=True)
            is_tampered = result.verdict == "Likely Tampered"
            color = "#b42318" if is_tampered else "#18794e"
            st.markdown(
                f"<h2 style='color:{color};'>{result.verdict}</h2>",
                unsafe_allow_html=True,
            )
            st.progress(
                min(max(int(round(result.tampering_probability)), 0), 100),
                text=f"Estimated tampering probability: {result.tampering_probability:.1f}%",
            )
            st.caption(
                f"Marked {len(result.suspicious_regions)} high-error region(s). "
                "The boxes show candidate areas, not exact manipulation masks."
            )
    else:
        st.info("Upload an image or select one of the sample images to begin.")

with performance_tab:
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        source = str(comparison["data_source"].iloc[0]) if "data_source" in comparison else ""
        if "Synthetic" in source or "synthetic" in source:
            st.warning(
                "These are synthetic smoke-test metrics only; they do not measure "
                "real-world or CASIA performance."
            )
        st.subheader("Held-out test metrics")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("Train the models first to create the performance report.")
    for image_name, caption in (
        ("accuracy_comparison.png", "Model metric comparison"),
        ("confusion_matrix_logistic_regression.png", "Logistic Regression confusion matrix"),
        ("confusion_matrix_decision_tree.png", "Decision Tree confusion matrix"),
        ("confusion_matrix_random_forest.png", "Random Forest confusion matrix"),
        ("feature_importance.png", "Random Forest feature importance"),
        ("roc_curve.png", "ROC curves"),
    ):
        image_path = OUTPUTS_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_column_width=True)
