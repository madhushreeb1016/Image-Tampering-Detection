# Implementation Plan for ELA-Based Image Tampering Detection

## 1. Project Overview
This project builds an image tampering detection system that uses Error Level Analysis (ELA) and a classical machine learning pipeline to identify suspicious edits in images. The system analyzes uploaded images, extracts ELA-based features, runs a trained classifier, and highlights likely manipulated regions.

The solution includes:
- dataset preparation and labeling
- ELA image generation
- feature extraction
- model training and comparison
- model persistence
- prediction for new inputs
- suspicious region localization
- Streamlit-based user interface
- evaluation and reporting

## 2. Objectives
- Detect whether an image is likely authentic or tampered.
- Estimate tampering probability.
- Highlight suspicious locations in the image.
- Provide an easy-to-use demo interface for image uploads.
- Train and compare multiple models to select the strongest one.
- Keep the pipeline simple, interpretable, and CPU-friendly.

## 3. Scope
The first implementation will focus on a practical, educational pipeline using:
- JPEG recompression-based ELA features
- handcrafted numeric descriptors
- classical ML classifiers
- visual overlays for detection explanation

This is intended as a research/demo-oriented tool and not forensic proof.

## 4. High-Level Workflow
1. Collect authentic and tampered images.
2. Read and prepare data.
3. Generate ELA images from the original files.
4. Extract discriminative features from ELA maps.
5. Train and compare several ML models.
6. Select the best-performing model.
7. Save the trained model, scaler, and feature metadata.
8. Predict on new image uploads.
9. Detect suspicious regions and mark them visually.
10. Show results in a Streamlit application.
11. Store evaluation metrics and output plots.

## 5. End-to-End Process

### Phase 1: Setup and Environment Preparation
- Create a Python virtual environment.
- Install the required dependencies from `requirements.txt`.
- Check project folder structure.
- Confirm support for image processing and ML libraries.

Expected result:
- A working development environment with all required packages installed.

### Phase 2: Dataset Collection and Organization
- Prepare authentic images in `data/raw/Au/`.
- Prepare tampered images in `data/raw/Tp/`.
- Support common image formats such as JPG, JPEG, PNG, TIFF, and BMP.
- Keep the dataset organized and separate from synthetic demo files.

Expected result:
- A clean, labeled dataset ready for model training.

### Phase 3: Data Loading and Labeling
- Traverse image folders recursively.
- Read images as RGB.
- Assign labels:
  - 0 = authentic
  - 1 = tampered
- Build a dataset dataframe with at least:
  - filename
  - label
  - image path

Expected result:
- A structured dataset that the training pipeline can consume.

### Phase 4: Error Level Analysis (ELA)
- Re-save each image as JPEG at a chosen compression quality.
- Compare the recompressed image to the original image.
- Compute the pixel-wise difference to produce an ELA image.
- Use ELA to emphasize compression inconsistencies that may correspond to editing.

Expected result:
- One ELA image per training sample and per inference image.

### Phase 5: Feature Extraction
- Extract numeric features from each ELA image.
- Features may include:
  - mean intensity
  - variance/std dev
  - brightness distribution statistics
  - texture descriptors
  - edge density measures
  - percentile-based values
- Keep the feature list stable so the same ordering is used during training and prediction.

Expected result:
- A feature matrix and a feature-name list saved for reproducibility.

### Phase 6: Dataset Assembly
- Aggregate all feature vectors into a single dataset.
- Store the dataset in CSV format.
- Ensure the dataset contains both authentic and tampered examples.
- Validate class balance and sample count.

Expected result:
- A training dataset ready for model fitting.

### Phase 7: Model Training
Train multiple models and compare their performance:
- Logistic Regression
- Decision Tree
- Random Forest

Use a stratified train/test split and a fixed random seed. Standardize features before model fitting.

Expected result:
- Several trained classifiers with evaluation output.

### Phase 8: Model Evaluation and Selection
Evaluate each model using:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC curve

Choose the best model using the strongest relevant metric, typically F1-score.

Expected result:
- Best model selected and saved for future use.

### Phase 9: Artifact Persistence
Save the following files:
- best trained model
- scaler
- feature names
- performance comparison CSV
- confusion matrices
- ROC curve plot
- feature importance plot
- ELA sample figures

Expected result:
- All model artifacts needed for inference are stored in the correct folders.

### Phase 10: Prediction Pipeline for New Images
- Load a user-uploaded image or sample image.
- Generate its ELA representation.
- Extract the same feature vector used for training.
- Apply the saved scaler.
- Pass the features into the trained model.
- Return:
  - verdict
  - tampering probability
  - original image
  - ELA image
  - heatmap overlay
  - annotated output image

Expected result:
- A model prediction for new images.

### Phase 11: Suspicious Region Localization
- Convert the ELA image to grayscale.
- Apply a percentile threshold to isolate high-error pixels.
- Find connected regions or contours.
- Keep the largest bounding boxes.
- Draw the bounding boxes on the original image.

Expected result:
- Region proposals marking likely edited areas.

### Phase 12: Streamlit Application
Build the app interface in `app.py` to:
- upload an image or use a sample
- adjust ELA quality
- adjust region sensitivity
- display original, ELA, heatmap, and annotated images
- show model verdict and probability
- show model performance details in a second tab

Expected result:
- A simple interactive UI for end users.

### Phase 13: Validation and Testing
Test the complete workflow:
- small training run
- artifact generation
- prediction on sample image
- model comparison output
- UI rendering and error handling

Expected result:
- Verified end-to-end functionality.

### Phase 14: Documentation and Reporting
Document the process in the project README including:
- setup instructions
- training steps
- prediction steps
- limitations of ELA-based detection
- expected outputs and folder structure

Expected result:
- A clear project guide for future use and maintenance.

## 6. Module Responsibility Breakdown

### `src/` Modules
- dataset logic: image scanning and labeling
- ELA logic: recompression and difference generation
- feature extraction: handcrafted signal statistics from ELA images
- model training: train/evaluate/compare multiple classifiers
- prediction: run the saved model on new inputs
- utils: image reading, helper functions, logging, shared configuration

### Top-Level Scripts
- `train.py`: training, evaluation, artifact generation
- `predict.py`: command-line inference
- `app.py`: interactive Streamlit UI

## 7. Implementation Sequence
1. Set up environment and dependencies
2. Prepare dataset folders and labels
3. Build image loading utilities
4. Implement ELA generation
5. Implement feature extraction
6. Assemble training dataset
7. Train and compare models
8. Save artifacts
9. Implement prediction logic
10. Add suspicious region detection
11. Build Streamlit UI
12. Validate end-to-end pipeline
13. Document and finalize

## 8. Key Risks and Limitations
- ELA performs best on JPEG images and may be unreliable for PNGs or heavily processed images.
- Compression artifacts can be confused with true tampering.
- Feature-based ML models may not generalize well outside the training domain.
- Bounding boxes show suspicious regions, not exact manipulation masks.
- The trained model is a clue-based detector, not forensic proof.

## 9. Success Criteria
The project is considered complete when:
- authentic and tampered images are correctly labeled and processed
- ELA images are generated successfully
- features are extracted consistently
- models are trained and compared
- the best model is saved and reused
- new images can be classified with a probability
- suspicious regions are highlighted visually
- the Streamlit demo works end-to-end
- outputs and reports are generated and documented

## 10. Recommended Next Step
Start by implementing the dataset pipeline and ELA generation first. Once the ELA images and feature extraction are working reliably, move immediately to model training and validation. This order reduces debugging complexity and allows the rest of the project to build on a stable data-processing foundation.
