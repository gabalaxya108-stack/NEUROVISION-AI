# Brain MRI Tumor Classification Project

A deep-learning classification pipeline for multi-class Brain MRI scans using transfer learning with **EfficientNetB0**.

---

## 1. Project Overview

This project classifies axial and coronal brain MRI scans into four categories:

- **`0`**: `glioma` (Glioma tumor)
- **`1`**: `meningioma` (Meningioma tumor)
- **`2`**: `pituitary` (Pituitary tumor)
- **`3`**: `notumor` (Healthy brain tissue / No tumor)

The project incorporates:

- **Reproducible Data Pipeline:** Fixed random seeds ($42$), programmatic 80/20 train-validation splitting, and an untouched 1,600-image test set.
- **Deterministic Preprocessing:** Aspect-ratio preserving letterbox resizing to $224 \times 224 \times 3$ with zero/black padding.
- **Two-Phase Transfer Learning:** Feature extraction with a frozen backbone followed by controlled fine-tuning of upper layers.
- **Standalone Prediction CLI:** Instant, single-image classification with full class probability breakdowns.

---

## 2. Directory Structure

```text
brain-tumor-project/
├── archive/
│   ├── Training/                # 5,600 original training images (4 classes, balanced)
│   └── Testing/                 # 1,600 untouched benchmark test images
├── models/
│   ├── best_brain_tumor_model.keras   # Best saved model checkpoint (96.25% val acc)
│   └── final_brain_tumor_model.keras  # Model checkpoint at final epoch
├── results/
│   ├── class_names.json         # Label mapping dictionary
│   ├── dataset_metadata.json    # Split statistics and data distributions
│   ├── model_metadata.json      # Training parameters, metrics, and parameters
│   ├── training_history.json    # Epoch-by-epoch loss & accuracy history
│   ├── training_accuracy.png    # Training vs. validation accuracy curve
│   ├── training_loss.png        # Training vs. validation loss curve
│   ├── confusion_matrix.png     # Test set confusion matrix (raw counts)
│   ├── confusion_matrix_normalized.png  # Normalized confusion matrix (percentages)
│   ├── misclassified_samples.png # Visual grid of misclassified test scans
│   ├── test_predictions.csv     # Complete test predictions with probabilities
│   ├── misclassified_images.csv # Filtered CSV containing only incorrect predictions
│   ├── classification_report.txt# Test classification report
│   ├── final_test_report.txt    # Comprehensive final evaluation report
│   ├── per_class_error_analysis.csv     # Error, recall, precision, and confidence per class
│   ├── misclassification_pairs.csv      # Error frequency for all actual -> predicted pairs
│   ├── misclassification_pairs.png      # Bar chart of directional misclassifications
│   ├── high_confidence_errors.csv       # Incorrect predictions with confidence >= 0.80
│   ├── gradcam/                 # Generated Grad-CAM heatmaps and composites
│   └── error_gradcam/           # Grad-CAM visualizations on misclassified test samples
├── src/
│   ├── config.py                # Hyperparameters, paths, and class labels
│   ├── dataset.py               # Deterministic tf.data pipelines and splitting
│   ├── preprocessing.py         # Aspect-ratio preserving letterbox resize (224x224)
│   ├── train.py                 # Two-phase transfer learning & fine-tuning script
│   ├── evaluate.py              # Untouched test evaluation & metrics generation
│   ├── predict.py               # Standalone prediction CLI script
│   ├── explain.py               # Grad-CAM explainability and heatmap generator
│   └── input_guard.py           # Multi-stage Brain MRI input guardrail
├── backend/                     # FastAPI backend application
│   ├── main.py                  # API endpoints (/health, /model-info, /predict, /explain)
│   ├── schemas.py               # Pydantic request & response contracts
│   ├── services/                # Model, prediction, and Grad-CAM services
│   └── test_api.py              # Automated API parity & test suite
├── frontend/                    # Vite + React research web application
│   ├── src/
│   │   ├── pages/               # Dashboard, Analyze MRI, Performance, Error Analysis, Methodology
│   │   ├── components/          # Sidebar, Header, PredictionCard, GradcamViewer, Modals
│   │   └── styles/              # Design tokens, layout, and component CSS
├── samples/                     # Academic verification MRI scans
│   ├── sample_glioma.jpg        # Curated Glioma test scan
│   ├── sample_meningioma.jpg    # Curated Meningioma test scan
│   ├── sample_pituitary.jpg     # Curated Pituitary test scan
│   └── sample_notumor.jpg       # Curated No Tumor test scan
└── requirements.txt             # Pinned project dependencies
```

---

## 3. How to Run Single-Image Predictions

The standalone prediction script [`src/predict.py`](src/predict.py) allows you to classify any brain MRI image directly from the terminal.

### Basic Usage

To run inference on an image:

```bash
python src/predict.py <path_to_image>
```

**Example:**

```bash
python src/predict.py samples/demo_external_mri.jpg
```

**Console Output:**

```text
========================================
BRAIN MRI CLASSIFICATION
========================================

Image:
demo_external_mri.jpg

Prediction:
notumor

Class probabilities:

Glioma         1.04%
Meningioma     0.37%
Pituitary      5.52%
No Tumor      93.07%

Model:
EfficientNetB0

========================================
Notice: Academic research tool. Class probabilities do
not constitute a medical diagnosis or clinical certainty.
========================================
```

---

### Saving Prediction Output to JSON

You can export the prediction and class probability distribution directly to a JSON file using the `--save-json` flag:

```bash
python src/predict.py path/to/mri.jpg --save-json results/my_prediction.json
```

**Output JSON Structure:**

```json
{
    "image": "mri.jpg",
    "predicted_class": "glioma",
    "probabilities": {
        "glioma": 0.9421,
        "meningioma": 0.0412,
        "pituitary": 0.0051,
        "notumor": 0.0116
    }
}
```

---

### Pipeline Verification

To verify that the model checkpoint, preprocessing modules, and dependencies load properly without specifying an image:

```bash
python src/predict.py --verify
```

---

## 4. How to Generate Grad-CAM Explanations

The explainability script [`src/explain.py`](src/explain.py) generates visual Grad-CAM heatmaps showing which regions of an MRI scan contributed most strongly to the model's classification.

### Single Image Explanation

```bash
python src/explain.py <path_to_image>
```

**Example:**

```bash
python src/explain.py archive/Testing/pituitary/Te-pi_1.jpg
```

**Console Output:**

```text
========================================
GRAD-CAM EXPLANATION
========================================

Image:
Te-pi_1.jpg

Prediction:
Pituitary

Class probabilities:

Glioma         0.00%
Meningioma     0.12%
Pituitary     99.88%
No Tumor       0.00%

Grad-CAM:
results/gradcam/gradcam_Te-pi_1.png

========================================
Interpretation: Highlighted regions indicate visual features
that contributed to the model's prediction. Not clinical evidence.
========================================
```

### Batch Directory Processing

To process a directory of MRI scans:

```bash
python src/explain.py --directory path/to/folder --max-images 10
```

### Output Files Generated in `results/gradcam/`

- `original_<filename>.png`: Clean original MRI image.
- `heatmap_<filename>.png`: Colored Grad-CAM activation heatmap.
- `overlay_<filename>.png`: 60/40 alpha-blended overlay.
- `gradcam_<filename>.png`: Presentation-quality 3-panel visualization with prediction card and research notice.

---

## 5. Model Evaluation Summary

| Dataset Split | Sample Count | Accuracy | Macro F1-Score | Status |
| --- | :---: | :---: | :---: | :---: |
| **Validation Set** | 1,120 | **96.25%** | **96.23%** | Used for checkpoint selection & early stopping |
| **Test Set** | 1,600 | **92.81%** | **92.63%** | Strictly held out until final evaluation |

---

## 6. Comprehensive Error and Confidence Analysis (Step 7)

To regenerate or rerun the complete error analysis on the test set:

```bash
python run_step7_analysis.py
```

### Key Error Analysis Insights

1. **Primary Misclassification Mode:** `glioma → meningioma` accounts for 48.70% (56/115) of all errors. Intra-axial gliomas with dense margins or peripheral locations can mimic extra-axial meningiomas.
2. **Confidence Separation:**
   - **Correct Predictions:** Mean confidence of **97.29%** (Median: **99.85%**).
   - **Incorrect Predictions:** Mean confidence of **75.91%** (Median: **78.84%**).
3. **High-Confidence Errors:** 56 cases (48.7% of errors) had confidence $\ge 0.80$, with the highest being `Te-gl_109.jpg` (99.93%).
4. **Borderline Uncertainty:** 46 cases had confidence $< 0.60$ (67.4% of which were incorrect), demonstrating the utility of confidence flagging for triage.
5. **Grad-CAM on Errors:** Layer activations in misclassified cases reveal model attention focused on peripheral dura-like contrast enhancements or skull margins.

---

## 7. Inference Backend API (Step 8)

The project includes a lightweight, production-grade inference REST API built with **FastAPI** and **Uvicorn**.

### 7.1 Backend Requirements & Installation

The backend uses Python 3.10+ and the existing project environment with FastAPI, Uvicorn, Python-Multipart, and HTTPX:

```bash
pip install -r requirements.txt
```

### 7.2 How to Start the API Server

Start the local server with auto-reload:

```bash
uvicorn backend.main:app --reload
```

Or via Python module:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Once running, navigate to:

- **Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc Documentation:** `http://127.0.0.1:8000/redoc`

### 7.3 API Endpoints

| Method | Endpoint | Description | Response / Behavior |
| --- | --- | --- | --- |
| `GET` | `/health` | Service health & model state | `{"status": "ok", "model_loaded": true}` |
| `GET` | `/model-info` | Architecture & benchmark metrics | Model parameters, 4 target classes, 92.81% test accuracy |
| `POST` | `/predict` | Single-image MRI classification | Returns predicted class, model confidence score, class probabilities |
| `POST` | `/explain` | Grad-CAM visual explanation | Returns prediction + base64 Data URIs of overlay, heatmap, & 3-panel figure |

### 7.4 Example Requests & Responses

#### A. Health Check

```bash
curl -X GET http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### B. Model Metadata

```bash
curl -X GET http://127.0.0.1:8000/model-info
```

```json
{
  "model": "EfficientNetB0",
  "input_shape": [224, 224, 3],
  "classes": [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
  ],
  "test_accuracy": 0.9281,
  "purpose": "educational/research image classification"
}
```

#### C. Predict Endpoint (`POST /predict`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@samples/demo_external_mri.jpg"
```

```json
{
  "predicted_class": "notumor",
  "confidence": 0.9307,
  "probabilities": {
    "glioma": 0.0104,
    "meningioma": 0.0037,
    "pituitary": 0.0552,
    "notumor": 0.9307
  }
}
```

#### D. Grad-CAM Explanation Endpoint (`POST /explain`)

```bash
curl -X POST http://127.0.0.1:8000/explain \
  -F "file=@samples/demo_external_mri.jpg"
```

```json
{
  "predicted_class": "notumor",
  "confidence": 0.9307,
  "probabilities": {
    "glioma": 0.0104,
    "meningioma": 0.0037,
    "pituitary": 0.0552,
    "notumor": 0.9307
  },
  "gradcam_overlay_base64": "data:image/png;base64,...",
  "gradcam_heatmap_base64": "data:image/png;base64,...",
  "gradcam_panel_base64": "data:image/png;base64,..."
}
```

### 7.5 Running Automated API Tests

To verify all endpoints, error handling, and inference parity against `src/predict.py`:

```bash
python backend/test_api.py
```

---

## 8. Interactive Frontend Web Application (Step 9)

A research-grade single-page application built with **React** and **Vite**, featuring a clean clinical/academic interface with Vanilla CSS.

### 8.1 Frontend Features

1. **Interactive Dashboard:** Project overview, four core benchmark statistics (explicitly highlighting the 92.81% test-set accuracy), and quick navigation.
2. **Analyze MRI Scanner:** Drag-and-drop file uploader, curated sample tray for instantaneous 1-click evaluation, image preview with dimensions, and dynamic loading indicator.
3. **Comprehensive Prediction Results:** Clear prediction winner badge, model confidence score, and horizontal distribution bars for all 4 classes.
4. **Explainable AI (Grad-CAM):** Multi-tab visualization viewer (Overlay, Heatmap, Original MRI, Complete 3-Panel Figure), full-resolution zoom modal, and interpretability guidance.
5. **Model Performance Hub:** Displays aggregate metrics (92.81% accuracy, 93.13% precision, 92.63% F1), per-class table, and interactive confusion matrices.
6. **Error Analysis Gallery:** Documents 115 test failure modes, misclassification frequencies (e.g. 56 Glioma &rarr; Meningioma), confidence histograms, and representative error scans with individual [ View Grad-CAM ] modal inspections.
7. **Methodology Pipeline:** Visual 9-step flowchart detailing the deep learning lifecycle and benchmark data leakage considerations.

### 8.2 How to Start the Frontend

In a separate terminal window:

```bash
cd frontend
npm install
npm run dev
```

The application will be live at:

- **Local Web App URL:** `http://127.0.0.1:5173` (or `http://localhost:5173`)

### 8.3 Full-Stack Local Execution Summary

To run the full stack simultaneously:

```bash
# Terminal 1: Backend API (FastAPI)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend Web App (Vite)
cd frontend && npm run dev
```

---

## 9. Brain MRI Input Guardrail (Section 19)

A mandatory, multi-stage input validation layer implemented in [`src/input_guard.py`](src/input_guard.py) safeguards the EfficientNetB0 tumor classifier.

Arbitrary non-medical images (e.g., selfies, photographs, documents, screenshots, non-brain radiographs, corrupted files) are intercepted and rejected **before** reaching preprocessing, the neural network, or Grad-CAM.

```text
USER IMAGE
    ↓
INPUT VALIDATION / MRI GUARDRAIL (src/input_guard.py)
    ↓
[If Valid Brain MRI]  → Proceed to Letterbox Preprocessing → EfficientNetB0 Classifier → Grad-CAM
[If Invalid Non-MRI]  → STOP. Tumor classifier NEVER executes.
[If Uncertain Input]  → STOP. Tumor classifier NEVER executes.
```

### 9.1 Guardrail Validation Criteria

1. **Format & Decoding:** Validates PIL decoding and resolution ($\ge 80 \times 80$ px, aspect ratio $\le 2.6$).
2. **Chromatic Saturation:** Evaluates mean color saturation across non-black tissue pixels to intercept color photographs, art, and selfies.
3. **Background Dark-Framing:** Inspects corner patches and perimeter borders to reject paper documents, book scans, and white-background screenshots.
4. **Tissue Intensity Entropy:** Verifies soft-tissue continuous grayscale gradients against bimodal text documents and flat graphics.
5. **Cranial Morphology & Centeredness:** Segments foreground cranial area (12% to 92% occupancy) and verifies centroid alignment.
6. **Rectilinear Edge Filtering:** Detects high concentrations of orthogonal grid lines characteristic of typography, tables, and UI windows.

### 9.2 Critical Intercept Verification Test

To verify that rejected and uncertain images never reach the tumor classifier:

```bash
python test_guardrail_pipeline.py
```

- **Audit Results:** Saved to [`results/input_guardrail_test.json`](results/input_guardrail_test.json).
- **Formal Guarantee:** Model classifier call count is strictly **0** for all rejected or uncertain inputs.

---

## 10. Educational and Research Notice

This software is developed strictly for educational, scientific, and technical evaluation purposes. The class probabilities and output predictions do **not** constitute medical advice, clinical diagnosis, or treatment decisions.
