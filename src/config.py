"""
Configuration settings for the Brain MRI Tumor Classification project.
Enforces fixed random seeds, label mappings, paths, and hyperparameters.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = BASE_DIR / "archive"
TRAIN_DATA_DIR = ARCHIVE_DIR / "Training"
TEST_DATA_DIR = ARCHIVE_DIR / "Testing"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Strict Label Mapping
# 0 = glioma, 1 = meningioma, 2 = pituitary, 3 = notumor
LABEL2ID = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)

# Dataset Split Configuration (Applied ONLY to Training directory)
TRAIN_SPLIT_RATIO = 0.80
VAL_SPLIT_RATIO = 0.20

# Input Dimensions & Normalization
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
INPUT_CHANNELS = 3

# Standard ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# DataLoader Settings
BATCH_SIZE = 32
NUM_WORKERS = 2

# Augmentation Parameters (Training Set Only)
AUGMENTATION_CONFIG = {
    "rotation_degrees": 10,
    "translate": (0.05, 0.05),
    "scale": (0.95, 1.05),
    "brightness": 0.10,
    "contrast": 0.10,
    "horizontal_flip_p": 0.50,
}

# Training Hyperparameters
BACKBONE_NAME = "EfficientNetB0"
DROPOUT_RATE = 0.30

# Phase 1: Feature Extraction (Backbone Frozen)
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

# Phase 2: Controlled Fine-Tuning (Upper Backbone Unfrozen)
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-4
FINE_TUNE_UNFREEZE_LAYERS = 25  # Unfreeze top layers (Block 7 & top conv)

# Callbacks
PATIENCE_EARLY_STOPPING = 4
PATIENCE_REDUCE_LR = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# Artifact File Paths
BEST_MODEL_PATH = MODELS_DIR / "best_brain_tumor_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_brain_tumor_model.keras"
CLASS_NAMES_PATH = RESULTS_DIR / "class_names.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
MODEL_METADATA_PATH = RESULTS_DIR / "model_metadata.json"
ACCURACY_PLOT_PATH = RESULTS_DIR / "training_accuracy.png"
LOSS_PLOT_PATH = RESULTS_DIR / "training_loss.png"
DATASET_METADATA_PATH = RESULTS_DIR / "dataset_metadata.json"

# Step 4 Test Evaluation Paths
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORM_PATH = RESULTS_DIR / "confusion_matrix_normalized.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
TEST_PREDICTIONS_CSV_PATH = RESULTS_DIR / "test_predictions.csv"
MISCLASSIFIED_CSV_PATH = RESULTS_DIR / "misclassified_images.csv"
MISCLASSIFIED_SAMPLES_PATH = RESULTS_DIR / "misclassified_samples.png"
FINAL_TEST_REPORT_PATH = RESULTS_DIR / "final_test_report.txt"
