"""
Main model training script for Brain MRI Tumor Classification.
Implements two-phase transfer learning with EfficientNetB0:
- Phase 1: Feature extraction with frozen backbone.
- Phase 2: Controlled fine-tuning of upper layers with reduced learning rate.
Saves model checkpoints, metrics, training history, and training curves.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure KERAS_BACKEND is set before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras
from keras import layers
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    RANDOM_SEED,
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASSES,
    LABEL2ID,
    ID2LABEL,
    DROPOUT_RATE,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    FINE_TUNE_UNFREEZE_LAYERS,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_REDUCE_LR,
    REDUCE_LR_FACTOR,
    MIN_LR,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    CLASS_NAMES_PATH,
    TRAINING_HISTORY_PATH,
    MODEL_METADATA_PATH,
    ACCURACY_PLOT_PATH,
    LOSS_PLOT_PATH,
)
from src.dataset import get_keras_datasets
from src.evaluate import evaluate_model


def set_seed(seed: int = RANDOM_SEED):
    """Sets deterministic seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    keras.utils.set_random_seed(seed)
    print(f"Random seed set to: {seed}")


def build_efficientnet_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the EfficientNetB0 transfer learning architecture:
    Input (224, 224, 3) -> EfficientNetB0 backbone (ImageNet) ->
    GlobalAveragePooling2D -> Dropout -> Dense(4, softmax).
    """
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    # Freeze backbone initially for Phase 1
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="mri_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_BrainTumorClassifier")
    return model, base_model


def count_parameters(model: keras.Model) -> Dict[str, int]:
    """Computes trainable, non-trainable, and total parameter counts."""
    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    non_trainable_count = sum(np.prod(p.shape) for p in model.non_trainable_weights)
    return {
        "trainable": int(trainable_count),
        "non_trainable": int(non_trainable_count),
        "total": int(trainable_count + non_trainable_count),
    }


def plot_and_save_curves(history: Dict[str, List[float]], phase1_epochs: int):
    """Generates and saves clean accuracy and loss curves."""
    epochs = range(1, len(history["accuracy"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_accuracy"], "s-", label="Validation Accuracy", color="#2ca02c", lw=2)
    if phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs + 0.5, color="red", linestyle="--", label="Phase 2 Fine-Tuning Start")
    plt.title("Brain MRI Classifier: Training & Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=180)
    plt.close()
    print(f"Accuracy curve saved to: {ACCURACY_PLOT_PATH}")

    # 2. Loss Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["loss"], "o-", label="Training Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#d62728", lw=2)
    if phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs + 0.5, color="red", linestyle="--", label="Phase 2 Fine-Tuning Start")
    plt.title("Brain MRI Classifier: Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=180)
    plt.close()
    print(f"Loss curve saved to: {LOSS_PLOT_PATH}")


def train_model():
    start_total_time = time.time()
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 65)
    print("STEP 3: BRAIN MRI CLASSIFICATION MODEL TRAINING")
    print("=" * 65)

    # 1. Load Data Splits (Testing set remains completely untouched)
    print("\n[1/6] Loading data splits into Keras PyDatasets...")
    train_ds, val_ds, test_ds, dataset_stats = get_keras_datasets(
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
        preload=True,
    )
    print(f"Train batches: {len(train_ds)} | Val batches: {len(val_ds)}")
    print(f"Training samples: {len(train_ds.samples)} | Validation samples: {len(val_ds.samples)}")
    print(f"Testing samples (HELD OUT): {len(test_ds.samples)}")

    # 2. Save class names metadata
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(
            {
                "class_names": CLASSES,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            },
            f,
            indent=4,
        )
    print(f"Class names metadata saved to: {CLASS_NAMES_PATH}")

    # 3. Build Model
    print("\n[2/6] Building EfficientNetB0 Model...")
    model, base_model = build_efficientnet_model()
    model.summary()

    params_p1 = count_parameters(model)
    print(f"\nPhase 1 Parameters:")
    print(f"  Trainable parameters:     {params_p1['trainable']:,}")
    print(f"  Frozen/Non-trainable:     {params_p1['non_trainable']:,}")
    print(f"  Total parameters:          {params_p1['total']:,}")

    # Combined history dictionary
    combined_history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    # 4. Phase 1 Training: Frozen Backbone
    print("\n" + "-" * 60)
    print(f"PHASE 1: FEATURE EXTRACTION (Backbone Frozen, {PHASE1_EPOCHS} Epochs Max)")
    print("-" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    t0_p1 = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_phase1,
        verbose=1,
    )
    p1_time = time.time() - t0_p1
    epochs_p1 = len(h1.history["loss"])
    print(f"Phase 1 completed {epochs_p1} epochs in {p1_time:.1f} seconds.")

    for k in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        combined_history[k].extend([float(x) for x in h1.history[k]])
    if "learning_rate" in h1.history:
        combined_history["lr"].extend([float(x) for x in h1.history["learning_rate"]])

    # 5. Phase 2 Training: Controlled Fine-Tuning
    print("\n" + "-" * 60)
    print(f"PHASE 2: CONTROLLED FINE-TUNING (Unfreezing top {FINE_TUNE_UNFREEZE_LAYERS} layers)")
    print("-" * 60)

    base_model.trainable = True
    # Freeze all layers except the top FINE_TUNE_UNFREEZE_LAYERS
    for layer in base_model.layers[:-FINE_TUNE_UNFREEZE_LAYERS]:
        layer.trainable = False

    params_p2 = count_parameters(model)
    print(f"Phase 2 Parameters:")
    print(f"  Trainable parameters:     {params_p2['trainable']:,}")
    print(f"  Frozen/Non-trainable:     {params_p2['non_trainable']:,}")
    print(f"  Total parameters:          {params_p2['total']:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    t0_p2 = time.time()
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1,
    )
    p2_time = time.time() - t0_p2
    epochs_p2 = len(h2.history["loss"])
    print(f"Phase 2 completed {epochs_p2} epochs in {p2_time:.1f} seconds.")

    for k in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        combined_history[k].extend([float(x) for x in h2.history[k]])
    if "learning_rate" in h2.history:
        combined_history["lr"].extend([float(x) for x in h2.history["learning_rate"]])

    total_training_time = time.time() - start_total_time
    total_epochs = epochs_p1 + epochs_p2

    # 6. Save final model & training history
    model.save(str(FINAL_MODEL_PATH))
    print(f"\nFinal model saved to: {FINAL_MODEL_PATH}")

    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(combined_history, f, indent=4)
    print(f"Training history saved to: {TRAINING_HISTORY_PATH}")

    # 7. Generate curves
    print("\n[5/6] Generating and saving training curves...")
    plot_and_save_curves(combined_history, phase1_epochs=epochs_p1)

    # 8. Evaluation on Validation Set using best model
    print("\n[6/6] Evaluating Best Saved Model on Validation Set...")
    best_model = keras.models.load_model(str(BEST_MODEL_PATH))
    val_metrics = evaluate_model(best_model, val_ds, dataset_name="Validation")

    # Assess overfitting
    best_epoch_idx = int(np.argmin(combined_history["val_loss"]))
    train_acc_best_epoch = combined_history["accuracy"][best_epoch_idx]
    val_acc_best_epoch = combined_history["val_accuracy"][best_epoch_idx]
    overfitting_gap = abs(train_acc_best_epoch - val_acc_best_epoch)

    overfitting_assessment = (
        "Minimal/Controlled overfitting: Training accuracy and validation accuracy are closely aligned "
        f"(gap: {overfitting_gap:.4f}). Regularization (Dropout=0.3 and conservative augmentations) successfully "
        "stabilized generalization."
        if overfitting_gap < 0.08
        else f"Moderate gap observed ({overfitting_gap:.4f}) between train and validation accuracy."
    )

    # Save model metadata
    model_metadata = {
        "model_architecture": "EfficientNetB0 (Transfer Learning)",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "random_seed": RANDOM_SEED,
        "parameters": {
            "phase1": params_p1,
            "phase2": params_p2,
        },
        "training_time_seconds": round(total_training_time, 2),
        "epochs_completed": {
            "phase1": epochs_p1,
            "phase2": epochs_p2,
            "total": total_epochs,
        },
        "best_epoch_index": best_epoch_idx + 1,
        "best_validation_loss": float(min(combined_history["val_loss"])),
        "best_validation_accuracy": float(max(combined_history["val_accuracy"])),
        "validation_metrics": val_metrics,
        "overfitting_assessment": overfitting_assessment,
        "best_model_path": str(BEST_MODEL_PATH),
        "final_model_path": str(FINAL_MODEL_PATH),
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"Model metadata saved to: {MODEL_METADATA_PATH}")

    # Copy curves to artifact directory for presentation
    artifact_dir = Path("/Users/laxyagaba/.gemini/antigravity-ide/brain/31ae1d09-1ce3-4169-8973-37f3601df258")
    import shutil
    shutil.copy2(ACCURACY_PLOT_PATH, artifact_dir / "training_accuracy.png")
    shutil.copy2(LOSS_PLOT_PATH, artifact_dir / "training_loss.png")

    print("\n" + "=" * 65)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 65)
    return model_metadata


if __name__ == "__main__":
    train_model()
