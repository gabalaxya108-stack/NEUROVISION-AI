"""
Verification script for Step 2: Data Preprocessing and Validation Setup.
Validates the programmatic split, DataLoader functionality, tensor shapes,
and outputs a visual verification grid of preprocessed and augmented samples.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    CLASSES,
    LABEL2ID,
    ID2LABEL,
    IMAGE_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.dataset import get_data_loaders
from src.preprocessing import denormalize_tensor


def run_verification():
    print("=" * 65)
    print("STEP 2: PREPROCESSING & VALIDATION PIPELINE VERIFICATION")
    print("=" * 65)

    # 1. Initialize DataLoaders and gather split statistics
    print("\n[1/4] Generating programmatic splits & DataLoaders...")
    train_loader, val_loader, test_loader, stats = get_data_loaders(
        batch_size=16,
        num_workers=0,  # 0 for safe, reproducible verification
        seed=RANDOM_SEED,
    )

    print(f"Random Seed: {stats['random_seed']}")
    print(f"Label Mapping: {stats['label_mapping']}")
    print(f"Input Tensor Shape Target: {stats['input_shape']}")

    # 2. Display Sample Counts & Class Distribution
    print("\n[2/4] Split Statistics & Class Distribution:")
    for split_name in ["train", "validation", "test"]:
        s_data = stats["splits"][split_name]
        print(f"\n  Split: '{split_name.upper()}' (Total Samples: {s_data['total_samples']})")
        for cls in CLASSES:
            count = s_data["class_counts"][cls]
            pct = s_data["class_distribution_percent"][cls]
            print(f"    - {cls:12s} (ID: {LABEL2ID[cls]}): {count:5d} images ({pct:.2f}%)")

    # 3. Test Batch Extraction & Tensor Shape Validation
    print("\n[3/4] Testing Batch Extraction & Tensor Shapes:")
    train_batch_imgs, train_batch_labels, train_batch_files = next(iter(train_loader))
    val_batch_imgs, val_batch_labels, val_batch_files = next(iter(val_loader))
    test_batch_imgs, test_batch_labels, test_batch_files = next(iter(test_loader))

    print(f"  Train Batch Images Shape: {train_batch_imgs.shape} | Labels Shape: {train_batch_labels.shape}")
    print(f"  Val Batch Images Shape:   {val_batch_imgs.shape} | Labels Shape: {val_batch_labels.shape}")
    print(f"  Test Batch Images Shape:  {test_batch_imgs.shape} | Labels Shape: {test_batch_labels.shape}")

    assert train_batch_imgs.shape[1:] == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), "Unexpected train tensor shape!"
    assert val_batch_imgs.shape[1:] == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), "Unexpected val tensor shape!"
    assert test_batch_imgs.shape[1:] == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), "Unexpected test tensor shape!"
    print("  [SUCCESS] All batch dimensions match expected [Batch, 3, 224, 224].")

    # 4. Generate Visual Verification Grid (Saved to results/)
    print("\n[4/4] Generating visual verification grid...")
    # Select 2 samples from each class from the train batch and val batch
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle(
        "Brain MRI Preprocessed & Augmented Samples Verification\n"
        "(Col 1-2: Training with Augmentation | Col 3-4: Validation Preprocessed Only)",
        fontsize=14,
        fontweight="bold",
    )

    # Denormalize batches for accurate visual representation
    train_vis = denormalize_tensor(train_batch_imgs).permute(0, 2, 3, 1).cpu().numpy()
    val_vis = denormalize_tensor(val_batch_imgs).permute(0, 2, 3, 1).cpu().numpy()

    # Organize samples by class
    for class_id, class_name in enumerate(CLASSES):
        # Find training samples of this class
        train_indices = (train_batch_labels == class_id).nonzero(as_tuple=True)[0]
        val_indices = (val_batch_labels == class_id).nonzero(as_tuple=True)[0]

        # Plot 2 Training samples (Columns 0, 1)
        for col in range(2):
            ax = axes[class_id, col]
            if len(train_indices) > col:
                idx = train_indices[col].item()
                ax.imshow(train_vis[idx])
                ax.set_title(
                    f"[Train/Aug] {class_name}\n"
                    f"Label: {class_id} | Shape: (3, 224, 224)\n"
                    f"File: {train_batch_files[idx]}",
                    fontsize=8,
                )
            else:
                ax.text(0.5, 0.5, "No sample in batch", ha="center", va="center")
            ax.axis("off")

        # Plot 2 Validation samples (Columns 2, 3)
        for col in range(2):
            ax = axes[class_id, col + 2]
            if len(val_indices) > col:
                idx = val_indices[col].item()
                ax.imshow(val_vis[idx])
                ax.set_title(
                    f"[Val/Preproc] {class_name}\n"
                    f"Label: {class_id} | Shape: (3, 224, 224)\n"
                    f"File: {val_batch_files[idx]}",
                    fontsize=8,
                )
            else:
                ax.text(0.5, 0.5, "No sample in batch", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    output_fig_path = RESULTS_DIR / "preprocessed_samples_grid.png"
    plt.savefig(output_fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [SUCCESS] Verification grid saved to: {output_fig_path}")

    # Also copy to artifact directory for presentation
    artifact_img = Path("/Users/laxyagaba/.gemini/antigravity-ide/brain/31ae1d09-1ce3-4169-8973-37f3601df258/preprocessed_samples_grid.png")
    import shutil
    shutil.copy2(output_fig_path, artifact_img)
    print(f"  [SUCCESS] Artifact copy saved to: {artifact_img}")
    print("\nPre-processing verification completed successfully without errors.")


if __name__ == "__main__":
    run_verification()
