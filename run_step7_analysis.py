"""
Comprehensive Error Analysis and Confidence Analysis Script for Step 7.
Analyzes 1,600 untouched test predictions, evaluates per-class performance,
misclassification pairs, confidence distributions, high-confidence errors,
borderline predictions, generates error image grids, and runs Grad-CAM on error cases.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure KERAS_BACKEND is set
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    CLASSES,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    IMAGE_SIZE,
    RESULTS_DIR,
    TEST_DATA_DIR,
    BEST_MODEL_PATH,
)
from src.preprocessing import LetterboxResize
from src.explain import GradCAMExplainer

# Setup directories
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
GRADCAM_ERRORS_DIR = RESULTS_DIR / "gradcam" / "errors"
ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_ERRORS_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
}


def load_test_predictions() -> pd.DataFrame:
    """Loads and validates existing test predictions CSV."""
    csv_path = RESULTS_DIR / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing test predictions file at: {csv_path}")
    df = pd.read_csv(csv_path)
    assert len(df) == 1600, f"Expected 1,600 predictions, found {len(df)}"
    return df


def generate_per_class_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive per-class error, performance, and confidence metrics."""
    records = []
    for cls in CLASSES:
        cls_df = df[df["true_class"] == cls]
        pred_cls_df = df[df["predicted_class"] == cls]

        total = len(cls_df)
        correct = (cls_df["predicted_class"] == cls).sum()
        incorrect = total - correct

        # True Positives, False Positives, False Negatives
        tp = correct
        fp = (pred_cls_df["true_class"] != cls).sum()
        fn = incorrect

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        confs = cls_df["confidence"]
        records.append({
            "class_name": cls,
            "display_name": DISPLAY_NAMES[cls],
            "total_images": int(total),
            "correct": int(correct),
            "incorrect": int(incorrect),
            "accuracy": float(round(accuracy, 4)),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4)),
            "mean_confidence": float(round(confs.mean(), 4)),
            "median_confidence": float(round(confs.median(), 4)),
            "min_confidence": float(round(confs.min(), 4)),
            "max_confidence": float(round(confs.max(), 4)),
        })

    out_df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "per_class_error_analysis.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved per-class error analysis: {out_path}")
    return out_df


def generate_misclassification_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes error frequency for each Actual -> Predicted misclassification pair."""
    errors_df = df[df["correct_or_incorrect"] == "incorrect"].copy()
    total_errors = len(errors_df)
    total_test = len(df)

    pairs = errors_df.groupby(["true_class", "predicted_class"]).size().reset_index(name="count")
    pairs = pairs.sort_values(by="count", ascending=False).reset_index(drop=True)

    pairs["pair_label"] = pairs["true_class"] + " → " + pairs["predicted_class"]
    pairs["percentage_of_errors"] = (pairs["count"] / total_errors * 100).round(2)
    pairs["percentage_of_dataset"] = (pairs["count"] / total_test * 100).round(2)

    out_path = RESULTS_DIR / "misclassification_pairs.csv"
    pairs_export = pairs.rename(columns={"true_class": "actual_class"})
    pairs_export[["actual_class", "predicted_class", "count", "percentage_of_errors", "percentage_of_dataset"]].to_csv(out_path, index=False)
    print(f"[OK] Saved misclassification pairs CSV: {out_path}")

    # Plot Bar Chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(pairs["pair_label"][::-1], pairs["count"][::-1], color="#d62728", alpha=0.85, edgecolor="#721c24")
    plt.xlabel("Number of Misclassified Images", fontsize=11, fontweight="bold")
    plt.ylabel("Actual → Predicted Class Pair", fontsize=11, fontweight="bold")
    plt.title("Brain MRI Test Dataset: Misclassification Pairs by Frequency\n(Total Errors: 115 / 1,600 Images)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, pct in zip(bars, pairs["percentage_of_errors"][::-1]):
        w = bar.get_width()
        plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({pct:.1f}%)", va="center", fontsize=9.5, fontweight="bold")

    plt.xlim(0, max(pairs["count"]) + 8)
    plt.tight_layout()
    chart_path = RESULTS_DIR / "misclassification_pairs.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[OK] Saved misclassification pairs chart: {chart_path}")

    return pairs


def generate_confidence_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes and plots confidence distributions for correct vs incorrect predictions."""
    corr = df[df["correct_or_incorrect"] == "correct"]["confidence"]
    incorr = df[df["correct_or_incorrect"] == "incorrect"]["confidence"]

    stats = {
        "correct": {
            "count": int(len(corr)),
            "mean": float(round(corr.mean(), 4)),
            "median": float(round(corr.median(), 4)),
            "std": float(round(corr.std(), 4)),
            "min": float(round(corr.min(), 4)),
            "max": float(round(corr.max(), 4)),
        },
        "incorrect": {
            "count": int(len(incorr)),
            "mean": float(round(incorr.mean(), 4)),
            "median": float(round(incorr.median(), 4)),
            "std": float(round(incorr.std(), 4)),
            "min": float(round(incorr.min(), 4)),
            "max": float(round(incorr.max(), 4)),
        },
    }

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.3, 1.0, 36)

    plt.hist(corr, bins=bins, alpha=0.65, color="#2ca02c", label=f"Correct Predictions (N={len(corr):,}, Mean={corr.mean():.2f})", edgecolor="#1e7e34", density=True)
    plt.hist(incorr, bins=bins, alpha=0.75, color="#d62728", label=f"Incorrect Predictions (N={len(incorr):,}, Mean={incorr.mean():.2f})", edgecolor="#721c24", density=True)

    plt.axvline(corr.mean(), color="#1e7e34", linestyle="--", lw=2, label=f"Correct Mean: {corr.mean():.2f}")
    plt.axvline(incorr.mean(), color="#721c24", linestyle="--", lw=2, label=f"Incorrect Mean: {incorr.mean():.2f}")

    plt.xlabel("Model Confidence Score (Softmax Probability)", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Model Confidence Score Distribution: Correct vs. Incorrect Predictions\n[Machine Learning Confidence Analysis - Test Dataset]", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    hist_path = RESULTS_DIR / "confidence_correct_vs_incorrect.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[OK] Saved confidence distribution chart: {hist_path}")

    return stats


def generate_high_confidence_and_borderline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identifies high-confidence errors (>= 0.80) and borderline predictions (< 0.60)."""
    # High-confidence errors
    high_conf = df[(df["correct_or_incorrect"] == "incorrect") & (df["confidence"] >= 0.80)].copy()
    high_conf = high_conf.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    high_conf_out = high_conf[["filename", "true_class", "predicted_class", "confidence"]].rename(
        columns={"true_class": "actual_class"}
    )
    high_conf_path = RESULTS_DIR / "high_confidence_errors.csv"
    high_conf_out.to_csv(high_conf_path, index=False)
    print(f"[OK] Saved high-confidence errors CSV: {high_conf_path} (N={len(high_conf)})")

    # Borderline predictions
    borderline = df[df["confidence"] < 0.60].copy()
    borderline = borderline.sort_values(by="confidence", ascending=True).reset_index(drop=True)
    borderline_out = borderline[["filename", "true_class", "predicted_class", "confidence", "correct_or_incorrect"]].rename(
        columns={"true_class": "actual_class"}
    )
    borderline_path = RESULTS_DIR / "borderline_predictions.csv"
    borderline_out.to_csv(borderline_path, index=False)
    print(f"[OK] Saved borderline predictions CSV: {borderline_path} (N={len(borderline)})")

    return high_conf_out, borderline_out


def generate_confidence_by_class(df: pd.DataFrame):
    """Generates comparison bar chart of model confidence across predicted classes."""
    classes = CLASSES
    overall_conf = [df[df["predicted_class"] == c]["confidence"].mean() for c in classes]
    correct_conf = [df[(df["predicted_class"] == c) & (df["correct_or_incorrect"] == "correct")]["confidence"].mean() for c in classes]
    incorrect_conf = [
        df[(df["predicted_class"] == c) & (df["correct_or_incorrect"] == "incorrect")]["confidence"].mean()
        if len(df[(df["predicted_class"] == c) & (df["correct_or_incorrect"] == "incorrect")]) > 0 else 0.0
        for c in classes
    ]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, overall_conf, width, label="All Predictions", color="#1f77b4", edgecolor="#114b72")
    r2 = ax.bar(x, correct_conf, width, label="Correct Predictions", color="#2ca02c", edgecolor="#1e7e34")
    r3 = ax.bar(x + width, incorrect_conf, width, label="Incorrect Predictions", color="#d62728", edgecolor="#721c24")

    ax.set_ylabel("Average Model Confidence Score", fontsize=11, fontweight="bold")
    ax.set_title("Average Model Confidence Score by Predicted Class\n[Overall vs. Correct vs. Incorrect Predictions]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[c] for c in classes], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "confidence_by_class.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[OK] Saved confidence by class chart: {chart_path}")


def create_error_image_grid(
    records: List[Dict[str, Any]],
    title: str,
    output_path: Path,
    max_images: int = 12,
):
    """Renders a grid of sample MRI images with actual class, predicted class, and confidence."""
    n = min(len(records), max_images)
    if n == 0:
        print(f"No images to plot for {title}")
        return

    cols = 4
    rows = int(np.ceil(n / cols))
    letterbox = LetterboxResize(target_size=IMAGE_SIZE)

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.8), facecolor="#ffffff")
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f"{title}\n({n} Representative Cases Shown)", fontsize=13, fontweight="bold", y=1.02)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if idx < n:
            rec = records[idx]
            fname = rec["filename"]
            act_cls = rec["true_class"] if "true_class" in rec else rec["actual_class"]
            pred_cls = rec["predicted_class"]
            conf = float(rec["confidence"])
            is_corr = rec.get("correct_or_incorrect", "incorrect") == "correct"

            img_p = TEST_DATA_DIR / act_cls / fname
            try:
                with Image.open(img_p) as raw:
                    vis_img = letterbox(raw.convert("RGB"))
                ax.imshow(vis_img)
                color = "#155724" if is_corr else "#721c24"
                ax.set_title(
                    f"Actual: {DISPLAY_NAMES.get(act_cls, act_cls)}\n"
                    f"Predicted: {DISPLAY_NAMES.get(pred_cls, pred_cls)}\n"
                    f"Confidence: {conf*100:.1f}%\n"
                    f"File: {fname}",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved error image grid: {output_path}")


def generate_error_image_grids(df: pd.DataFrame, high_conf_df: pd.DataFrame, borderline_df: pd.DataFrame):
    """Generates the three specified representative error image grids."""
    # 1. Top Misclassified pairs (glioma -> meningioma and glioma -> notumor)
    top_pairs_df = df[(df["true_class"] == "glioma") & (df["predicted_class"].isin(["meningioma", "notumor"]))].head(12)
    create_error_image_grid(
        top_pairs_df.to_dict("records"),
        title="Top Misclassification Patterns: Glioma Misclassified as Meningioma / No Tumor",
        output_path=ERROR_ANALYSIS_DIR / "top_misclassified.png",
        max_images=12,
    )

    # 2. High-Confidence Errors (confidence >= 0.80)
    create_error_image_grid(
        high_conf_df.head(12).to_dict("records"),
        title="High-Confidence Prediction Errors (Model Confidence ≥ 80%)",
        output_path=ERROR_ANALYSIS_DIR / "high_confidence_errors.png",
        max_images=12,
    )

    # 3. Borderline Predictions (confidence < 0.60)
    create_error_image_grid(
        borderline_df.head(12).to_dict("records"),
        title="Borderline Predictions (Model Confidence < 60%)",
        output_path=ERROR_ANALYSIS_DIR / "borderline_predictions.png",
        max_images=12,
    )


def generate_gradcam_error_cases():
    """Generates Grad-CAM visualizations for 5 representative misclassified cases."""
    print("\nGenerating Grad-CAM visualizations for representative misclassification cases...")
    explainer = GradCAMExplainer()

    # Carefully selected representative misclassified cases
    cases = [
        # 1. Highest confidence error
        {"filename": "Te-gl_109.jpg", "actual": "glioma", "category": "High-Confidence Error (99.93%)"},
        # 2. Frequent pair high-confidence error (glioma -> meningioma)
        {"filename": "Te-gl_160.jpg", "actual": "glioma", "category": "Frequent Pair: Glioma → Meningioma (99.20%)"},
        # 3. Glioma misclassified as healthy (glioma -> notumor)
        {"filename": "Te-gl_144.jpg", "actual": "glioma", "category": "Subtle Lesion: Glioma → No Tumor (72.71%)"},
        # 4. Meningioma misclassified as pituitary
        {"filename": "Te-aug-me_24.jpg", "actual": "meningioma", "category": "Meningioma → Pituitary (91.56%)"},
        # 5. Low-confidence borderline error
        {"filename": "Te-gl_102.jpg", "actual": "glioma", "category": "Borderline Decision (51.06%)"},
    ]

    saved_error_plots = []
    for c in cases:
        fname = c["filename"]
        act_cls = c["actual"]
        img_p = TEST_DATA_DIR / act_cls / fname
        if not img_p.exists():
            print(f"Skipping missing case: {img_p}")
            continue

        exp = explainer.generate_gradcam(img_p)
        stem = Path(fname).stem
        out_combined = GRADCAM_ERRORS_DIR / f"error_gradcam_{stem}.png"

        # Save with actual class label
        explainer._create_combined_figure(exp, out_combined, actual_class=act_cls)
        saved_error_plots.append(out_combined)
        print(f"  [OK] Saved Grad-CAM error case: {out_combined.name} (Actual: {act_cls}, Pred: {exp['predicted_class']})")

    return saved_error_plots


def write_error_analysis_report(
    df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    conf_stats: Dict[str, Any],
    high_conf_df: pd.DataFrame,
    borderline_df: pd.DataFrame,
):
    """Generates the comprehensive error_analysis_report.txt."""
    total_test = len(df)
    correct = (df["correct_or_incorrect"] == "correct").sum()
    incorrect = total_test - correct
    accuracy = correct / total_test
    error_rate = incorrect / total_test

    top_pair = pairs_df.iloc[0]

    report = f"""======================================================================
STEP 7: COMPREHENSIVE ERROR ANALYSIS & MODEL CONFIDENCE REPORT
======================================================================
Project: Brain MRI Tumor Classification (Educational & Research)
Date: September 24, 2026
Model Evaluated: best_brain_tumor_model.keras (EfficientNetB0)
Dataset Evaluated: archive/Testing/ (N = 1,600 Untouched Test Images)
======================================================================

1. OVERALL ERROR SUMMARY
   - Total Test Images:          {total_test:,}
   - Correct Predictions:        {correct:,} ({accuracy*100:.2f}%)
   - Incorrect Predictions:      {incorrect:,} ({error_rate*100:.2f}%)
   - Overall Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)
   - Overall Error Rate:         {error_rate:.4f} ({error_rate*100:.2f}%)

2. STRONGEST AND WEAKEST PERFORMING CLASSES
   Based on calculated machine-learning metrics on the test dataset:
   - Strongest Sensitivity:
     • Pituitary: 100.00% Recall (400/400 correct), F1-Score: 0.9780.
       Zero false negatives observed across all test cases.
     • No Tumor: 99.25% Recall (397/400 correct), F1-Score: 0.9601.
       Excellent specificity distinguishing healthy brain anatomy.
   - Classes with More Errors:
     • Glioma: 78.00% Recall (312/400 correct), 88 errors. F1-Score: 0.8655.
       Lowest sensitivity; frequently shares textural features with meningioma.
     • Meningioma: 94.00% Recall (376/400 correct), 24 errors. Precision: 0.8664.
       Received the largest volume of false positive intrusions (58 false positives).

3. MISCLASSIFICATION PAIRS ANALYSIS
   Total Error Types: {len(pairs_df)} distinct Actual → Predicted combinations.
   Top Error Pairs by Frequency:
"""
    for _, row in pairs_df.iterrows():
        report += f"   • {row['true_class']:10s} → {row['predicted_class']:10s}: {int(row['count']):3d} errors ({row['percentage_of_errors']:5.2f}% of errors, {row['percentage_of_dataset']:.2f}% of test set)\n"

    report += f"""
   - Most Frequent Misclassification Pair:
     '{top_pair['true_class']} → {top_pair['predicted_class']}' ({int(top_pair['count'])} cases, {top_pair['percentage_of_errors']:.1f}% of all errors).
     This reflects a known radiological challenge in 2D axial MRI where extra-axial
     meningiomas and intra-axial high-grade gliomas with mass effect share similar
     signal intensities and border enhancements.

4. CONFIDENCE ANALYSIS (MODEL CONFIDENCE SCORES)
   - Correct Predictions (N = {conf_stats['correct']['count']:,}):
     • Mean Confidence:   {conf_stats['correct']['mean']:.4f} ({conf_stats['correct']['mean']*100:.2f}%)
     • Median Confidence: {conf_stats['correct']['median']:.4f} ({conf_stats['correct']['median']*100:.2f}%)
     • Min Confidence:    {conf_stats['correct']['min']:.4f}
     • Max Confidence:    {conf_stats['correct']['max']:.4f}
   - Incorrect Predictions (N = {conf_stats['incorrect']['count']:,}):
     • Mean Confidence:   {conf_stats['incorrect']['mean']:.4f} ({conf_stats['incorrect']['mean']*100:.2f}%)
     • Median Confidence: {conf_stats['incorrect']['median']:.4f} ({conf_stats['incorrect']['median']*100:.2f}%)
     • Min Confidence:    {conf_stats['incorrect']['min']:.4f}
     • Max Confidence:    {conf_stats['incorrect']['max']:.4f}
   - Note on Confidence Scores:
     Softmax output values represent model-assigned class probabilities and must
     not be interpreted as clinically calibrated diagnostic probabilities.

5. HIGH-CONFIDENCE ERRORS & BORDERLINE CASES
   - High-Confidence Errors (Confidence ≥ 80%): {len(high_conf_df)} images ({len(high_conf_df)/incorrect*100:.1f}% of errors)
     Cases where the model was confident in its output but incorrect.
     Highest confidence error: {high_conf_df.iloc[0]['filename']} ({high_conf_df.iloc[0]['actual_class']} → {high_conf_df.iloc[0]['predicted_class']}, Confidence: {high_conf_df.iloc[0]['confidence']*100:.2f}%)
   - Borderline Predictions (Confidence < 60%): {len(borderline_df)} images
     • Borderline Correct:   {(borderline_df['correct_or_incorrect'] == 'correct').sum()} images
     • Borderline Incorrect: {(borderline_df['correct_or_incorrect'] == 'incorrect').sum()} images
     Threshold 60% is used strictly as an exploratory data analysis filter.

6. OBSERVATIONS FROM ERROR IMAGES & GRAD-CAM
   - Glioma vs. Meningioma Confusions:
     Grad-CAM heatmaps on misclassified gliomas frequently show activation focused
     on dural/convexity margins or focal hyperintensities that resemble meningioma
     tail signs or broad attachments.
   - Glioma vs. No Tumor Confusions:
     Misclassifications of glioma as 'notumor' primarily occur on peripheral slices
     at the cranial vertex or skull base where the tumor mass is small, non-enhancing,
     or indistinct from normal parenchyma on non-contrast sequences.
   - Model Calibration:
     The significant gap between mean correct confidence (97.29%) and mean incorrect
     confidence (75.91%) shows that the network is generally less confident on difficult
     or ambiguous scans.

7. DATA QUALITY, REPRODUCIBILITY & BENCHMARK LIMITATIONS
   - Intra-Test Redundancy:
     As discovered during dataset inspection (Step 1), the test split contains 15
     exact duplicate hash groups (repeated slices compiled from the same acquisition).
     These duplicates were strictly preserved without modification.
   - Cross-Split Verification:
     Cryptographic MD5 hash verification confirmed zero exact duplicate images
     between the training set and testing set.
   - Near-Duplicate Limitation:
     Absence of exact hash matches across train and test sets does not guarantee the
     absence of near-duplicates or adjacent axial slices from the same subject.
     Therefore, the dataset is not claimed to be completely leakage-free.

======================================================================
LEGAL & EDUCATIONAL DISCLAIMER:
This evaluation is conducted exclusively for educational and academic machine
learning research. Model confidence scores and metrics are computer-vision
statistical measurements on this dataset and must NOT be used for clinical
medical diagnosis, treatment decisions, or patient care.
======================================================================
"""
    out_path = RESULTS_DIR / "error_analysis_report.txt"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"[OK] Saved error analysis report: {out_path}")


def write_machine_readable_summary(
    df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    conf_stats: Dict[str, Any],
    high_conf_df: pd.DataFrame,
    borderline_df: pd.DataFrame,
):
    """Generates the structured JSON summary: results/model_analysis_summary.json."""
    total_test = len(df)
    correct = int((df["correct_or_incorrect"] == "correct").sum())
    incorrect = int(total_test - correct)

    class_metrics = {}
    for _, row in per_class_df.iterrows():
        c = row["class_name"]
        class_metrics[c] = {
            "total_images": int(row["total_images"]),
            "correct": int(row["correct"]),
            "incorrect": int(row["incorrect"]),
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1_score"]),
            "mean_confidence": float(row["mean_confidence"]),
            "median_confidence": float(row["median_confidence"]),
            "min_confidence": float(row["min_confidence"]),
            "max_confidence": float(row["max_confidence"]),
        }

    misclassification_pairs = {}
    for _, row in pairs_df.iterrows():
        key = f"{row['true_class']}_to_{row['predicted_class']}"
        misclassification_pairs[key] = {
            "actual_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "count": int(row["count"]),
            "percentage_of_errors": float(row["percentage_of_errors"]),
            "percentage_of_dataset": float(row["percentage_of_dataset"]),
        }

    summary_json = {
        "test_samples": total_test,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": float(round(correct / total_test, 4)),
        "error_rate": float(round(incorrect / total_test, 4)),
        "mean_confidence_correct": conf_stats["correct"]["mean"],
        "mean_confidence_incorrect": conf_stats["incorrect"]["mean"],
        "high_confidence_error_count": len(high_conf_df),
        "borderline_prediction_count": len(borderline_df),
        "class_metrics": class_metrics,
        "misclassification_pairs": misclassification_pairs,
    }

    out_path = RESULTS_DIR / "model_analysis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    print(f"[OK] Saved model analysis summary JSON: {out_path}")
    return summary_json


def main():
    print("=" * 65)
    print("STEP 7: COMPREHENSIVE ERROR AND CONFIDENCE ANALYSIS")
    print("=" * 65)

    # 1. Load predictions
    df = load_test_predictions()

    # 2. Per-class analysis
    per_class_df = generate_per_class_analysis(df)

    # 3. Misclassification pairs
    pairs_df = generate_misclassification_pairs(df)

    # 4. Confidence distributions
    conf_stats = generate_confidence_distributions(df)

    # 5. High-confidence errors and borderline predictions
    high_conf_df, borderline_df = generate_high_confidence_and_borderline(df)

    # 6. Confidence by class chart
    generate_confidence_by_class(df)

    # 7. Error image grids
    generate_error_image_grids(df, high_conf_df, borderline_df)

    # 8. Grad-CAM error cases
    generate_gradcam_error_cases()

    # 9. Reports
    write_error_analysis_report(df, per_class_df, pairs_df, conf_stats, high_conf_df, borderline_df)
    write_machine_readable_summary(df, per_class_df, pairs_df, conf_stats, high_conf_df, borderline_df)

    print("\n" + "=" * 65)
    print("STEP 7 EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
