"""
Explainable AI (XAI) using Grad-CAM for Brain MRI Classification.
Generates class activation heatmaps to visualize the spatial regions that
contributed most strongly to the model's classification decisions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import argparse
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import keras

from src.config import (
    BEST_MODEL_PATH,
    CLASSES,
    ID2LABEL,
    IMAGE_SIZE,
    RESULTS_DIR,
)
from src.preprocessing import LetterboxResize

# Output directory for Grad-CAM artifacts
GRADCAM_OUTPUT_DIR = RESULTS_DIR / "gradcam"
GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable display mapping
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
}

# Targeted convolutional feature layer identified from model architecture inspection
TARGET_CONV_LAYER_NAME = "top_activation"


class GradCAMExplainer:
    """
    Grad-CAM explanation engine for EfficientNetB0 brain tumor classification.
    Computes gradients of target class score with respect to the final convolutional
    feature layer ('top_activation' of EfficientNetB0), producing spatially aligned heatmaps.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = BEST_MODEL_PATH,
        target_layer_name: str = TARGET_CONV_LAYER_NAME,
        model: Optional[Any] = None,
    ):
        self.model_path = Path(model_path)
        if model is not None:
            self.model = model
        else:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            # 1. Load trained model (Weights frozen)
            self.model = keras.models.load_model(str(self.model_path))

        # 2. Locate the backbone and target convolutional feature layer
        self.target_layer_name = target_layer_name
        self.target_layer = self._find_target_layer()

        # 3. Deterministic letterbox preprocessor (identical to predict.py)
        self.letterbox = LetterboxResize(target_size=IMAGE_SIZE)

    def _find_target_layer(self):
        """
        Inspects model architecture to dynamically locate the specified convolutional layer.
        """
        # Search in top-level layers
        for layer in self.model.layers:
            if layer.name == self.target_layer_name:
                return layer
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if sub_layer.name == self.target_layer_name:
                        return sub_layer

        # Fallback to backbone output
        eff_layer = self.model.get_layer("efficientnetb0")
        if hasattr(eff_layer, "get_layer"):
            try:
                return eff_layer.get_layer(self.target_layer_name)
            except Exception:
                pass
        raise ValueError(f"Could not locate target layer '{self.target_layer_name}' in model.")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> Tuple[np.ndarray, Image.Image, Tuple[int, int]]:
        """
        Applies exact deterministic preprocessing:
        - Validates existence
        - Converts explicitly to RGB
        - Letterbox resize to 224x224 (aspect ratio preserved with zero-padding)
        Returns:
            (input_tensor, original_rgb_pil, (orig_w, orig_h))
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found at path: {path}")
            with Image.open(path) as raw:
                orig_rgb = raw.convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_rgb = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = orig_rgb.size

        # Preprocess identically to predict.py
        processed_img = self.letterbox(orig_rgb)
        img_arr = np.array(processed_img, dtype=np.float32)
        input_tensor = np.expand_dims(img_arr, axis=0)  # (1, 224, 224, 3)

        return input_tensor, orig_rgb, (orig_w, orig_h)

    def generate_gradcam(
        self,
        image_input: Union[str, Path, Image.Image],
        target_class_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap, prediction probabilities, and overlays.
        """
        filename = Path(image_input).name if isinstance(image_input, (str, Path)) else "sample_mri.jpg"
        stem = Path(filename).stem

        input_tensor, orig_rgb, (orig_w, orig_h) = self.preprocess_image(image_input)
        x_torch = torch.from_numpy(input_tensor)

        # PyTorch hooks to capture activations and gradients of the target layer
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        self.model.zero_grad()
        preds = self.model(x_torch)
        probs = preds[0].detach().cpu().numpy()

        # Determine target class
        predicted_class_id = int(torch.argmax(preds[0]).item())
        target_id = predicted_class_id if target_class_id is None else target_class_id
        predicted_class = ID2LABEL[predicted_class_id]

        # Backward pass for the target class logit score
        score = preds[0, target_id]
        score.backward()

        # Clean up hooks immediately
        fwd_handle.remove()
        bwd_handle.remove()

        # Extract captured feature activations and gradients
        act = activations[0].detach()  # Shape: (1, 7, 7, 1280)
        grad = gradients[0].detach()   # Shape: (1, 7, 7, 1280)

        # Spatially pool the gradients across spatial dimensions (height, width)
        # to calculate the importance weights for each feature channel
        weights = torch.mean(grad, dim=(1, 2), keepdim=True)  # (1, 1, 1, 1280)
        cam = torch.sum(weights * act, dim=-1)                # (1, 7, 7)
        cam = torch.clamp(cam, min=0)                         # ReLU: positive influence only

        heatmap_raw = cam[0].cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min = float(heatmap_raw.min())
        cam_max = float(heatmap_raw.max())
        if cam_max > cam_min:
            heatmap_norm_7x7 = (heatmap_raw - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            heatmap_norm_7x7 = np.zeros_like(heatmap_raw)

        # --- Geometric Un-Letterboxing ---
        # Map the 7x7 heatmap back to the 224x224 canvas, crop out the letterbox border,
        # and resize the active region to match the original MRI dimensions exactly.
        heatmap_224 = cv2.resize(heatmap_norm_7x7, (IMAGE_SIZE[0], IMAGE_SIZE[1]), interpolation=cv2.INTER_LINEAR)

        scale = min(IMAGE_SIZE[0] / orig_w, IMAGE_SIZE[1] / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        paste_x = (IMAGE_SIZE[0] - new_w) // 2
        paste_y = (IMAGE_SIZE[1] - new_h) // 2

        active_crop = heatmap_224[paste_y : paste_y + new_h, paste_x : paste_x + new_w]
        aligned_heatmap = cv2.resize(active_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        aligned_heatmap = np.clip(aligned_heatmap, 0.0, 1.0)

        # --- Color Map & Overlay ---
        orig_arr = np.array(orig_rgb, dtype=np.uint8)
        heatmap_uint8 = np.uint8(255 * aligned_heatmap)

        # Apply JET colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Alpha blend overlay: 60% original image + 40% heatmap
        overlay_arr = np.uint8(0.60 * orig_arr + 0.40 * heatmap_color)

        class_probabilities = {
            CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASSES))
        }

        return {
            "filename": filename,
            "stem": stem,
            "orig_img": orig_rgb,
            "orig_arr": orig_arr,
            "heatmap": aligned_heatmap,
            "heatmap_color": heatmap_color,
            "overlay_arr": overlay_arr,
            "predicted_class": predicted_class,
            "predicted_class_id": predicted_class_id,
            "confidence": float(probs[predicted_class_id]),
            "class_probabilities": class_probabilities,
            "target_layer_name": self.target_layer_name,
            "original_dimensions": (orig_w, orig_h),
        }

    def save_explanation_artifacts(
        self,
        explanation: Dict[str, Any],
        output_dir: Path = GRADCAM_OUTPUT_DIR,
        actual_class: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Saves individual component images and the comprehensive combined visualization.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = explanation["stem"]

        orig_path = output_dir / f"original_{stem}.png"
        heatmap_path = output_dir / f"heatmap_{stem}.png"
        overlay_path = output_dir / f"overlay_{stem}.png"
        combined_path = output_dir / f"gradcam_{stem}.png"

        # 1. Save original MRI
        explanation["orig_img"].save(orig_path)

        # 2. Save colored heatmap
        Image.fromarray(explanation["heatmap_color"]).save(heatmap_path)

        # 3. Save overlay
        Image.fromarray(explanation["overlay_arr"]).save(overlay_path)

        # 4. Save Combined Visual Presentation Figure
        self._create_combined_figure(explanation, combined_path, actual_class=actual_class)

        return {
            "original": orig_path,
            "heatmap": heatmap_path,
            "overlay": overlay_path,
            "combined": combined_path,
        }

    def _create_combined_figure(
        self,
        explanation: Dict[str, Any],
        output_path: Path,
        actual_class: Optional[str] = None,
    ):
        """Creates a presentation-quality 3-panel Grad-CAM visualization."""
        fig = plt.figure(figsize=(15, 6), facecolor="#ffffff")

        pred_name = DISPLAY_NAMES.get(explanation["predicted_class"], explanation["predicted_class"])
        conf_pct = explanation["confidence"] * 100

        # Title and header
        title_text = f"Grad-CAM Model Explanation | Image: {explanation['filename']}"
        if actual_class:
            actual_display = DISPLAY_NAMES.get(actual_class, actual_class)
            is_correct = (actual_class == explanation["predicted_class"])
            status_text = "CORRECT PREDICTION" if is_correct else "MISCLASSIFIED SAMPLE"
            subtitle_text = (
                f"Actual: {actual_display} | Predicted: {pred_name} ({conf_pct:.2f}% confidence) | [{status_text}]"
            )
        else:
            subtitle_text = f"Predicted Class: {pred_name} (Confidence: {conf_pct:.2f}%)"

        plt.suptitle(f"{title_text}\n{subtitle_text}", fontsize=13, fontweight="bold", y=0.98)

        # Panel 1: Original MRI
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(explanation["orig_arr"])
        dims = explanation["original_dimensions"]
        ax1.set_title(f"A. Original MRI\nSize: {dims[0]}x{dims[1]} px", fontsize=11, fontweight="bold", pad=8)
        ax1.axis("off")

        # Panel 2: Grad-CAM Heatmap
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(explanation["heatmap"], cmap="jet", vmin=0, vmax=1)
        ax2.set_title(f"B. Grad-CAM Heatmap\nLayer: {explanation['target_layer_name']}", fontsize=11, fontweight="bold", pad=8)
        ax2.axis("off")
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Relative Activation Contribution", fontsize=9)

        # Panel 3: Overlay with Prediction Card
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(explanation["overlay_arr"])
        ax3.set_title("C. Grad-CAM Overlay\nRegions contributing to prediction", fontsize=11, fontweight="bold", pad=8)
        ax3.axis("off")

        # Add Probability Card text box below or within figure
        probs = explanation["class_probabilities"]
        prob_text = (
            f"Class Probabilities:\n"
            f"  • Glioma:      {probs.get('glioma', 0.0)*100:5.2f}%\n"
            f"  • Meningioma:  {probs.get('meningioma', 0.0)*100:5.2f}%\n"
            f"  • Pituitary:   {probs.get('pituitary', 0.0)*100:5.2f}%\n"
            f"  • No Tumor:    {probs.get('notumor', 0.0)*100:5.2f}%\n"
            f"Model: EfficientNetB0"
        )
        fig.text(
            0.50, 0.03,
            prob_text,
            ha="center", va="bottom",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", lw=1.2)
        )

        fig.text(
            0.50, 0.005,
            "Notice: Academic research explanation. Highlighted regions represent feature contribution to model decision, not clinical lesion boundaries.",
            ha="center", va="bottom",
            fontsize=8, style="italic", color="#6c757d"
        )

        plt.subplots_adjust(top=0.86, bottom=0.22, wspace=0.15)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    def format_console_output(self, explanation: Dict[str, Any], combined_path: Path) -> str:
        """Formats the terminal output according to the exact specification."""
        probs = explanation["class_probabilities"]
        pred_display = DISPLAY_NAMES.get(explanation["predicted_class"], explanation["predicted_class"])

        rel_path = combined_path.relative_to(PROJECT_ROOT) if combined_path.is_relative_to(PROJECT_ROOT) else combined_path

        lines = [
            "========================================",
            "GRAD-CAM EXPLANATION",
            "========================================",
            "",
            "Image:",
            explanation["filename"],
            "",
            "Prediction:",
            pred_display,
            "",
            "Class probabilities:",
            "",
            f"Glioma       {probs.get('glioma', 0.0) * 100:6.2f}%",
            f"Meningioma   {probs.get('meningioma', 0.0) * 100:6.2f}%",
            f"Pituitary    {probs.get('pituitary', 0.0) * 100:6.2f}%",
            f"No Tumor     {probs.get('notumor', 0.0) * 100:6.2f}%",
            "",
            "Grad-CAM:",
            str(rel_path),
            "",
            "========================================",
            "Interpretation: Highlighted regions indicate visual features",
            "that contributed to the model's prediction. Not clinical evidence.",
            "========================================",
        ]
        return "\n".join(lines)


def process_single_image(
    explainer: GradCAMExplainer,
    image_path: Path,
    actual_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes a single image, saves outputs, and prints console report."""
    explanation = explainer.generate_gradcam(image_path)
    saved_paths = explainer.save_explanation_artifacts(explanation, actual_class=actual_class)
    print(explainer.format_console_output(explanation, saved_paths["combined"]))
    return {"explanation": explanation, "saved_paths": saved_paths}


def process_batch_directory(
    explainer: GradCAMExplainer,
    directory_path: Path,
    max_images: int = 10,
) -> List[Dict[str, Any]]:
    """
    Processes a limited batch of images from a directory without modifying source files.
    """
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in directory_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    if not image_files:
        print(f"No image files found in {directory_path}")
        return []

    selected_files = image_files[:max_images]
    print(f"\nProcessing {len(selected_files)} images from {directory_path} (max limit: {max_images})...\n")

    results = []
    for img_file in selected_files:
        res = process_single_image(explainer, img_file)
        results.append(res)

    print(f"\n[SUCCESS] Completed Grad-CAM batch processing for {len(results)} images.")
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Classification - Grad-CAM Explanation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="Path to an individual MRI image (.jpg, .png) to explain",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Path to directory of images to process in batch mode (limited to safe sample size)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process in batch mode (default: 10)",
    )
    parser.add_argument(
        "--actual-class",
        type=str,
        default=None,
        help="Optional actual ground-truth class name (used for error analysis comparisons)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BEST_MODEL_PATH),
        help=f"Path to the trained .keras model (default: {BEST_MODEL_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.image_path and not args.directory:
        print("\nError: Please provide an image path or --directory to explain.")
        print("Usage:")
        print("  python src/explain.py <path_to_mri.jpg>")
        print("  python src/explain.py --directory <path_to_images_folder>\n")
        sys.exit(1)

    explainer = GradCAMExplainer(model_path=args.model_path)

    if args.directory:
        process_batch_directory(explainer, Path(args.directory), max_images=args.max_images)
    elif args.image_path:
        img_p = Path(args.image_path)
        if not img_p.exists():
            print(f"\nError: Specified image file does not exist: {img_p}\n")
            sys.exit(1)
        process_single_image(explainer, img_p, actual_class=args.actual_class)


if __name__ == "__main__":
    main()
