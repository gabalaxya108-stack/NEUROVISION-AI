"""
Standalone Image Prediction Pipeline for Brain MRI Tumor Classification.
Accepts an MRI image path, applies deterministic letterbox preprocessing,
runs inference using the best trained EfficientNetB0 model, and displays
class probabilities and predicted classification.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from src.config import (
    BEST_MODEL_PATH,
    CLASSES,
    ID2LABEL,
    IMAGE_SIZE,
)
from src.preprocessing import LetterboxResize

# Human-readable display mapping
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
}


class BrainTumorPredictor:
    """
    Inference engine for loading the saved brain tumor classification model
    and running standalone predictions on individual MRI images.
    """

    def __init__(self, model_path: Union[str, Path] = BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Load trained model (Inference mode, weights frozen)
        self.model = keras.models.load_model(str(self.model_path))
        # Exact deterministic inference preprocessing
        self.letterbox = LetterboxResize(target_size=IMAGE_SIZE)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Applies exact deterministic preprocessing:
        1. Validates existence (if path).
        2. Converts image explicitly to 3-channel RGB.
        3. Applies aspect-ratio preserving letterbox resize to 224x224 with zero padding.
        4. Expands dimensions to (1, 224, 224, 3) float32 array.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found at path: {path}")
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        processed_img = self.letterbox(rgb_img)
        img_arr = np.array(processed_img, dtype=np.float32)
        return np.expand_dims(img_arr, axis=0)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        save_json_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs model inference on the provided MRI image.
        
        Returns:
            Dictionary containing filename, predicted class, and probability distribution.
        """
        filename = Path(image_input).name if isinstance(image_input, (str, Path)) else "in_memory_image"
        from src.input_guard import input_guard

        # --- Mandatory Input Guardrail Stage ---
        guard_result = input_guard.validate(image_input)
        if not guard_result.accepted:
            result = {
                "accepted": False,
                "input_type": guard_result.input_type,
                "reason": guard_result.reason,
                "message": guard_result.message,
                "image": filename,
                "predicted_class": None,
                "probabilities": None,
            }
            if save_json_path:
                out_p = Path(save_json_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w") as f:
                    json.dump(result, f, indent=4)
            return result

        # 1. Preprocess (Executed ONLY if verified VALID_MRI)
        input_tensor = self.preprocess_image(image_input)

        # 2. Forward pass
        probabilities = self.model.predict(input_tensor, verbose=0)[0]

        # 3. Determine highest-probability class
        pred_id = int(np.argmax(probabilities))
        pred_class = ID2LABEL[pred_id]
        confidence = float(probabilities[pred_id])

        class_probs = {
            CLASSES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASSES))
        }

        result = {
            "accepted": True,
            "input_type": "brain_mri",
            "image": filename,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": class_probs,
        }

        # Save JSON if requested
        if save_json_path:
            out_p = Path(save_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                json.dump(result, f, indent=4)
            print(f"\n[INFO] Prediction result saved to JSON: {out_p}")

        return result

    def format_console_output(self, result: Dict[str, Any]) -> str:
        """Formats the prediction result according to the exact specification."""
        if not result.get("accepted", True):
            return "\n".join([
                "========================================",
                "BRAIN MRI INPUT VALIDATION: REJECTED",
                "========================================",
                "",
                "Image:",
                result["image"],
                "",
                "Status:",
                f"REJECTED ({result.get('reason', 'invalid_input')})",
                "",
                "Message:",
                result.get("message", "This image does not appear to be a brain MRI."),
                "",
                "Notice:",
                "Tumor classification was NOT executed.",
                "This application only accepts valid brain MRI images.",
                "========================================",
            ])

        probs = result["probabilities"]
        pred_class = result["predicted_class"]

        output_lines = [
            "========================================",
            "BRAIN MRI CLASSIFICATION",
            "========================================",
            "",
            "Image:",
            result["image"],
            "",
            "Prediction:",
            pred_class,
            "",
            "Class probabilities:",
            "",
            f"Glioma       {probs.get('glioma', 0.0) * 100:6.2f}%",
            f"Meningioma   {probs.get('meningioma', 0.0) * 100:6.2f}%",
            f"Pituitary    {probs.get('pituitary', 0.0) * 100:6.2f}%",
            f"No Tumor     {probs.get('notumor', 0.0) * 100:6.2f}%",
            "",
            "Model:",
            "EfficientNetB0",
            "",
            "========================================",
            "Notice: Academic research tool. Class probabilities do",
            "not constitute a medical diagnosis or clinical certainty.",
            "========================================",
        ]
        return "\n".join(output_lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Classification Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="Path to the MRI image file to classify (.jpg, .png, etc.)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional file path to save prediction results as a JSON file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BEST_MODEL_PATH),
        help=f"Path to the trained .keras model file (default: {BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verifies that the inference pipeline, preprocessing, and model load correctly.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verify:
        print("\nVerifying Brain Tumor Prediction Pipeline...")
        predictor = BrainTumorPredictor(model_path=args.model_path)
        print(f"[OK] Model successfully loaded from: {predictor.model_path}")
        print(f"[OK] Input shape expected: {predictor.model.input_shape}")
        print(f"[OK] Classes: {CLASSES}")
        print("[OK] Prediction pipeline verified and ready for inference.\n")
        return

    if not args.image_path:
        print("\nError: Please provide a valid image path to classify.")
        print("Usage: python src/predict.py <path_to_mri_image> [--save-json <path>]")
        print("Example: python src/predict.py sample.jpg --save-json output.json\n")
        sys.exit(1)

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"\nError: Specified image file does not exist: {image_path}\n")
        sys.exit(1)

    # Initialize predictor and run inference
    predictor = BrainTumorPredictor(model_path=args.model_path)
    result = predictor.predict(image_path, save_json_path=args.save_json)

    # Display console output
    print(predictor.format_console_output(result))


if __name__ == "__main__":
    main()
