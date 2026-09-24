"""
Prediction Service for Brain MRI Classification Backend.
Handles:
- Input image validation (size, MIME type, format integrity).
- Reusing the project's exact deterministic LetterboxResize preprocessing.
- Executing model inference via ModelService.
- Formatting class probabilities and model confidence scores.
"""

import io
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import CLASSES, ID2LABEL, IMAGE_SIZE
from src.preprocessing import LetterboxResize
from backend.services.model_service import model_service

# Maximum image upload size: 10 MB
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG"}


import logging
from src.input_guard import input_guard

logger = logging.getLogger("brain_tumor_backend.prediction_service")


class PredictionService:
    """
    Coordinates image validation, input guardrail verification, deterministic preprocessing,
    and model inference. Processes uploaded MRI scans in-memory without saving them to disk.
    """

    def __init__(self):
        # Exact deterministic preprocessor used during training, evaluation, and predict.py
        self.letterbox = LetterboxResize(target_size=IMAGE_SIZE)

    def validate_and_decode_image(self, image_bytes: bytes, filename: str = "upload") -> Image.Image:
        """
        Validates raw image bytes:
        - Checks non-empty content
        - Enforces size limit (10MB)
        - Decodes image format and verifies against allowed formats (JPEG, PNG)
        - Ensures image is uncorrupted and converts explicitly to 3-channel RGB.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("No image data provided. Uploaded file is empty.")

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image file size ({len(image_bytes) / (1024*1024):.1f}MB) exceeds maximum limit of 10MB.")

        try:
            byte_stream = io.BytesIO(image_bytes)
            img = Image.open(byte_stream)
            img_format = (img.format or "").upper()
            
            # Map jpg -> jpeg
            if img_format == "JPG":
                img_format = "JPEG"

            if img_format not in {"JPEG", "PNG"}:
                raise ValueError(
                    f"Unsupported image format '{img.format or 'unknown'}'. "
                    "Only JPG/JPEG and PNG images are supported."
                )

            # Ensure valid pixel data can be decoded
            img.verify()

            # Re-open after verify() as required by PIL
            byte_stream.seek(0)
            img = Image.open(byte_stream)
            rgb_img = img.convert("RGB")
            return rgb_img

        except ValueError:
            raise
        except Exception:
            raise ValueError("Corrupted or unreadable image file. Unable to decode image data.")

    def preprocess_image(self, rgb_img: Image.Image) -> np.ndarray:
        """
        Applies identical letterbox preprocessing:
        - Aspect-ratio preserving letterbox resize to 224x224 with zero padding.
        - Expands dimensions to (1, 224, 224, 3) float32 array.
        """
        processed_img = self.letterbox(rgb_img)
        img_arr = np.array(processed_img, dtype=np.float32)
        return np.expand_dims(img_arr, axis=0)

    def predict(self, image_bytes: bytes, filename: str = "upload") -> Dict[str, Any]:
        """
        Full prediction pipeline for uploaded image bytes:
        1. Validate and decode into RGB PIL image in-memory.
        2. Execute Brain MRI Input Guardrail.
           -> If rejected or uncertain: STOP. Tumor classifier NEVER executes.
        3. Apply LetterboxResize to (1, 224, 224, 3).
        4. Model forward pass.
        5. Package predicted class and model confidence scores.
        """
        # 1. Validation & decoding
        rgb_img = self.validate_and_decode_image(image_bytes, filename=filename)

        # 2. Mandatory Input Guardrail Check
        guard_result = input_guard.validate(rgb_img)
        if not guard_result.accepted:
            logger.info(
                f"[GUARDRAIL INTERCEPT] Prediction request for '{filename}' rejected. "
                f"Status: {guard_result.status.value}, Type: {guard_result.input_type}, Reason: {guard_result.reason}"
            )
            # CRITICAL SAFETY REQUIREMENT: STOP! Do NOT call model_service.predict()!
            return {
                "accepted": False,
                "input_type": guard_result.input_type,
                "reason": guard_result.reason,
                "message": guard_result.message,
                "predicted_class": None,
                "confidence": None,
                "probabilities": None,
            }

        # 3. Preprocessing (Executed ONLY if input is verified VALID_MRI)
        input_tensor = self.preprocess_image(rgb_img)

        # 4. Model inference
        probabilities = model_service.predict(input_tensor)

        # 5. Process predictions
        pred_id = int(np.argmax(probabilities))
        pred_class = ID2LABEL[pred_id]
        confidence = float(probabilities[pred_id])

        class_probs = {
            CLASSES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASSES))
        }

        return {
            "accepted": True,
            "input_type": "brain_mri",
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": class_probs,
            "reason": None,
            "message": "Brain MRI image verified.",
        }


# Global prediction service instance
prediction_service = PredictionService()
