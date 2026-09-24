"""
Explanation Service for Brain MRI Classification Backend.
Handles:
- Reusing the existing GradCAMExplainer from src/explain.py.
- Utilizing the in-memory trained model from ModelService.
- Generating spatial activation heatmaps, overlays, and complete 3-panel figures in-memory.
- Formatting visualizations into base64 Data URIs for client consumption without disk storage.
"""

import io
import sys
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

from src.explain import GradCAMExplainer
from backend.services.model_service import model_service
from backend.services.prediction_service import prediction_service


def _image_to_base64_data_uri(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64-encoded Data URI string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


import logging
from src.input_guard import input_guard

logger = logging.getLogger("brain_tumor_backend.explanation_service")


class ExplanationService:
    """
    Service generating Grad-CAM explainability artifacts in-memory.
    Shares the pre-loaded EfficientNetB0 model to eliminate redundant model loading.
    """

    def __init__(self):
        self._explainer: Optional[GradCAMExplainer] = None

    def _get_explainer(self) -> GradCAMExplainer:
        """
        Lazily initializes the GradCAMExplainer using the already-loaded model instance.
        """
        if self._explainer is None:
            if not model_service.is_loaded():
                model_service.load_model()
            # Reuse the model already in memory
            self._explainer = GradCAMExplainer(model=model_service.model)
        return self._explainer

    def explain(self, image_bytes: bytes, filename: str = "upload") -> Dict[str, Any]:
        """
        Generates Grad-CAM visual explanation for uploaded MRI scan bytes:
        1. Validates and decodes image bytes in-memory into RGB PIL image.
        2. Executes Brain MRI Input Guardrail.
           -> If rejected or uncertain: STOP. Grad-CAM is NEVER computed.
        3. Computes Grad-CAM activations and predicted class scores using existing GradCAMExplainer.
        4. Encodes overlay, heatmap, and 3-panel figure into base64 Data URIs.
        5. Returns structured payload without persisting user uploads to disk.
        """
        # 1. In-memory validation & decoding
        rgb_img = prediction_service.validate_and_decode_image(image_bytes, filename=filename)

        # 2. Mandatory Input Guardrail Check
        guard_result = input_guard.validate(rgb_img)
        if not guard_result.accepted:
            logger.info(
                f"[GUARDRAIL INTERCEPT] Explain request for '{filename}' rejected. "
                f"Status: {guard_result.status.value}, Type: {guard_result.input_type}, Reason: {guard_result.reason}"
            )
            # CRITICAL SAFETY REQUIREMENT: STOP! Do NOT run Grad-CAM!
            return {
                "accepted": False,
                "input_type": guard_result.input_type,
                "reason": guard_result.reason,
                "message": guard_result.message,
                "predicted_class": None,
                "confidence": None,
                "probabilities": None,
                "gradcam_overlay_base64": None,
                "gradcam_heatmap_base64": None,
                "gradcam_panel_base64": None,
            }

        # 3. Generate Grad-CAM activations and forward scores (Only if VALID_MRI)
        explainer = self._get_explainer()
        explanation = explainer.generate_gradcam(rgb_img)

        # 4. In-memory base64 encoding of overlay and heatmap
        overlay_pil = Image.fromarray(explanation["overlay_arr"])
        heatmap_pil = Image.fromarray(explanation["heatmap_color"])

        overlay_b64 = _image_to_base64_data_uri(overlay_pil, format="PNG")
        heatmap_b64 = _image_to_base64_data_uri(heatmap_pil, format="PNG")

        # 5. Generate 3-panel presentation figure to memory buffer
        buf = io.BytesIO()
        explainer._create_combined_figure(explanation, output_path=buf)
        buf.seek(0)
        panel_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        return {
            "accepted": True,
            "input_type": "brain_mri",
            "predicted_class": explanation["predicted_class"],
            "confidence": round(float(explanation["confidence"]), 4),
            "probabilities": explanation["class_probabilities"],
            "gradcam_overlay_base64": overlay_b64,
            "gradcam_heatmap_base64": heatmap_b64,
            "gradcam_panel_base64": panel_b64,
            "reason": None,
            "message": "Brain MRI image verified.",
        }


# Global explanation service instance
explanation_service = ExplanationService()
