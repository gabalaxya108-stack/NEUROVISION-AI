"""
Pydantic data models and schemas for the Brain MRI Inference Backend API.
Enforces strict response formats, types, and academic research documentation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str = Field(default="ok", example="ok")
    model_loaded: bool = Field(..., description="Whether the classification model is currently loaded in memory")


class ModelInfoResponse(BaseModel):
    """Metadata describing the deployed model architecture, classes, and research scope."""
    model: str = Field(default="EfficientNetB0", description="Model backbone architecture")
    input_shape: List[int] = Field(default=[224, 224, 3], description="Expected input dimensions [height, width, channels]")
    classes: List[str] = Field(
        default=["glioma", "meningioma", "pituitary", "notumor"],
        description="Target class labels corresponding to class IDs 0, 1, 2, 3"
    )
    test_accuracy: float = Field(
        default=0.9281,
        description="Empirical benchmark accuracy achieved on the untouched 1,600-image test set"
    )
    purpose: str = Field(
        default="educational/research image classification",
        description="Project scope and research purpose"
    )


class PredictionResponse(BaseModel):
    """Inference response containing guardrail status, predicted class, model confidence score, and class probabilities."""
    accepted: bool = Field(..., description="Whether the uploaded image was accepted by the brain MRI guardrail")
    input_type: str = Field(..., description="Detected input type: 'brain_mri', 'non_brain_mri', or 'uncertain'")
    predicted_class: Optional[str] = Field(None, description="Predicted class label with highest model confidence score")
    confidence: Optional[float] = Field(
        None,
        description="Model confidence score (0.0 to 1.0). Educational/research metric; not a clinical diagnostic probability."
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Model confidence scores across all 4 target classes (sum to 1.0)"
    )
    reason: Optional[str] = Field(None, description="Rejection reason code if rejected or uncertain")
    message: Optional[str] = Field(None, description="User-facing validation status message")


class ExplanationResponse(BaseModel):
    """Grad-CAM explainability response with guardrail verification, prediction details, and visual heatmaps."""
    accepted: bool = Field(..., description="Whether the uploaded image was accepted by the brain MRI guardrail")
    input_type: str = Field(..., description="Detected input type: 'brain_mri', 'non_brain_mri', or 'uncertain'")
    predicted_class: Optional[str] = Field(None, description="Predicted class label")
    confidence: Optional[float] = Field(None, description="Model confidence score for the predicted class")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Model confidence scores across all target classes")
    gradcam_overlay_base64: Optional[str] = Field(None, description="Base64 Data URI of MRI + Grad-CAM overlay")
    gradcam_heatmap_base64: Optional[str] = Field(None, description="Base64 Data URI of Grad-CAM activation heatmap")
    gradcam_panel_base64: Optional[str] = Field(None, description="Base64 Data URI of 3-panel visualization figure")
    reason: Optional[str] = Field(None, description="Rejection reason code if rejected or uncertain")
    message: Optional[str] = Field(None, description="User-facing validation status message")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    error: str = Field(..., description="Human-readable error description")
