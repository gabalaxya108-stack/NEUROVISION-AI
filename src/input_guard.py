"""
Brain MRI Input Guardrail Module.
Provides pre-inference validation layer to intercept and reject non-brain MRI inputs
(e.g., selfies, photographs, documents, screenshots, non-brain radiographs, corrupted files)
before they can reach the EfficientNetB0 brain tumor classification model.

Educational & Safety Notice:
The input guardrail is an image-modality validation mechanism, not a medical diagnostic system.
A guardrail cannot guarantee that an uploaded image is clinically appropriate, correctly acquired,
or from a particular patient or imaging protocol.
"""

import io
import math
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
from PIL import Image
import numpy as np


class GuardrailStatus(str, Enum):
    VALID_MRI = "VALID_MRI"
    INVALID_NON_MRI = "INVALID_NON_MRI"
    UNCERTAIN = "UNCERTAIN"


class GuardrailResult:
    """Encapsulates the decision of the input validation guardrail."""

    def __init__(
        self,
        status: GuardrailStatus,
        accepted: bool,
        input_type: str,
        reason: Optional[str] = None,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self.accepted = accepted
        self.input_type = input_type
        self.reason = reason
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Formats the result conforming to the project API response specification."""
        if self.accepted:
            return {
                "accepted": True,
                "input_type": self.input_type,
                "message": self.message,
            }
        return {
            "accepted": False,
            "input_type": self.input_type,
            "reason": self.reason,
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"<GuardrailResult status={self.status} accepted={self.accepted} reason={self.reason}>"


class BrainMRIInputGuard:
    """
    Multi-stage heuristic & morphological input guardrail for Brain MRI verification.
    Evaluates:
    1. Decoding & format integrity
    2. Chromatic saturation (MRI grayscale verification)
    3. Background dark-perimeter framing (cranial contrast)
    4. Grayscale tissue intensity histogram & entropy
    5. Morphological cranial geometry (convexity, centeredness, area occupancy)
    6. High-frequency rectilinear edge density (document/screenshot detection)
    """

    def __init__(
        self,
        min_dimension: int = 80,
        max_aspect_ratio: float = 2.6,
        max_mean_saturation: float = 0.12,
        max_saturated_pixel_ratio: float = 0.06,
        max_corner_brightness: float = 0.45,
        min_foreground_ratio: float = 0.12,
        max_foreground_ratio: float = 0.92,
        max_rectilinear_edge_ratio: float = 0.40,
    ):
        self.min_dimension = min_dimension
        self.max_aspect_ratio = max_aspect_ratio
        self.max_mean_saturation = max_mean_saturation
        self.max_saturated_pixel_ratio = max_saturated_pixel_ratio
        self.max_corner_brightness = max_corner_brightness
        self.min_foreground_ratio = min_foreground_ratio
        self.max_foreground_ratio = max_foreground_ratio
        self.max_rectilinear_edge_ratio = max_rectilinear_edge_ratio

    def _decode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Decodes image into PIL Image in RGB format."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image file does not exist: {p}")
            with Image.open(p) as img:
                return img.convert("RGB")
        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ValueError("Image data is empty (0 bytes).")
            bio = io.BytesIO(image_input)
            img = Image.open(bio)
            img.verify()
            bio.seek(0)
            img = Image.open(bio)
            return img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def validate(self, image_input: Union[str, Path, bytes, Image.Image]) -> GuardrailResult:
        """
        Executes complete multi-stage guardrail validation.
        Returns GuardrailResult with status VALID_MRI, INVALID_NON_MRI, or UNCERTAIN.
        """
        # --- Stage 1: File & Image Decoding ---
        try:
            rgb_img = self._decode_image(image_input)
        except Exception as exc:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="corrupted_or_invalid",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a valid brain MRI image.",
                details={"stage": "decoding", "error": str(exc)},
            )

        w, h = rgb_img.size

        # Dimension checks
        if w < self.min_dimension or h < self.min_dimension:
            return GuardrailResult(
                status=GuardrailStatus.UNCERTAIN,
                accepted=False,
                input_type="uncertain",
                reason="uncertain_input",
                message="The image could not be confidently verified as a brain MRI. Please upload a clear brain MRI image.",
                details={"stage": "dimensions", "reason": f"Resolution too low ({w}x{h} px)"},
            )

        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > self.max_aspect_ratio:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={"stage": "aspect_ratio", "aspect_ratio": round(aspect_ratio, 2)},
            )

        arr = np.array(rgb_img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]

        # --- Stage 2: Chromatic Saturation Analysis (Grayscale Test) ---
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Evaluate saturation on non-black pixels (V > 0.10) to avoid dark JPEG compression noise
        tissue_pixels = max_c > 0.10
        if np.any(tissue_pixels):
            sat_tissue = diff[tissue_pixels] / (max_c[tissue_pixels] + 1e-7)
            mean_saturation = float(np.mean(sat_tissue))
            saturated_pixel_ratio = float(np.mean(sat_tissue > 0.25))
        else:
            mean_saturation = 0.0
            saturated_pixel_ratio = 0.0

        if mean_saturation > self.max_mean_saturation or saturated_pixel_ratio > self.max_saturated_pixel_ratio:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={
                    "stage": "color_saturation",
                    "mean_saturation": round(mean_saturation, 4),
                    "saturated_pixel_ratio": round(saturated_pixel_ratio, 4),
                },
            )

        # Grayscale luminance (0.299 R + 0.587 G + 0.114 B)
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # --- Stage 3: Background Perimeter & Corner Dark Framing ---
        # Brain MRIs are centered inside a dark scan chamber; corners are dark air.
        # Documents, screenshots, and bright photos have white/light corners.
        corner_size_y = max(4, int(h * 0.08))
        corner_size_x = max(4, int(w * 0.08))

        top_left = gray[:corner_size_y, :corner_size_x]
        top_right = gray[:corner_size_y, -corner_size_x:]
        bottom_left = gray[-corner_size_y:, :corner_size_x:]
        bottom_right = gray[-corner_size_y:, -corner_size_x:]

        corners = [top_left, top_right, bottom_left, bottom_right]
        mean_corner_intensity = float(np.mean([np.mean(c) for c in corners]))
        max_corner_intensity = float(np.max([np.mean(c) for c in corners]))

        if mean_corner_intensity > self.max_corner_brightness or max_corner_intensity > 0.65:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={
                    "stage": "corner_background",
                    "mean_corner_intensity": round(mean_corner_intensity, 4),
                    "max_corner_intensity": round(max_corner_intensity, 4),
                },
            )

        # Perimeter border brightness check
        border_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        mean_border_intensity = float(np.mean(border_pixels))
        if mean_border_intensity > 0.45:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={"stage": "border_intensity", "mean_border_intensity": round(mean_border_intensity, 4)},
            )

        # --- Stage 4: Tissue Grayscale Histogram & Entropy ---
        # Distinguish soft-tissue gradient from strictly bimodal text documents or flat graphics
        hist, bin_edges = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
        hist_norm = hist / (np.sum(hist) + 1e-8)
        non_zero_probs = hist_norm[hist_norm > 0]
        shannon_entropy = float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))

        # Documents or solid blanks have extremely low entropy (< 1.6)
        if shannon_entropy < 1.6:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={"stage": "shannon_entropy", "entropy": round(shannon_entropy, 3)},
            )

        # --- Stage 5: Morphological Cranial Geometry & Foreground Occupancy ---
        # Foreground threshold: pixels noticeably above background air level
        air_level = max(0.04, float(np.percentile(border_pixels, 60)))
        tissue_threshold = max(air_level + 0.05, 0.10)
        foreground_mask = gray > tissue_threshold
        foreground_ratio = float(np.mean(foreground_mask))

        # Brain tissue occupies an intermediate area (not empty 0% and not full canvas 95%+)
        if foreground_ratio < self.min_foreground_ratio or foreground_ratio > self.max_foreground_ratio:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={"stage": "foreground_occupancy", "foreground_ratio": round(foreground_ratio, 4)},
            )

        # Check foreground centroid centeredness
        ys, xs = np.nonzero(foreground_mask)
        if len(xs) > 0:
            centroid_x = float(np.mean(xs)) / w
            centroid_y = float(np.mean(ys)) / h
            dist_from_center = math.sqrt((centroid_x - 0.5) ** 2 + (centroid_y - 0.5) ** 2)
            if dist_from_center > 0.28:
                return GuardrailResult(
                    status=GuardrailStatus.INVALID_NON_MRI,
                    accepted=False,
                    input_type="non_brain_mri",
                    reason="non_brain_mri",
                    message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                    details={"stage": "centroid_offset", "distance_from_center": round(dist_from_center, 3)},
                )

        # --- Stage 6: High-Frequency Rectilinear Edge Density (Text/Screenshot Detection) ---
        # Compute horizontal & vertical discrete gradient differences
        diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_y = np.abs(gray[1:, :] - gray[:-1, :])

        # Strong edges threshold
        strong_edge_x = diff_x > 0.20
        strong_edge_y = diff_y > 0.20
        rectilinear_edge_density = float(np.mean(strong_edge_x) + np.mean(strong_edge_y))

        if rectilinear_edge_density > self.max_rectilinear_edge_ratio:
            return GuardrailResult(
                status=GuardrailStatus.INVALID_NON_MRI,
                accepted=False,
                input_type="non_brain_mri",
                reason="non_brain_mri",
                message="This image does not appear to be a brain MRI. Please upload a brain MRI image for analysis.",
                details={"stage": "rectilinear_edges", "rectilinear_density": round(rectilinear_edge_density, 4)},
            )

        # --- Stage 7: Ambiguity & Uncertainty Filter ---
        # Marginal cases where metrics hover near limits are flagged as UNCERTAIN
        is_marginal_corners = (0.28 < mean_corner_intensity <= self.max_corner_brightness)
        is_marginal_occupancy = (foreground_ratio < 0.18 or foreground_ratio > 0.85)
        is_marginal_entropy = (1.6 <= shannon_entropy < 2.1)

        if is_marginal_corners or is_marginal_occupancy or is_marginal_entropy:
            return GuardrailResult(
                status=GuardrailStatus.UNCERTAIN,
                accepted=False,
                input_type="uncertain",
                reason="uncertain_input",
                message="The image could not be confidently verified as a brain MRI. Please upload a clear brain MRI image.",
                details={
                    "stage": "uncertainty_boundary",
                    "mean_corner_intensity": round(mean_corner_intensity, 3),
                    "foreground_ratio": round(foreground_ratio, 3),
                    "entropy": round(shannon_entropy, 3),
                },
            )

        # --- All Checks Passed: Confirmed Brain MRI ---
        return GuardrailResult(
            status=GuardrailStatus.VALID_MRI,
            accepted=True,
            input_type="brain_mri",
            message="Brain MRI image verified.",
            details={
                "mean_saturation": round(mean_saturation, 4),
                "foreground_ratio": round(foreground_ratio, 3),
                "entropy": round(shannon_entropy, 3),
                "mean_corner_intensity": round(mean_corner_intensity, 4),
            },
        )


# Global guardrail instance
input_guard = BrainMRIInputGuard()
