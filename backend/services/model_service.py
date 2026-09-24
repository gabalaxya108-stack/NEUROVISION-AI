"""
Model Service for Brain MRI Classification Backend.
Responsible for:
- Loading the trained EfficientNetB0 Keras model once at application startup.
- Maintaining model state in memory across requests.
- Exposing model metadata and class mapping.
- Providing core forward-pass inference.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure torch backend for Keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from src.config import (
    BEST_MODEL_PATH,
    CLASSES,
    ID2LABEL,
    LABEL2ID,
    IMAGE_SIZE,
    INPUT_SHAPE,
    BACKBONE_NAME,
)

logger = logging.getLogger("brain_tumor_backend.model_service")


class ModelService:
    """
    Singleton service managing the lifecycle of the trained brain tumor model.
    Loads checkpoint into memory on application startup and executes forward passes.
    """

    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[keras.Model] = None
        self._classes = CLASSES
        self._id2label = ID2LABEL
        self._label2id = LABEL2ID
        self._input_shape = list(INPUT_SHAPE)
        self._test_accuracy = 0.9281
        self._architecture = BACKBONE_NAME

    def load_model(self) -> None:
        """
        Loads the trained Keras model from disk.
        Raises RuntimeError with a clear message if loading fails.
        """
        if self.model is not None:
            logger.info("Model is already loaded.")
            return

        if not self.model_path.exists():
            error_msg = f"Model checkpoint not found at: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading trained brain tumor model from: {self.model_path}")
            self.model = keras.models.load_model(str(self.model_path))
            logger.info(f"Model successfully loaded into memory. Backbone: {self._architecture}")
        except Exception as exc:
            error_msg = f"Failed to initialize model from '{self.model_path}': {exc}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

    def is_loaded(self) -> bool:
        """Checks whether the model is loaded and ready for inference."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns model metadata including architecture, input shape, classes,
        empirical test accuracy, and educational research purpose.
        """
        return {
            "model": self._architecture,
            "input_shape": self._input_shape,
            "classes": self._classes,
            "test_accuracy": self._test_accuracy,
            "purpose": "educational/research image classification",
        }

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs model inference on a preprocessed input batch.
        Args:
            input_tensor: numpy array of shape (1, 224, 224, 3)
        Returns:
            numpy array of softmax probabilities of shape (4,)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Run forward pass (inference mode)
        probabilities = self.model.predict(input_tensor, verbose=0)[0]
        return probabilities


# Global model service instance
model_service = ModelService()
