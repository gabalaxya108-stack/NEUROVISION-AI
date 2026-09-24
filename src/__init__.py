"""
Brain Tumor Classification package initialization.
Configures backend and exposes core modules.
"""

import os
# Ensure Keras uses PyTorch backend before any Keras module is imported
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

from src.config import (
    CLASSES,
    LABEL2ID,
    ID2LABEL,
    IMAGE_SIZE,
    RANDOM_SEED,
)
from src.preprocessing import (
    LetterboxResize,
    get_train_transforms,
    get_eval_transforms,
    denormalize_tensor,
)
from src.dataset import (
    BrainMRIDataset,
    KerasMRIDataset,
    create_splits,
    get_data_loaders,
    get_keras_datasets,
)
