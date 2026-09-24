"""
Data preprocessing and augmentation pipelines for Brain MRI images.
Provides aspect-ratio preserving letterbox resizing, RGB conversion,
and conservative training augmentations suitable for brain MRI scans.
"""

from typing import Tuple, Union
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from src.config import (
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    AUGMENTATION_CONFIG,
)


class LetterboxResize:
    """
    Resizes an image while strictly preserving its original aspect ratio,
    placing the resized image on a centered black canvas of target_size.
    
    Medical MRI Rationale:
    1. Prevents geometric distortion of anatomical structures and tumor shapes.
    2. Prevents cropping out peripheral brain tissues, meninges, or skull boundaries.
    3. Natural match for MRI, where the background field of view is already zero/black.
    """

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE, fill_color: Tuple[int, int, int] = (0, 0, 0)):
        self.target_w, self.target_h = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == self.target_w and h == self.target_h:
            return img

        # Compute scaling factor to fit within target boundaries
        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # High-quality bilinear resize
        resized_img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Create target canvas and paste centered
        canvas = Image.new("RGB", (self.target_w, self.target_h), self.fill_color)
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        canvas.paste(resized_img, (paste_x, paste_y))

        return canvas


def get_train_transforms() -> transforms.Compose:
    """
    Returns PyTorch training transforms:
    LetterboxResize -> Conservative Augmentation -> Tensor Conversion -> Normalization.
    """
    cfg = AUGMENTATION_CONFIG
    return transforms.Compose([
        LetterboxResize(target_size=IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=cfg["horizontal_flip_p"]),
        transforms.RandomAffine(
            degrees=cfg["rotation_degrees"],
            translate=cfg["translate"],
            scale=cfg["scale"],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=cfg["brightness"],
            contrast=cfg["contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Returns strictly deterministic evaluation transforms for validation and testing:
    LetterboxResize -> Tensor Conversion -> Normalization.
    No random augmentations are applied.
    """
    return transforms.Compose([
        LetterboxResize(target_size=IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_pil_train_augmentation() -> transforms.Compose:
    """
    Returns image-level PIL transform pipeline for training (used in Keras PyDataset).
    Outputs a PIL Image of size (224, 224, 3).
    """
    cfg = AUGMENTATION_CONFIG
    return transforms.Compose([
        LetterboxResize(target_size=IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=cfg["horizontal_flip_p"]),
        transforms.RandomAffine(
            degrees=cfg["rotation_degrees"],
            translate=cfg["translate"],
            scale=cfg["scale"],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=cfg["brightness"],
            contrast=cfg["contrast"],
        ),
    ])


def get_pil_eval_preprocessing() -> transforms.Compose:
    """
    Returns deterministic image-level PIL transform for evaluation (validation/testing).
    """
    return transforms.Compose([
        LetterboxResize(target_size=IMAGE_SIZE),
    ])


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverts the ImageNet normalization to return pixel values in [0, 1] range for visualization.
    Accepts (C, H, W) or (B, C, H, W) tensor.
    """
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    unnorm = tensor * std + mean
    return torch.clamp(unnorm, 0.0, 1.0)
