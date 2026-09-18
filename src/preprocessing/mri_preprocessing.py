"""
MRI Preprocessing and Augmentation Module.

Provides robust, reproducible preprocessing and augmentations for brain MRI slices.
Ensures zero data leakage: augmentations are applied strictly to training data only.
"""

import os
from typing import Tuple, Optional
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class MRINormalize(torch.nn.Module):
    """
    Min-max or mean-std normalization tailored for brain MRI grayscale intensities.
    Maps pixel values to standard zero-mean or standard dynamic range.
    """
    def __init__(self, mean: float = 0.25, std: float = 0.25):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Standardize: (x - mean) / std
        return (tensor - self.mean) / (self.std + 1e-7)


def get_mri_transforms(
    is_training: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    in_channels: int = 1,
    mean: float = 0.25,
    std: float = 0.25,
    rotation_degrees: float = 10.0,
    allow_horizontal_flip: bool = True,
) -> T.Compose:
    """
    Construct torchvision transform pipeline for MRI slices.

    Args:
        is_training: If True, applies data augmentations (rotation, subtle affine, flip).
                     If False, applies only deterministic resize and normalization.
        image_size: Target (height, width), default (224, 224).
        in_channels: 1 for grayscale, 3 for RGB replication.
        mean: Normalization mean.
        std: Normalization standard deviation.
        rotation_degrees: Maximum degree for random rotation during training.
        allow_horizontal_flip: Whether to allow random horizontal flip during training.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []

    # 1. Resize to canonical dimensions
    transform_list.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))

    # 2. Training augmentations (applied ONLY when is_training is True)
    if is_training:
        if allow_horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
        if rotation_degrees > 0:
            transform_list.append(
                T.RandomRotation(
                    degrees=(-rotation_degrees, rotation_degrees),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            )
        transform_list.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
                interpolation=T.InterpolationMode.BILINEAR,
            )
        )

    # 3. Convert PIL Image to PyTorch Tensor [0.0, 1.0]
    transform_list.append(T.ToTensor())

    # 4. Handle channel replication if in_channels == 3
    if in_channels == 3:
        transform_list.append(T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t))

    # 5. Intensity Normalization
    transform_list.append(MRINormalize(mean=mean, std=std))

    return T.Compose(transform_list)


def load_and_preprocess_slice(
    filepath: str,
    transform: Optional[T.Compose] = None,
    image_size: Tuple[int, int] = (224, 224),
    in_channels: int = 1,
) -> torch.Tensor:
    """
    Safely load a single PNG slice, convert to grayscale, and apply transform.
    Returns a zero tensor if the file is missing or corrupted.
    """
    if not os.path.exists(filepath):
        # Fallback to zeros if missing
        return torch.zeros((in_channels, image_size[0], image_size[1]), dtype=torch.float32)

    try:
        with Image.open(filepath) as img:
            img = img.convert("L")  # Ensure 8-bit grayscale
            if transform is not None:
                tensor = transform(img)
            else:
                tensor = TF.to_tensor(img)
            return tensor
    except Exception as e:
        print(f"Warning: Corrupted image at {filepath}: {e}")
        return torch.zeros((in_channels, image_size[0], image_size[1]), dtype=torch.float32)


if __name__ == "__main__":
    # Self-test unit check
    print("Testing MRI Preprocessing Transforms...")
    dummy_img = Image.new("L", (224, 224), color=128)
    
    train_tf = get_mri_transforms(is_training=True, in_channels=1)
    val_tf = get_mri_transforms(is_training=False, in_channels=1)
    
    t_train = train_tf(dummy_img)
    t_val = val_tf(dummy_img)
    
    print(f"Train tensor shape: {t_train.shape}, dtype: {t_train.dtype}, min: {t_train.min():.2f}, max: {t_train.max():.2f}")
    print(f"Val tensor shape:   {t_val.shape}, dtype: {t_val.dtype}, min: {t_val.min():.2f}, max: {t_val.max():.2f}")
    assert t_train.shape == (1, 224, 224), "Incorrect train tensor shape"
    assert t_val.shape == (1, 224, 224), "Incorrect val tensor shape"
    print("MRI Preprocessing self-test passed successfully!")
