"""Final MRI classifier model for AD / MCI / CN prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mri.mri_encoder import MultiViewMRIEncoder


class MultiViewMRIClassifier(nn.Module):
    """Subject-level MRI classifier using all three MRI views."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MultiViewMRIEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
            num_views=3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, sample: torch.Tensor, view_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(sample, view_valid)
        logits = self.classifier(embedding)
        return logits, embedding
