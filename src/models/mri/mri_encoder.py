"""View-aware MRI feature encoder for a multi-view 2D CNN pipeline.

The pipeline uses a shared ResNet18 backbone across the three views (axial, coronal,
Sagittal). For each subject, a small set of representative slices is selected per view,
encoded independently, then aggregated by mean pooling before fusion into a single MRI
feature vector.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class SliceEncoder(nn.Module):
    """Shared 2D CNN encoder for one MRI slice."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W)"""
        features = self.backbone(x)
        return self.projector(features)


class MultiViewMRIEncoder(nn.Module):
    """Aggregate slice embeddings from axial, coronal, and sagittal views."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        pretrained: bool = True,
        num_views: int = 3,
    ):
        super().__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.slice_encoder = SliceEncoder(in_channels=in_channels, embedding_dim=feature_dim, pretrained=pretrained)

    def forward(self, sample: torch.Tensor, view_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """sample: (B, num_views, max_slices, C, H, W)"""
        batch_size = sample.shape[0]
        if view_valid is None:
            view_valid = torch.ones(batch_size, self.num_views, device=sample.device)

        view_outputs = []
        for view_idx in range(self.num_views):
            view_embeddings = torch.zeros(batch_size, self.feature_dim, device=sample.device, dtype=sample.dtype)
            for subject_idx in range(batch_size):
                if float(view_valid[subject_idx, view_idx]) <= 0:
                    continue
                view_input = sample[subject_idx, view_idx]
                nonzero_mask = torch.abs(view_input).sum(dim=(1, 2, 3)) > 1e-6
                actual_slices = view_input[nonzero_mask]
                if actual_slices.numel() == 0:
                    continue
                slice_emb = self.slice_encoder(actual_slices)
                view_embeddings[subject_idx] = slice_emb.mean(dim=0)
            view_outputs.append(view_embeddings)

        stacked = torch.stack(view_outputs, dim=1)  # (B, num_views, feature_dim)
        fused = stacked.mean(dim=1)
        return fused
