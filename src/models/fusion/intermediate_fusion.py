
import torch
import torch.nn as nn


class IntermediateFusionModel(nn.Module):
    """
    Intermediate Fusion model for combining MRI and PET features.

    Each modality is first transformed independently into a
    learned representation. These representations are then
    combined and passed through fusion layers for classification.
    """

    def __init__(
        self,
        mri_feature_dim,
        pet_feature_dim,
        modality_hidden_dim=128,
        fusion_hidden_dim=128,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_feature_dim, modality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pet_encoder = nn.Sequential(
            nn.Linear(pet_feature_dim, modality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                modality_hidden_dim * 2,
                fusion_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden_dim,
                fusion_hidden_dim // 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden_dim // 2,
                num_classes,
            ),
        )

    def forward(self, mri_features, pet_features):
        """
        Process MRI and PET independently, then fuse
        their learned representations.

        Args:
            mri_features:
                Shape: (batch_size, mri_feature_dim)

            pet_features:
                Shape: (batch_size, pet_feature_dim)

        Returns:
            Class logits.
            Shape: (batch_size, num_classes)
        """

        mri_representation = self.mri_encoder(
            mri_features
        )

        pet_representation = self.pet_encoder(
            pet_features
        )

        fused_representation = torch.cat(
            [
                mri_representation,
                pet_representation,
            ],
            dim=1,
        )

        logits = self.fusion(
            fused_representation
        )

        return logits
