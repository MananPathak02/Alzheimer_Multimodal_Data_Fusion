
import torch
import torch.nn as nn


class EarlyFusionModel(nn.Module):
    """
    Early Fusion model for combining MRI and PET feature vectors.

    MRI features and PET features are concatenated first.
    The combined representation is then passed through
    fully connected layers for classification.
    """

    def __init__(
        self,
        mri_feature_dim,
        pet_feature_dim,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.mri_feature_dim = mri_feature_dim
        self.pet_feature_dim = pet_feature_dim

        total_feature_dim = mri_feature_dim + pet_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, mri_features, pet_features):
        """
        Combine MRI and PET features and produce class logits.

        Args:
            mri_features: MRI feature tensor.
                Shape: (batch_size, mri_feature_dim)

            pet_features: PET feature tensor.
                Shape: (batch_size, pet_feature_dim)

        Returns:
            Class logits.
            Shape: (batch_size, num_classes)
        """

        fused_features = torch.cat(
            [mri_features, pet_features],
            dim=1,
        )

        logits = self.classifier(fused_features)

        return logits
