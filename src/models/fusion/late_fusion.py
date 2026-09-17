
import torch
import torch.nn as nn


class ModalityClassifier(nn.Module):
    """
    Standalone classifier for one modality.

    This model takes a modality-specific feature vector
    and produces classification logits for CN, MCI, and AD.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


class LateFusionModel(nn.Module):
    """
    Late Fusion model.

    MRI and PET are classified independently first.
    Their probability distributions are then combined.

    Default fusion strategy:
        Weighted average of MRI and PET probabilities.
    """

    def __init__(
        self,
        mri_feature_dim,
        pet_feature_dim,
        hidden_dim=128,
        num_classes=3,
        dropout=0.3,
        mri_weight=0.5,
        pet_weight=0.5,
    ):
        super().__init__()

        if mri_weight < 0 or pet_weight < 0:
            raise ValueError(
                "Fusion weights must be non-negative."
            )

        total_weight = mri_weight + pet_weight

        if total_weight == 0:
            raise ValueError(
                "At least one fusion weight must be greater than zero."
            )

        self.mri_weight = mri_weight / total_weight
        self.pet_weight = pet_weight / total_weight

        self.mri_classifier = ModalityClassifier(
            feature_dim=mri_feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.pet_classifier = ModalityClassifier(
            feature_dim=pet_feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, mri_features, pet_features):
        """
        Generate independent MRI and PET predictions,
        then combine their probabilities.

        Args:
            mri_features:
                Shape: (batch_size, mri_feature_dim)

            pet_features:
                Shape: (batch_size, pet_feature_dim)

        Returns:
            final_logits:
                Combined logits.

            mri_probabilities:
                MRI-only probability distribution.

            pet_probabilities:
                PET-only probability distribution.

            fused_probabilities:
                Final late-fusion probability distribution.
        """

        mri_logits = self.mri_classifier(
            mri_features
        )

        pet_logits = self.pet_classifier(
            pet_features
        )

        mri_probabilities = torch.softmax(
            mri_logits,
            dim=1,
        )

        pet_probabilities = torch.softmax(
            pet_logits,
            dim=1,
        )

        fused_probabilities = (
            self.mri_weight * mri_probabilities
            + self.pet_weight * pet_probabilities
        )

        final_logits = torch.log(
            fused_probabilities.clamp(min=1e-8)
        )

        return (
            final_logits,
            mri_probabilities,
            pet_probabilities,
            fused_probabilities,
        )