
import torch

from src.models.fusion.early_fusion import EarlyFusionModel
from src.models.fusion.intermediate_fusion import IntermediateFusionModel
from src.models.fusion.late_fusion import LateFusionModel


MRI_FEATURE_DIM = 256
PET_FEATURE_DIM = 256
NUM_CLASSES = 3
BATCH_SIZE = 4


def create_test_features():
    torch.manual_seed(42)

    mri_features = torch.randn(
        BATCH_SIZE,
        MRI_FEATURE_DIM,
    )

    pet_features = torch.randn(
        BATCH_SIZE,
        PET_FEATURE_DIM,
    )

    return mri_features, pet_features


def test_early_fusion():
    mri_features, pet_features = create_test_features()

    model = EarlyFusionModel(
        mri_feature_dim=MRI_FEATURE_DIM,
        pet_feature_dim=PET_FEATURE_DIM,
        num_classes=NUM_CLASSES,
    )

    output = model(
        mri_features,
        pet_features,
    )

    assert output.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )


def test_intermediate_fusion():
    mri_features, pet_features = create_test_features()

    model = IntermediateFusionModel(
        mri_feature_dim=MRI_FEATURE_DIM,
        pet_feature_dim=PET_FEATURE_DIM,
        num_classes=NUM_CLASSES,
    )

    output = model(
        mri_features,
        pet_features,
    )

    assert output.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )


def test_late_fusion():
    mri_features, pet_features = create_test_features()

    model = LateFusionModel(
        mri_feature_dim=MRI_FEATURE_DIM,
        pet_feature_dim=PET_FEATURE_DIM,
        num_classes=NUM_CLASSES,
    )

    (
        final_logits,
        mri_probabilities,
        pet_probabilities,
        fused_probabilities,
    ) = model(
        mri_features,
        pet_features,
    )

    assert final_logits.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    assert mri_probabilities.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    assert pet_probabilities.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    assert fused_probabilities.shape == (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    probability_sums = fused_probabilities.sum(dim=1)

    assert torch.allclose(
        probability_sums,
        torch.ones(BATCH_SIZE),
        atol=1e-6,
    )


def test_late_fusion_weights_are_normalized():
    model = LateFusionModel(
        mri_feature_dim=MRI_FEATURE_DIM,
        pet_feature_dim=PET_FEATURE_DIM,
        mri_weight=2.0,
        pet_weight=1.0,
    )

    assert (
        abs(
            model.mri_weight
            + model.pet_weight
            - 1.0
        )
        < 1e-6
    )


def test_late_fusion_rejects_zero_weights():
    try:
        LateFusionModel(
            mri_feature_dim=MRI_FEATURE_DIM,
            pet_feature_dim=PET_FEATURE_DIM,
            mri_weight=0.0,
            pet_weight=0.0,
        )
    except ValueError:
        return

    raise AssertionError(
        "LateFusionModel should reject zero total fusion weight."
    )
