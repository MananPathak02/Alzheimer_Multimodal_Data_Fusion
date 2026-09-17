
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion.early_fusion import EarlyFusionModel
from src.models.fusion.intermediate_fusion import IntermediateFusionModel
from src.models.fusion.late_fusion import LateFusionModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "dummy"
    / "features"
    / "aligned_multimodal_features.csv"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fusion"
RESULTS_DIR = PROJECT_ROOT / "results" / "fusion"

MRI_FEATURE_DIM = 256
PET_FEATURE_DIM = 256
NUM_CLASSES = 3

BATCH_SIZE = 16
RANDOM_SEED = 42

LABEL_MAPPING = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}

REVERSE_LABEL_MAPPING = {
    0: "CN",
    1: "MCI",
    2: "AD",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Aligned feature file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "patient_id",
        "label",
        "split",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def get_feature_columns(df):
    mri_columns = [
        column
        for column in df.columns
        if column.startswith("mri_")
    ]

    pet_columns = [
        column
        for column in df.columns
        if column.startswith("pet_")
    ]

    if len(mri_columns) != MRI_FEATURE_DIM:
        raise ValueError(
            f"Expected {MRI_FEATURE_DIM} MRI features, "
            f"found {len(mri_columns)}."
        )

    if len(pet_columns) != PET_FEATURE_DIM:
        raise ValueError(
            f"Expected {PET_FEATURE_DIM} PET features, "
            f"found {len(pet_columns)}."
        )

    return mri_columns, pet_columns


def prepare_split(
    df,
    split,
    mri_columns,
    pet_columns,
):
    split_df = df[
        df["split"] == split
    ].copy()

    if split_df.empty:
        raise ValueError(
            f"No samples found for split: {split}"
        )

    labels = split_df["label"].map(
        LABEL_MAPPING
    )

    if labels.isna().any():
        raise ValueError(
            f"Unknown labels found in {split} split."
        )

    mri_features = split_df[
        mri_columns
    ].to_numpy(dtype=np.float32)

    pet_features = split_df[
        pet_columns
    ].to_numpy(dtype=np.float32)

    labels = labels.to_numpy(
        dtype=np.int64
    )

    return (
        split_df,
        mri_features,
        pet_features,
        labels,
    )


def create_loader(
    mri_features,
    pet_features,
    labels,
):
    dataset = TensorDataset(
        torch.tensor(
            mri_features,
            dtype=torch.float32,
        ),
        torch.tensor(
            pet_features,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def load_scalers(train_mri, train_pet):
    mri_scaler = StandardScaler()
    pet_scaler = StandardScaler()

    train_mri_scaled = (
        mri_scaler.fit_transform(
            train_mri
        )
    )

    train_pet_scaled = (
        pet_scaler.fit_transform(
            train_pet
        )
    )

    return (
        mri_scaler,
        pet_scaler,
        train_mri_scaled,
        train_pet_scaled,
    )


def create_model(
    model_name,
    checkpoint,
    device,
):
    if model_name == "early_fusion":
        model = EarlyFusionModel(
            mri_feature_dim=MRI_FEATURE_DIM,
            pet_feature_dim=PET_FEATURE_DIM,
            num_classes=NUM_CLASSES,
        )

    elif model_name == "intermediate_fusion":
        model = IntermediateFusionModel(
            mri_feature_dim=MRI_FEATURE_DIM,
            pet_feature_dim=PET_FEATURE_DIM,
            num_classes=NUM_CLASSES,
        )

    elif model_name == "late_fusion":
        model = LateFusionModel(
            mri_feature_dim=MRI_FEATURE_DIM,
            pet_feature_dim=PET_FEATURE_DIM,
            num_classes=NUM_CLASSES,
            mri_weight=checkpoint.get(
                "mri_weight",
                0.5,
            ),
            pet_weight=checkpoint.get(
                "pet_weight",
                0.5,
            ),
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def get_model_output(
    model_name,
    model,
    mri_features,
    pet_features,
):
    if model_name == "late_fusion":
        (
            final_logits,
            mri_probabilities,
            pet_probabilities,
            fused_probabilities,
        ) = model(
            mri_features,
            pet_features,
        )

        return (
            fused_probabilities,
            mri_probabilities,
            pet_probabilities,
        )

    logits = model(
        mri_features,
        pet_features,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return (
        probabilities,
        None,
        None,
    )


def generate_predictions(
    model_name,
    model,
    dataloader,
    split_df,
    device,
):
    all_probabilities = []
    all_mri_probabilities = []
    all_pet_probabilities = []
    all_labels = []

    with torch.no_grad():
        for (
            mri_features,
            pet_features,
            labels,
        ) in dataloader:

            mri_features = mri_features.to(
                device
            )

            pet_features = pet_features.to(
                device
            )

            (
                probabilities,
                mri_probabilities,
                pet_probabilities,
            ) = get_model_output(
                model_name,
                model,
                mri_features,
                pet_features,
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            if mri_probabilities is not None:
                all_mri_probabilities.extend(
                    mri_probabilities.cpu().numpy()
                )

            if pet_probabilities is not None:
                all_pet_probabilities.extend(
                    pet_probabilities.cpu().numpy()
                )

    all_probabilities = np.asarray(
        all_probabilities
    )

    predictions = np.argmax(
        all_probabilities,
        axis=1,
    )

    result = pd.DataFrame(
        {
            "patient_id": split_df[
                "patient_id"
            ].tolist(),
            "label": split_df[
                "label"
            ].tolist(),
            "split": split_df[
                "split"
            ].tolist(),
            "predicted_label": [
                REVERSE_LABEL_MAPPING[
                    prediction
                ]
                for prediction in predictions
            ],
            "prob_CN": all_probabilities[:, 0],
            "prob_MCI": all_probabilities[:, 1],
            "prob_AD": all_probabilities[:, 2],
        }
    )

    result["correct"] = (
        result["label"]
        == result["predicted_label"]
    )

    if model_name == "late_fusion":
        mri_probabilities = np.asarray(
            all_mri_probabilities
        )

        pet_probabilities = np.asarray(
            all_pet_probabilities
        )

        result["mri_prob_CN"] = (
            mri_probabilities[:, 0]
        )

        result["mri_prob_MCI"] = (
            mri_probabilities[:, 1]
        )

        result["mri_prob_AD"] = (
            mri_probabilities[:, 2]
        )

        result["pet_prob_CN"] = (
            pet_probabilities[:, 0]
        )

        result["pet_prob_MCI"] = (
            pet_probabilities[:, 1]
        )

        result["pet_prob_AD"] = (
            pet_probabilities[:, 2]
        )

    return result


def validate_predictions(
    predictions,
    expected_split,
):
    if predictions.empty:
        raise ValueError(
            f"No predictions generated for "
            f"{expected_split}."
        )

    if not (
        predictions["split"]
        == expected_split
    ).all():
        raise ValueError(
            f"Invalid split detected in "
            f"{expected_split} predictions."
        )

    if predictions["patient_id"].duplicated().any():
        raise ValueError(
            f"Duplicate patient IDs found in "
            f"{expected_split} predictions."
        )

    expected_probability_sum = (
        predictions["prob_CN"]
        + predictions["prob_MCI"]
        + predictions["prob_AD"]
    )

    if not np.allclose(
        expected_probability_sum,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"Probability values do not sum to "
            f"1.0 in {expected_split} predictions."
        )


def generate_model_outputs(
    model_name,
    device,
    df,
    mri_columns,
    pet_columns,
):
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"\nLoading {model_name} checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    (
        train_df,
        train_mri,
        train_pet,
        train_labels,
    ) = prepare_split(
        df,
        "train",
        mri_columns,
        pet_columns,
    )

    (
        validation_df,
        validation_mri,
        validation_pet,
        validation_labels,
    ) = prepare_split(
        df,
        "validation",
        mri_columns,
        pet_columns,
    )

    (
        test_df,
        test_mri,
        test_pet,
        test_labels,
    ) = prepare_split(
        df,
        "test",
        mri_columns,
        pet_columns,
    )

    # Fit scalers ONLY on the training split.
    mri_scaler = StandardScaler()
    pet_scaler = StandardScaler()

    train_mri = (
        mri_scaler.fit_transform(
            train_mri
        )
    )

    train_pet = (
        pet_scaler.fit_transform(
            train_pet
        )
    )

    validation_mri = (
        mri_scaler.transform(
            validation_mri
        )
    )

    validation_pet = (
        pet_scaler.transform(
            validation_pet
        )
    )

    test_mri = (
        mri_scaler.transform(
            test_mri
        )
    )

    test_pet = (
        pet_scaler.transform(
            test_pet
        )
    )

    model = create_model(
        model_name,
        checkpoint,
        device,
    )

    split_data = {
        "train": (
            train_df,
            train_mri,
            train_pet,
            train_labels,
        ),
        "validation": (
            validation_df,
            validation_mri,
            validation_pet,
            validation_labels,
        ),
        "test": (
            test_df,
            test_mri,
            test_pet,
            test_labels,
        ),
    }

    split_predictions = {}

    for split_name, (
        split_df,
        split_mri,
        split_pet,
        split_labels,
    ) in split_data.items():

        print(
            f"Generating {split_name} "
            f"predictions..."
        )

        loader = create_loader(
            split_mri,
            split_pet,
            split_labels,
        )

        predictions = generate_predictions(
            model_name,
            model,
            loader,
            split_df,
            device,
        )

        validate_predictions(
            predictions,
            split_name,
        )

        split_predictions[
            split_name
        ] = predictions

    output_directory = (
        RESULTS_DIR / model_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_predictions = (
        split_predictions["train"]
    )

    validation_predictions = (
        split_predictions["validation"]
    )

    test_predictions = (
        split_predictions["test"]
    )

    combined_predictions = pd.concat(
        [
            train_predictions,
            validation_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    combined_predictions = (
        combined_predictions.sort_values(
            by=[
                "split",
                "patient_id",
            ]
        ).reset_index(drop=True)
    )

    if len(combined_predictions) != len(df):
        raise ValueError(
            f"Combined prediction count "
            f"({len(combined_predictions)}) "
            f"does not match dataset count "
            f"({len(df)})."
        )

    if combined_predictions[
        "patient_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate patient IDs found "
            "in combined predictions."
        )

    train_predictions.to_csv(
        output_directory
        / "train_predictions.csv",
        index=False,
    )

    validation_predictions.to_csv(
        output_directory
        / "validation_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        output_directory
        / "test_predictions.csv",
        index=False,
    )

    combined_predictions.to_csv(
        output_directory
        / "predictions.csv",
        index=False,
    )

    history_source = (
        RESULTS_DIR
        / f"{model_name}_training_history.csv"
    )

    if history_source.exists():
        history_destination = (
            output_directory
            / "training_history.csv"
        )

        history = pd.read_csv(
            history_source
        )

        history.to_csv(
            history_destination,
            index=False,
        )

    print(
        f"\n{model_name} outputs:"
    )

    print(
        f"  Train:      {len(train_predictions)}"
    )

    print(
        f"  Validation: {len(validation_predictions)}"
    )

    print(
        f"  Test:       {len(test_predictions)}"
    )

    print(
        f"  Combined:   {len(combined_predictions)}"
    )

    print(
        f"  Saved to:   {output_directory}"
    )


def main():
    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    df = load_dataset()

    print(
        f"Dataset shape: {df.shape}"
    )

    mri_columns, pet_columns = (
        get_feature_columns(df)
    )

    expected_counts = {
        "train": 109,
        "validation": 23,
        "test": 24,
    }

    actual_counts = (
        df["split"].value_counts().to_dict()
    )

    for split, expected_count in (
        expected_counts.items()
    ):
        actual_count = actual_counts.get(
            split,
            0,
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Unexpected {split} count: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    models = [
        "early_fusion",
        "intermediate_fusion",
        "late_fusion",
    ]

    for model_name in models:
        generate_model_outputs(
            model_name,
            device,
            df,
            mri_columns,
            pet_columns,
        )

    print(
        "\nAll fusion prediction files "
        "generated successfully."
    )


if __name__ == "__main__":
    main()
