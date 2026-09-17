
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


RANDOM_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30

MRI_FEATURE_DIM = 256
PET_FEATURE_DIM = 256
NUM_CLASSES = 3

LABEL_MAPPING = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
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

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def prepare_features(df):
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


def create_split_data(
    df,
    split,
    mri_columns,
    pet_columns,
):
    split_df = df[df["split"] == split].copy()

    if split_df.empty:
        raise ValueError(
            f"No samples found for split: {split}"
        )

    mri_features = split_df[
        mri_columns
    ].to_numpy(dtype=np.float32)

    pet_features = split_df[
        pet_columns
    ].to_numpy(dtype=np.float32)

    labels = split_df["label"].map(
        LABEL_MAPPING
    )

    if labels.isna().any():
        unknown_labels = split_df.loc[
            labels.isna(),
            "label",
        ].unique()

        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    labels = labels.to_numpy(dtype=np.int64)

    return (
        split_df,
        mri_features,
        pet_features,
        labels,
    )


def create_dataloader(
    mri_features,
    pet_features,
    labels,
    shuffle,
):
    mri_tensor = torch.tensor(
        mri_features,
        dtype=torch.float32,
    )

    pet_tensor = torch.tensor(
        pet_features,
        dtype=torch.float32,
    )

    label_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        mri_tensor,
        pet_tensor,
        label_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for (
        mri_features,
        pet_features,
        labels,
    ) in dataloader:

        mri_features = mri_features.to(device)
        pet_features = pet_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        (
            final_logits,
            _,
            _,
            _,
        ) = model(
            mri_features,
            pet_features,
        )

        loss = criterion(
            final_logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            final_logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (
            mri_features,
            pet_features,
            labels,
        ) in dataloader:

            mri_features = mri_features.to(device)
            pet_features = pet_features.to(device)
            labels = labels.to(device)

            (
                final_logits,
                _,
                _,
                _,
            ) = model(
                mri_features,
                pet_features,
            )

            loss = criterion(
                final_logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                final_logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def generate_predictions(
    model,
    dataloader,
    patient_ids,
    labels,
    splits,
    device,
):
    model.eval()

    predictions = []
    probabilities = []
    mri_probabilities = []
    pet_probabilities = []

    with torch.no_grad():
        for (
            mri_features,
            pet_features,
            _,
        ) in dataloader:

            mri_features = mri_features.to(device)
            pet_features = pet_features.to(device)

            (
                _,
                mri_probs,
                pet_probs,
                fused_probs,
            ) = model(
                mri_features,
                pet_features,
            )

            preds = torch.argmax(
                fused_probs,
                dim=1,
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            probabilities.extend(
                fused_probs.cpu().numpy()
            )

            mri_probabilities.extend(
                mri_probs.cpu().numpy()
            )

            pet_probabilities.extend(
                pet_probs.cpu().numpy()
            )

    prediction_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": labels,
            "split": splits,
            "predicted_label": predictions,
            "prob_CN": [
                probability[0]
                for probability in probabilities
            ],
            "prob_MCI": [
                probability[1]
                for probability in probabilities
            ],
            "prob_AD": [
                probability[2]
                for probability in probabilities
            ],
            "mri_prob_CN": [
                probability[0]
                for probability in mri_probabilities
            ],
            "mri_prob_MCI": [
                probability[1]
                for probability in mri_probabilities
            ],
            "mri_prob_AD": [
                probability[2]
                for probability in mri_probabilities
            ],
            "pet_prob_CN": [
                probability[0]
                for probability in pet_probabilities
            ],
            "pet_prob_MCI": [
                probability[1]
                for probability in pet_probabilities
            ],
            "pet_prob_AD": [
                probability[2]
                for probability in pet_probabilities
            ],
        }
    )

    prediction_df["correct"] = (
        prediction_df["label"].map(
            LABEL_MAPPING
        )
        == prediction_df["predicted_label"]
    )

    reverse_label_mapping = {
        0: "CN",
        1: "MCI",
        2: "AD",
    }

    prediction_df["predicted_label"] = (
        prediction_df["predicted_label"].map(
            reverse_label_mapping
        )
    )

    return prediction_df


def main():
    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    df = load_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    (
        mri_columns,
        pet_columns,
    ) = prepare_features(df)

    (
        train_df,
        train_mri,
        train_pet,
        train_labels,
    ) = create_split_data(
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
    ) = create_split_data(
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
    ) = create_split_data(
        df,
        "test",
        mri_columns,
        pet_columns,
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    # Fit scalers ONLY on training data.
    mri_scaler = StandardScaler()
    pet_scaler = StandardScaler()

    train_mri = mri_scaler.fit_transform(
        train_mri
    )

    validation_mri = mri_scaler.transform(
        validation_mri
    )

    test_mri = mri_scaler.transform(
        test_mri
    )

    train_pet = pet_scaler.fit_transform(
        train_pet
    )

    validation_pet = pet_scaler.transform(
        validation_pet
    )

    test_pet = pet_scaler.transform(
        test_pet
    )

    train_loader = create_dataloader(
        train_mri,
        train_pet,
        train_labels,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        validation_mri,
        validation_pet,
        validation_labels,
        shuffle=False,
    )

    test_loader = create_dataloader(
        test_mri,
        test_pet,
        test_labels,
        shuffle=False,
    )

    model = LateFusionModel(
        mri_feature_dim=MRI_FEATURE_DIM,
        pet_feature_dim=PET_FEATURE_DIM,
        hidden_dim=128,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        mri_weight=0.5,
        pet_weight=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    best_validation_accuracy = 0.0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / "late_fusion_best.pt"
    )

    print(
        "\nStarting Late Fusion training...\n"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mri_feature_dim": MRI_FEATURE_DIM,
                    "pet_feature_dim": PET_FEATURE_DIM,
                    "num_classes": NUM_CLASSES,
                    "mri_weight": model.mri_weight,
                    "pet_weight": model.pet_weight,
                    "label_mapping": LABEL_MAPPING,
                    "mri_scaler_mean": mri_scaler.mean_,
                    "mri_scaler_scale": mri_scaler.scale_,
                    "pet_scaler_mean": pet_scaler.mean_,
                    "pet_scaler_scale": pet_scaler.scale_,
                },
                checkpoint_path,
            )

    history_df = pd.DataFrame(history)

    history_path = (
        RESULTS_DIR
        / "late_fusion_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print("\nTraining complete.")

    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Model saved to: {checkpoint_path}"
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_accuracy = validate(
        model,
        test_loader,
        criterion,
        device,
    )

    test_predictions = generate_predictions(
        model,
        test_loader,
        test_df["patient_id"].tolist(),
        test_df["label"].tolist(),
        test_df["split"].tolist(),
        device,
    )

    prediction_dir = (
        RESULTS_DIR
        / "late_fusion"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        prediction_dir
        / "predictions.csv"
    )

    test_predictions.to_csv(
        prediction_path,
        index=False,
    )

    print("\nTest evaluation:")

    print(
        f"Test loss: {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Predictions saved to: "
        f"{prediction_path}"
    )


if __name__ == "__main__":
    main()
