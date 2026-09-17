# Load MRI feature CSV.
# Load PET feature CSV.
# Match them using patient_id.
# Verify that labels match.
# Verify that train/validation/test splits match.
# Prefix MRI features with mri_.
# Prefix PET features with pet_.
# Produce one combined dataset.

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MRI_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "dummy"
    / "features"
    / "mri_features.csv"
)

PET_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "dummy"
    / "features"
    / "pet_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "dummy"
    / "features"
    / "aligned_multimodal_features.csv"
)


def load_features(path):
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    return pd.read_csv(path)


def validate_columns(df, modality):
    required_columns = {"patient_id", "label", "split"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{modality} features are missing columns: "
            f"{sorted(missing_columns)}"
        )


def validate_unique_patients(df, modality):
    if df["patient_id"].duplicated().any():
        duplicated = df.loc[
            df["patient_id"].duplicated(),
            "patient_id"
        ].tolist()

        raise ValueError(
            f"{modality} contains duplicate patient IDs: {duplicated}"
        )


def align_features(mri_df, pet_df):
    validate_columns(mri_df, "MRI")
    validate_columns(pet_df, "PET")

    validate_unique_patients(mri_df, "MRI")
    validate_unique_patients(pet_df, "PET")

    mri_patients = set(mri_df["patient_id"])
    pet_patients = set(pet_df["patient_id"])

    missing_in_pet = mri_patients - pet_patients
    missing_in_mri = pet_patients - mri_patients

    if missing_in_pet:
        raise ValueError(
            "Patients present in MRI but missing in PET: "
            f"{sorted(missing_in_pet)}"
        )

    if missing_in_mri:
        raise ValueError(
            "Patients present in PET but missing in MRI: "
            f"{sorted(missing_in_mri)}"
        )

    mri = mri_df.copy()
    pet = pet_df.copy()

    mri_feature_columns = [
        column
        for column in mri.columns
        if column not in {"patient_id", "label", "split"}
    ]

    pet_feature_columns = [
        column
        for column in pet.columns
        if column not in {"patient_id", "label", "split"}
    ]

    mri = mri.rename(
        columns={
            column: f"mri_{column}"
            for column in mri_feature_columns
        }
    )

    pet = pet.rename(
        columns={
            column: f"pet_{column}"
            for column in pet_feature_columns
        }
    )

    merged = pd.merge(
        mri,
        pet,
        on="patient_id",
        how="inner",
        suffixes=("_mri", "_pet"),
    )

    if merged.empty:
        raise ValueError(
            "No patients could be aligned between MRI and PET."
        )

    if not (merged["label_mri"] == merged["label_pet"]).all():
        mismatched = merged.loc[
            merged["label_mri"] != merged["label_pet"],
            ["patient_id", "label_mri", "label_pet"],
        ]

        raise ValueError(
            "MRI and PET labels do not match:\n"
            f"{mismatched.to_string(index=False)}"
        )

    if not (merged["split_mri"] == merged["split_pet"]).all():
        mismatched = merged.loc[
            merged["split_mri"] != merged["split_pet"],
            ["patient_id", "split_mri", "split_pet"],
        ]

        raise ValueError(
            "MRI and PET splits do not match:\n"
            f"{mismatched.to_string(index=False)}"
        )

    aligned = merged[
        ["patient_id", "label_mri", "split_mri"]
        + [
            column
            for column in merged.columns
            if column.startswith("mri_")
        ]
        + [
            column
            for column in merged.columns
            if column.startswith("pet_")
        ]
    ].copy()

    aligned = aligned.rename(
        columns={
            "label_mri": "label",
            "split_mri": "split",
        }
    )

    return aligned


def main():
    print("Loading MRI features...")
    mri_df = load_features(MRI_FEATURES_PATH)

    print("Loading PET features...")
    pet_df = load_features(PET_FEATURES_PATH)

    print(f"MRI shape: {mri_df.shape}")
    print(f"PET shape: {pet_df.shape}")

    print("Aligning MRI and PET features...")

    aligned_df = align_features(mri_df, pet_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    aligned_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Alignment successful.")
    print(f"Aligned shape: {aligned_df.shape}")
    print(f"Patients: {len(aligned_df)}")

    print()
    print("Class distribution:")
    print(aligned_df["label"].value_counts())

    print()
    print("Split distribution:")
    print(aligned_df["split"].value_counts())

    print()
    print("Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()