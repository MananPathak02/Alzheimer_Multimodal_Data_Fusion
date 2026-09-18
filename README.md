# Alzheimer Multimodal Data Fusion

## MRI pipeline (Anuj Singh)

This repository branch focuses on the MRI single-modality pipeline for Alzheimer's disease detection. The MRI workflow follows a strict subject-level design so that all slices from the same subject stay in one split and never leak across train/validation/test.

### Data requirements

Set the dataset paths with environment variables:

```bash
export MRI_METADATA="/path/to/ADNI-1_Baseline_MRI_3_31_2026.csv"
export MRI_DATA_DIR="/path/to/slices_multiview"
```

The image data should contain:

- `axial/`
- `coronal/`
- `sagittal/`

Each file name is expected to contain a subject ID in the format `XXX_S_XXXX`.

### Subject-level preprocessing

Run:

```bash
python scripts/preprocess_mri.py
```

This scans the CSV metadata and the MRI folders, matches subjects, assigns the label at the patient level, and creates the split files in `data/splits/` and `outputs/split.csv`.

### Training

Run:

```bash
python scripts/train_mri.py --config configs/mri_config.yaml
```

This trains a multi-view 2D CNN with axial, coronal, and sagittal inputs using a ResNet18 backbone and an Apple Silicon/MPS-aware device selection strategy.

### MRI embeddings and predictions

After a checkpoint exists, generate the MRI feature vectors and subject-level predictions with:

```bash
python scripts/generate_mri_features.py --config configs/mri_config.yaml --checkpoint outputs/checkpoints/best_mri_model.pth
```

Outputs include:

- `outputs/mri_features.csv`
- `outputs/mri_predictions.csv`
- `outputs/checkpoints/best_mri_model.pth`
- `outputs/mri_metrics.json`

### Fusion interface

The MRI feature file is saved as a row-per-subject table with columns like:

- `subject_id`
- `group`
- `split`
- `feature_0`, `feature_1`, ... `feature_255`

The fusion team can load `subject_id` + the MRI embedding vector and combine it with PET embeddings later.

### Notes

- No slice is treated as an independent patient.
- The train/validation/test split is created at the subject level.
- Missing views are handled with zero-filled masks rather than dropping subjects unnecessarily.
- The implementation supports CUDA, MPS, or CPU automatically.
