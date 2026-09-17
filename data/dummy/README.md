# Dummy Multimodal Feature Dataset

Development-only synthetic feature data for testing the MRI/PET fusion and
evaluation pipelines before real modality encoder outputs are available.

The subject IDs, labels, and patient-level split structure are based on the
uploaded MRI dataset. Feature values are synthetic and are NOT medical data.

## Layout

- `metadata/`: modality metadata and shared manifest
- `splits/`: fixed patient-level train/validation/test lists
- `features/`: synthetic MRI and PET feature vectors

## Feature interface

Each feature file contains:

`patient_id`, `feature_001` ... `feature_256`, `label`, `split`

MRI and PET rows with the same `patient_id` represent the same subject.

## Replacement

When Members 1 and 2 produce real encoder features, replace the synthetic
feature source through the modality adapters. Keep the patient IDs and
patient-level split logic intact.

Do not use these synthetic features for scientific conclusions.
