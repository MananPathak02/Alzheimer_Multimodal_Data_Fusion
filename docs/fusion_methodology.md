    
# Multimodal Fusion Methodology

## 1. Role and Objective

This module implements the multimodal fusion component of the Alzheimer
Multimodal Data Fusion project.

The purpose of the fusion module is to combine learned representations from
two imaging modalities:

- MRI
- PET

MRI and PET processing are handled by their respective modality pipelines.
The fusion module consumes the resulting modality-level feature
representations and produces a final three-class classification output.

The target classes are:

- CN — Cognitively Normal
- MCI — Mild Cognitive Impairment
- AD — Alzheimer's Disease

---

## 2. Current Development Status

The current fusion implementation has been developed and tested using
synthetic dummy feature data.

The dummy dataset contains:

- 156 subjects
- 256 MRI features per subject
- 256 PET features per subject
- 3 classification classes
- Patient-level train, validation, and test splits

The dummy features are used only to verify the software architecture,
data alignment, training process, checkpoint generation, and prediction
pipeline.

The resulting dummy-data accuracy values must not be interpreted as
clinical, medical, or scientific Alzheimer disease results.

---

## 3. Feature Contract

The fusion module expects modality features to be aligned using a common
patient identifier.

The required logical structure is:

```text
patient_id
MRI features
PET features
label
split
````

The fusion pipeline verifies that:

1. MRI and PET patient identifiers are unique.
2. MRI and PET contain the same patient set.
3. Labels agree between modalities.
4. Dataset splits agree between modalities.

This alignment is necessary because the MRI and PET representations must
belong to the same patient before they can be fused.

---

## 4. Feature Alignment

The file:

```text
src/features/fused_features.py
```

performs multimodal feature alignment.

It loads:

```text
data/dummy/features/mri_features.csv
data/dummy/features/pet_features.csv
```

and produces:

```text
data/dummy/features/aligned_multimodal_features.csv
```

The resulting aligned dataset contains:

* patient identifier
* label
* split
* MRI feature columns
* PET feature columns

MRI and PET feature names are prefixed so that the two modalities remain
unambiguous.

---

## 5. Data Scaling

Feature scaling is performed separately for MRI and PET.

A StandardScaler is fitted only using the training split.

The fitted MRI scaler is then applied to:

* MRI training data
* MRI validation data
* MRI test data

The fitted PET scaler is independently applied to:

* PET training data
* PET validation data
* PET test data

The validation and test sets are never used to fit the scalers.

This prevents information from the evaluation data from influencing
training-time preprocessing.

---

## 6. Patient-Level Data Splitting

The project uses patient-level dataset splitting.

A patient must belong to only one of:

* training set
* validation set
* test set

Different slices or representations belonging to the same patient must
not be distributed across different splits.

This is important for medical imaging experiments because slice-level
splitting can allow information from the same patient to appear in both
training and evaluation data.

---

# 7. Early Fusion

## 7.1 Concept

Early fusion combines MRI and PET feature representations before the
main classification layers.

The basic operation is:

```text
MRI features
      |
      +------ concatenate ------> Fusion Network ---> Classification
      |
PET features
```

For the current development configuration:

```text
MRI: 256 features
PET: 256 features
-----------------
Combined: 512 features
```

The implemented network contains:

```text
512
 ↓
Linear
 ↓
ReLU
 ↓
Dropout
 ↓
256
 ↓
Linear
 ↓
ReLU
 ↓
Dropout
 ↓
128
 ↓
Linear
 ↓
3 classes
```

The implementation is located at:

```text
src/models/fusion/early_fusion.py
```

Training is performed by:

```text
src/training/train_fusion.py
```

---

# 8. Intermediate Fusion

## 8.1 Concept

Intermediate fusion first transforms the MRI and PET feature vectors into
learned modality-specific representations.

The representations are then combined by a fusion network.

The current architecture is:

```text
MRI features
    |
Linear
    |
ReLU
    |
Dropout
    |
128-dimensional MRI representation
             \
              \
               Concatenate
              /
             /
128-dimensional PET representation
    |
Linear
    |
ReLU
    |
Dropout
    |
128
    |
Linear
    |
ReLU
    |
Dropout
    |
64
    |
Linear
    |
3 classes
```

The implementation is located at:

```text
src/models/fusion/intermediate_fusion.py
```

Training is performed by:

```text
src/training/train_intermediate_fusion.py
```

---

# 9. Late Fusion

## 9.1 Concept

Late fusion keeps the modality classifiers separate until their class
probabilities have been generated.

The architecture is:

```text
MRI features ---> MRI classifier ---> MRI probabilities
                                           |
                                           |
                                           +---- Weighted combination
                                           |
                                           |
PET features ---> PET classifier ---> PET probabilities
                                           |
                                           ↓
                                  Final fused probabilities
                                           |
                                           ↓
                                      Prediction
```

Each modality has an independent classifier.

The current modality classifier is:

```text
256
 ↓
128
 ↓
64
 ↓
3 classes
```

ReLU activations and dropout are used between the linear layers.

The final fusion uses weighted probability averaging.

The default weights are:

```text
MRI weight = 0.5
PET weight = 0.5
```

The fused probability is conceptually:

```text
P_fused =
    MRI_weight × P_MRI
    +
    PET_weight × P_PET
```

The implementation is located at:

```text
src/models/fusion/late_fusion.py
```

Training is performed by:

```text
src/training/train_late_fusion.py
```

---

# 10. Training Configuration

The current development configuration uses:

```text
Batch size:       16
Learning rate:    0.001
Epochs:           30
Random seed:      42
Optimizer:        Adam
```

The classification problem contains three classes:

```text
CN  -> 0
MCI -> 1
AD  -> 2
```

The model with the best validation accuracy is saved as the selected
checkpoint for each fusion strategy.

---

# 11. Fusion Outputs

Each fusion strategy produces three primary outputs.

## Model checkpoint

```text
checkpoints/fusion/
```

Examples:

```text
early_fusion_best.pt
intermediate_fusion_best.pt
late_fusion_best.pt
```

## Training history

```text
results/fusion/
```

Examples:

```text
early_fusion_training_history.csv
intermediate_fusion_training_history.csv
late_fusion_training_history.csv
```

## Test predictions

Examples:

```text
results/fusion/early_fusion/predictions.csv
results/fusion/intermediate_fusion/predictions.csv
results/fusion/late_fusion/predictions.csv
```

The prediction files contain the patient identifier, true label, predicted
label, class probabilities, and correctness information.

Late fusion additionally stores the MRI-only and PET-only branch
probabilities for analysis.

---

# 12. Evaluation Interface

The fusion module provides predictions that can be consumed by the common
evaluation framework.

Member 4 can use the prediction files to calculate and visualize:

* Accuracy
* Precision
* Recall
* F1-score
* Sensitivity
* Specificity
* ROC-AUC
* Confusion matrix
* ROC curves
* Class-wise performance

The same patient-level test split should be used when comparing MRI-only,
PET-only, and multimodal fusion models.

---

# 13. Leakage Prevention

The following rules must be maintained throughout the project:

1. Patient identifiers must be unique.
2. MRI and PET must be aligned by patient.
3. Patient-level splitting must be performed before model evaluation.
4. The test set must remain isolated until final evaluation.
5. Feature scalers must be fitted only on training data.
6. Validation data must not be used to tune the final test result.
7. MRI and PET data from the same patient must remain in the same split.
8. Dummy data must never be mixed with real medical imaging data.

---

# 14. Integration With Other Team Members

## Member 1 — MRI

Member 1 provides the MRI representation.

Expected logical interface:

```text
patient_id
MRI feature vector
label
split
```

## Member 2 — PET

Member 2 provides the PET representation.

Expected logical interface:

```text
patient_id
PET feature vector
label
split
```

## Member 3 — Multimodal Fusion

This module:

```text
MRI features
      +
PET features
      ↓
Patient alignment
      ↓
Feature scaling
      ↓
Fusion strategy
      ↓
Final prediction
```

## Member 4 — Evaluation

Member 4 consumes the generated predictions and performs common evaluation
and visualization.

---

# 15. Current Development Results

The fusion implementations have been successfully executed on the
synthetic dummy dataset.

Development pipeline results:

```text
Early Fusion:
Validation accuracy: 0.9565
Test accuracy:       0.9583

Intermediate Fusion:
Validation accuracy: 1.0000
Test accuracy:       1.0000

Late Fusion:
Validation accuracy: 1.0000
Test accuracy:       1.0000
```

These values demonstrate that the implementation and data flow operate
successfully on the synthetic development data.

They are NOT valid Alzheimer disease model performance measurements.

Real scientific evaluation will only be performed after the real MRI and
PET modality pipelines produce the required patient-aligned features.

---

# 16. Future Integration

The next integration stage is:

```text
Real MRI preprocessing
        ↓
MRI encoder
        ↓
MRI features
        ↓
                         Patient alignment
                              ↓
Real PET preprocessing → PET encoder
                              ↓
                         PET features
                              ↓
                       Fusion pipeline
                              ↓
                    Final classification
                              ↓
                         Evaluation
```

The fusion implementation should remain independent of the internal
architecture of the MRI and PET encoders.

Only the feature interface needs to remain consistent.

---

# 17. Member 3 Completion Criteria

The Member 3 fusion implementation is considered complete when:

* [x] MRI/PET feature alignment is implemented.
* [x] Early fusion model is implemented.
* [x] Intermediate fusion model is implemented.
* [x] Late fusion model is implemented.
* [x] Training pipelines are implemented.
* [x] Validation and test prediction generation is implemented.
* [x] Model checkpoints are generated.
* [x] Training histories are generated.
* [x] Configuration file is defined.
* [x] Fusion methodology is documented.
* [ ] Common project evaluation is completed by Member 4.
* [ ] Real MRI + PET features are integrated.
* [ ] Final scientific results are generated.

The remaining unchecked items belong to later project integration rather
than the core fusion implementation.

