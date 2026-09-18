"""Training and feature extraction pipeline for the MRI-only model.

This module is intentionally scoped to the MRI single-modality pipeline only and does not
implement PET or fusion. It creates a subject-level split, trains a multi-view 2D CNN on
MRI slices, and produces both MRI predictions and a compact MRI feature embedding for later
fusion with PET features.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.mri_dataset import MRISubjectDataset, collate_subject_batch
from src.data.splits import create_subject_splits
from src.models.mri.mri_classifier import MultiViewMRIClassifier
from src.preprocessing.mri_preprocessing import get_mri_transforms

CLASS_NAMES = ["CN", "MCI", "AD"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def ensure_directories(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def class_weight_from_labels(labels: Iterable[str]) -> torch.Tensor:
    counts = pd.Series(list(labels)).value_counts()
    class_weights = []
    for label in CLASS_NAMES:
        count = counts.get(label, 0)
        class_weights.append(1.0 / max(count, 1))
    weights = torch.tensor(class_weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def subject_records_from_split(split_file: str) -> List[Dict[str, str]]:
    df = pd.read_csv(split_file)
    records = []
    for row in df.to_dict("records"):
        records.append({
            "subject_id": str(row["subject_id"]),
            "group": str(row["group"]),
            "split": str(row["split"]),
            "label": LABEL_TO_INDEX.get(str(row["group"]), 0),
        })
    return records


def make_loader(records: List[Dict[str, str]], images_dir: str, cfg: Dict, is_training: bool) -> DataLoader:
    image_size = tuple(cfg["data"]["image_size"])
    transforms_fn = get_mri_transforms(
        is_training=is_training,
        image_size=image_size,
        in_channels=cfg["data"].get("in_channels", 1),
        mean=float(cfg["preprocessing"]["normalize"]["mean"][0]),
        std=float(cfg["preprocessing"]["normalize"]["std"][0]),
        rotation_degrees=float(cfg["preprocessing"]["augmentation"].get("random_rotation_degrees", 10)),
        allow_horizontal_flip=bool(cfg["preprocessing"]["augmentation"].get("random_horizontal_flip", True)),
    )
    dataset = MRISubjectDataset(
        subject_records=records,
        images_dir=images_dir,
        views=cfg["data"].get("views", ["axial", "coronal", "sagittal"]),
        image_size=image_size,
        in_channels=cfg["data"].get("in_channels", 1),
        slices_per_view=cfg["data"].get("slices_per_view", 3),
        transform=transforms_fn,
    )

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=is_training,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_subject_batch,
        pin_memory=(torch.cuda.is_available()),
    )


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, List[str], List[int], List[float], List[np.ndarray]]:
    model.eval()
    all_true = []
    all_pred = []
    all_subjects = []
    all_emb = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            sample = batch["sample"].to(device)
            view_valid = batch["view_valid"].to(device)
            logits, embeddings = model(sample, view_valid)
            probs = torch.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)

            all_true.extend(batch["label"].cpu().tolist())
            all_pred.extend(pred_labels.cpu().tolist())
            all_subjects.extend(batch["subject_id"])
            all_probs.extend(probs.max(dim=1).values.cpu().tolist())
            all_emb.extend(embeddings.cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    f1macro = f1_score(all_true, all_pred, average="macro")
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    return acc, f1macro, bal_acc, all_subjects, all_pred, all_probs, all_emb


def save_predictions(predictions: List[Dict[str, object]], output_path: str) -> None:
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)


def save_features(features: List[Dict[str, object]], output_path: str) -> None:
    df = pd.DataFrame(features)
    df.to_csv(output_path, index=False)


def train_mri_pipeline(config_path: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    set_seed(cfg["project"].get("random_seed", 42))

    metadata_path = os.environ.get("MRI_METADATA", cfg["data"]["metadata_path"])
    images_dir = os.environ.get("MRI_DATA_DIR", cfg["data"]["images_dir"])
    metadata_path = os.path.expanduser(metadata_path)
    images_dir = os.path.expanduser(images_dir)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"MRI metadata file not found: '{metadata_path}'. Set MRI_METADATA to the real CSV path. "
            "Example: /Users/anujsingh/Downloads/New folder/ADNI-1_Baseline_MRI_3_31_2026.csv"
        )
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"MRI image directory not found: '{images_dir}'. Set MRI_DATA_DIR to the real slices_multiview path. "
            "Example: /Users/anujsingh/Downloads/New folder/slices_multiview"
        )

    output_dir = cfg["outputs"].get("output_dir", "outputs")
    checkpoints_dir = cfg["outputs"].get("checkpoints_dir", os.path.join(output_dir, "checkpoints"))
    ensure_directories([output_dir, checkpoints_dir])

    train_data, val_data, test_data = create_subject_splits(
        metadata_path=metadata_path,
        images_dir=images_dir,
        views=cfg["data"].get("views", ["axial", "coronal", "sagittal"]),
        train_ratio=cfg["split"].get("train_ratio", 0.70),
        val_ratio=cfg["split"].get("val_ratio", 0.15),
        test_ratio=cfg["split"].get("test_ratio", 0.15),
        random_seed=cfg["project"].get("random_seed", 42),
        allow_missing_views=cfg["data"].get("allow_missing_views", True),
        output_dir=cfg["split"].get("splits_dir", "data/splits"),
        summary_file=cfg["split"].get("split_summary_file", os.path.join(output_dir, "split.csv")),
    )

    train_records = subject_records_from_split(os.path.join(cfg["split"].get("splits_dir", "data/splits"), "train.csv"))
    val_records = subject_records_from_split(os.path.join(cfg["split"].get("splits_dir", "data/splits"), "validation.csv"))
    test_records = subject_records_from_split(os.path.join(cfg["split"].get("splits_dir", "data/splits"), "test.csv"))

    train_loader = make_loader(train_records, images_dir, cfg, is_training=True)
    val_loader = make_loader(val_records, images_dir, cfg, is_training=False)
    test_loader = make_loader(test_records, images_dir, cfg, is_training=False)

    class_weights = class_weight_from_labels([record["group"] for record in train_records]).to(device := get_device(cfg["device"].get("preferred", "auto")))
    model = MultiViewMRIClassifier(
        in_channels=cfg["data"].get("in_channels", 1),
        feature_dim=cfg["model"].get("feature_dim", 256),
        num_classes=cfg["model"].get("num_classes", 3),
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    criterion = CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"].get("epochs", 30),
        eta_min=cfg["training"].get("min_lr", 1e-6),
    )

    best_val_f1 = -1.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(cfg["training"].get("epochs", 30)):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['training'].get('epochs', 30)} [train]", leave=False):
            sample = batch["sample"].to(device)
            view_valid = batch["view_valid"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(sample, view_valid)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        val_acc, val_f1, val_bal_acc, _, _, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={np.mean(epoch_losses):.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, val_bal_acc={val_bal_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["training"].get("early_stopping_patience", 8):
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        checkpoint_path = os.path.join(checkpoints_dir, cfg["outputs"].get("best_model_name", "best_mri_model.pth"))
        torch.save(best_model_state, checkpoint_path)
        model.load_state_dict(best_model_state)

    test_acc, test_f1, test_bal_acc, test_subjects, test_preds, test_probs, test_emb = evaluate_model(model, test_loader, device)

    all_test_df = pd.DataFrame({
        "subject_id": test_subjects,
        "true_label": [INDEX_TO_LABEL[int(p)] for p in test_preds],
        "predicted_label": [INDEX_TO_LABEL[int(p)] for p in test_preds],
        "confidence": test_probs,
        "split": "test",
    })
    all_test_df.to_csv(cfg["outputs"].get("predictions_file", os.path.join(output_dir, "mri_predictions.csv")), index=False)

    # Recompute embeddings explicitly for all subjects to save the MRI feature vectors.
    all_subject_rows = []
    for split_name, split_records in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        split_loader = make_loader(split_records, images_dir, cfg, is_training=False)
        feature_list = []
        subject_list = []
        group_list = []
        with torch.no_grad():
            for batch in split_loader:
                sample = batch["sample"].to(device)
                view_valid = batch["view_valid"].to(device)
                _, embedding = model(sample, view_valid)
                feature_list.append(embedding.cpu().numpy())
                subject_list.extend(batch["subject_id"])
                group_list.extend([INDEX_TO_LABEL[int(lbl)] for lbl in batch["label"].cpu().tolist()])

        if feature_list:
            feats = np.concatenate(feature_list, axis=0)
            for idx, subject_id in enumerate(subject_list):
                row = {"subject_id": subject_id, "group": group_list[idx], "split": split_name}
                for j in range(feats.shape[1]):
                    row[f"feature_{j}"] = float(feats[idx, j])
                all_subject_rows.append(row)

    features_path = cfg["outputs"].get("features_file", os.path.join(output_dir, "mri_features.csv"))
    pd.DataFrame(all_subject_rows).to_csv(features_path, index=False)

    metrics = {
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_balanced_accuracy": float(test_bal_acc),
        "best_validation_f1": float(best_val_f1),
        "model_checkpoint": os.path.join(checkpoints_dir, cfg["outputs"].get("best_model_name", "best_mri_model.pth")),
        "class_names": CLASS_NAMES,
    }
    with open(cfg["outputs"].get("metrics_file", os.path.join(output_dir, "mri_metrics.json")), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "metrics": metrics,
        "predictions_file": cfg["outputs"].get("predictions_file", os.path.join(output_dir, "mri_predictions.csv")),
        "features_file": features_path,
        "checkpoint": os.path.join(checkpoints_dir, cfg["outputs"].get("best_model_name", "best_mri_model.pth")),
    }


def generate_mri_features_for_split(model: torch.nn.Module, loader: DataLoader, split_name: str, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows = []
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            sample = batch["sample"].to(device)
            view_valid = batch["view_valid"].to(device)
            logits, embedding = model(sample, view_valid)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1)
            pred_labels = [INDEX_TO_LABEL[int(i)] for i in pred_idx.cpu().tolist()]

            for idx, subject_id in enumerate(batch["subject_id"]):
                true_label = INDEX_TO_LABEL[int(batch["label"][idx].item())]
                row = {"subject_id": subject_id, "group": true_label, "split": split_name}
                for j in range(embedding.shape[1]):
                    row[f"feature_{j}"] = float(embedding[idx, j].cpu().item())
                rows.append(row)

                prediction_rows.append({
                    "subject_id": subject_id,
                    "true_label": true_label,
                    "predicted_label": pred_labels[idx],
                    "confidence": float(probs[idx].max().cpu().item()),
                    "split": split_name,
                })

    feature_df = pd.DataFrame(rows)
    prediction_df = pd.DataFrame(prediction_rows)
    return feature_df, prediction_df


def generate_mri_features(config_path: str, checkpoint_path: str | None = None) -> Dict[str, str]:
    cfg = load_config(config_path)
    set_seed(cfg["project"].get("random_seed", 42))
    device = get_device(cfg["device"].get("preferred", "auto"))

    metadata_path = os.environ.get("MRI_METADATA", cfg["data"]["metadata_path"])
    images_dir = os.environ.get("MRI_DATA_DIR", cfg["data"]["images_dir"])
    metadata_path = os.path.expanduser(metadata_path)
    images_dir = os.path.expanduser(images_dir)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"MRI metadata file not found: '{metadata_path}'. Set MRI_METADATA to the real CSV path. "
            "Example: /Users/anujsingh/Downloads/New folder/ADNI-1_Baseline_MRI_3_31_2026.csv"
        )
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"MRI image directory not found: '{images_dir}'. Set MRI_DATA_DIR to the real slices_multiview path. "
            "Example: /Users/anujsingh/Downloads/New folder/slices_multiview"
        )

    output_dir = cfg["outputs"].get("output_dir", "outputs")
    ensure_directories([output_dir])

    split_file = cfg["split"].get("split_summary_file", os.path.join(output_dir, "split.csv"))
    if not os.path.exists(split_file):
        create_subject_splits(
            metadata_path=metadata_path,
            images_dir=images_dir,
            views=cfg["data"].get("views", ["axial", "coronal", "sagittal"]),
            train_ratio=cfg["split"].get("train_ratio", 0.70),
            val_ratio=cfg["split"].get("val_ratio", 0.15),
            test_ratio=cfg["split"].get("test_ratio", 0.15),
            random_seed=cfg["project"].get("random_seed", 42),
            allow_missing_views=cfg["data"].get("allow_missing_views", True),
            output_dir=cfg["split"].get("splits_dir", "data/splits"),
            summary_file=split_file,
        )

    summary_df = pd.read_csv(split_file)
    subject_rows = summary_df.to_dict("records")
    model = MultiViewMRIClassifier(
        in_channels=cfg["data"].get("in_channels", 1),
        feature_dim=cfg["model"].get("feature_dim", 256),
        num_classes=cfg["model"].get("num_classes", 3),
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    checkpoint = checkpoint_path or os.path.join(cfg["outputs"].get("checkpoints_dir", os.path.join(output_dir, "checkpoints")), cfg["outputs"].get("best_model_name", "best_mri_model.pth"))
    checkpoint = os.path.expanduser(checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"MRI checkpoint not found: '{checkpoint}'. Train the model first with:\n"
            "python scripts/train_mri.py --config configs/mri_config.yaml"
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    all_features = []
    all_predictions = []
    for split_name in ["train", "validation", "test"]:
        split_records = [
            {"subject_id": str(row["subject_id"]), "group": str(row["group"]), "split": str(row["split"]), "label": LABEL_TO_INDEX.get(str(row["group"]), 0)}
            for row in subject_rows if str(row["split"]) == split_name
        ]
        if not split_records:
            continue
        loader = make_loader(split_records, images_dir, cfg, is_training=False)
        feature_df, prediction_df = generate_mri_features_for_split(model, loader, split_name, device)
        all_features.append(feature_df)
        all_predictions.append(prediction_df)

    feature_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    prediction_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    features_path = cfg["outputs"].get("features_file", os.path.join(output_dir, "mri_features.csv"))
    predictions_path = cfg["outputs"].get("predictions_file", os.path.join(output_dir, "mri_predictions.csv"))
    feature_df.to_csv(features_path, index=False)
    prediction_df.to_csv(predictions_path, index=False)

    return {"features_file": features_path, "predictions_file": predictions_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MRI-only AD detection model.")
    parser.add_argument("--config", type=str, default="configs/mri_config.yaml", help="Path to MRI config YAML")
    args = parser.parse_args()
    train_mri_pipeline(args.config)


if __name__ == "__main__":
    main()
