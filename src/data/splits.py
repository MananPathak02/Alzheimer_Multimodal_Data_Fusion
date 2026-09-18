"""
Subject-Level Stratified Dataset Splitting Module.

Performs strict subject-level train / validation / test splits to prevent
data leakage across MRI slices belonging to the same subject.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import yaml
from sklearn.model_selection import train_test_split


def extract_subject_id_from_filename(filename: str) -> Optional[str]:
    """Extract standard ADNI Subject ID (e.g., '020_S_1288') from a slice filename."""
    match = re.match(r"^(\d{3}_S_\d{4})", filename)
    return match.group(1) if match else None


def load_metadata(metadata_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load ADNI CSV metadata and extract unique subject demographics and group diagnosis.
    Returns:
        dict mapping subject_id -> {Group, Sex, Age, Visit, etc.}
    """
    subjects = {}
    with open(metadata_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row["Subject"].strip()
            grp = row["Group"].strip()
            if subj not in subjects:
                subjects[subj] = {
                    "Subject": subj,
                    "Group": grp,
                    "Sex": row.get("Sex", "").strip(),
                    "Age": row.get("Age", "").strip(),
                    "Visit": row.get("Visit", "bl").strip(),
                }
    return subjects


def scan_image_subjects(images_dir: str, views: List[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan axial, coronal, and sagittal image subdirectories.
    Returns:
        dict mapping subject_id -> {view: [list_of_relative_filepaths]}
    """
    if views is None:
        views = ["axial", "coronal", "sagittal"]

    subject_views = defaultdict(lambda: {v: [] for v in views})
    images_dir_path = Path(images_dir)

    for view in views:
        view_dir = images_dir_path / view
        if not view_dir.exists():
            continue
        for fname in os.listdir(view_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            subj_id = extract_subject_id_from_filename(fname)
            if subj_id:
                rel_path = os.path.join(view, fname)
                subject_views[subj_id][view].append(rel_path)

    return dict(subject_views)


def create_subject_splits(
    metadata_path: str,
    images_dir: str,
    views: List[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    allow_missing_views: bool = True,
    output_dir: str = "data/splits",
    summary_file: str = "outputs/split.csv",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform subject-level stratified splitting into train, validation, and test sets.
    
    Guarantees:
        1. Strict subject-level partitioning (ZERO data leakage).
        2. Disjoint subject sets: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.
        3. Stratification preserves class balance (CN, MCI, AD).
    """
    if views is None:
        views = ["axial", "coronal", "sagittal"]

    metadata_path = os.path.expanduser(metadata_path)
    images_dir = os.path.expanduser(images_dir)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"MRI metadata file not found: '{metadata_path}'. "
            "Set MRI_METADATA to the real ADNI CSV path, e.g. "
            "'/Users/anujsingh/Downloads/New folder/ADNI-1_Baseline_MRI_3_31_2026.csv'."
        )
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"MRI image directory not found: '{images_dir}'. "
            "Set MRI_DATA_DIR to the real slices_multiview directory, e.g. "
            "'/Users/anujsingh/Downloads/New folder/slices_multiview'."
        )

    # 1. Load CSV metadata and scan image files
    metadata = load_metadata(metadata_path)
    image_subjects = scan_image_subjects(images_dir, views=views)

    # 2. Match subjects present in both images and CSV
    matched_subjects = []
    for subj_id, view_dict in image_subjects.items():
        if subj_id not in metadata:
            continue
        
        has_views = {v: len(view_dict[v]) > 0 for v in views}
        all_views_present = all(has_views.values())

        if not allow_missing_views and not all_views_present:
            continue

        matched_subjects.append({
            "subject_id": subj_id,
            "group": metadata[subj_id]["Group"],
            "sex": metadata[subj_id]["Sex"],
            "age": metadata[subj_id]["Age"],
            "num_axial": len(view_dict.get("axial", [])),
            "num_coronal": len(view_dict.get("coronal", [])),
            "num_sagittal": len(view_dict.get("sagittal", [])),
            "has_all_views": all_views_present,
        })

    print("=" * 60)
    print("SUBJECT-LEVEL DATA SPLITTING SUMMARY")
    print("=" * 60)
    print(f"Total CSV subjects:        {len(metadata)}")
    print(f"Total image subjects:      {len(image_subjects)}")
    print(f"Matched eligible subjects: {len(matched_subjects)}")

    group_counts = Counter(s["group"] for s in matched_subjects)
    for grp, count in group_counts.items():
        print(f"  Class '{grp}': {count} subjects ({count / len(matched_subjects) * 100:.1f}%)")

    # 3. Stratified split: Train vs (Val + Test)
    subj_ids = [s["subject_id"] for s in matched_subjects]
    labels = [s["group"] for s in matched_subjects]

    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        range(len(matched_subjects)),
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    # Secondary stratified split: Val vs Test
    val_rel_ratio = val_ratio / val_test_ratio
    val_labels = [labels[i] for i in val_test_idx]
    val_sub_idx, test_sub_idx = train_test_split(
        range(len(val_test_idx)),
        test_size=(1.0 - val_rel_ratio),
        random_state=random_seed,
        stratify=val_labels,
    )

    val_idx = [val_test_idx[i] for i in val_sub_idx]
    test_idx = [val_test_idx[i] for i in test_sub_idx]

    train_data = []
    for i in train_idx:
        rec = dict(matched_subjects[i])
        rec["split"] = "train"
        train_data.append(rec)

    val_data = []
    for i in val_idx:
        rec = dict(matched_subjects[i])
        rec["split"] = "validation"
        val_data.append(rec)

    test_data = []
    for i in test_idx:
        rec = dict(matched_subjects[i])
        rec["split"] = "test"
        test_data.append(rec)

    # 4. Strict Leakage Verification
    train_set = {s["subject_id"] for s in train_data}
    val_set = {s["subject_id"] for s in val_data}
    test_set = {s["subject_id"] for s in test_data}

    assert train_set.isdisjoint(val_set), "LEAKAGE DETECTED: Train and Val overlap!"
    assert train_set.isdisjoint(test_set), "LEAKAGE DETECTED: Train and Test overlap!"
    assert val_set.isdisjoint(test_set), "LEAKAGE DETECTED: Val and Test overlap!"

    print("-" * 60)
    print(f"Train subjects:      {len(train_data)} ({len(train_data)/len(matched_subjects)*100:.1f}%) -> {dict(Counter(s['group'] for s in train_data))}")
    print(f"Validation subjects: {len(val_data)} ({len(val_data)/len(matched_subjects)*100:.1f}%) -> {dict(Counter(s['group'] for s in val_data))}")
    print(f"Test subjects:       {len(test_data)} ({len(test_data)/len(matched_subjects)*100:.1f}%) -> {dict(Counter(s['group'] for s in test_data))}")
    print(f"Subject overlap check passed: 100% disjoint, 0 data leakage.")
    print("=" * 60)

    # 5. Export split CSV files
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(summary_file)
    os.makedirs(summary_path.parent, exist_ok=True)

    fieldnames = ["subject_id", "group", "split", "num_axial", "num_coronal", "num_sagittal", "has_all_views", "sex", "age"]

    for split_name, split_list in [("train", train_data), ("validation", val_data), ("test", test_data)]:
        split_filepath = Path(output_dir) / f"{split_name}.csv"
        with open(split_filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_list)

    # Export combined summary split file
    all_splits = train_data + val_data + test_data
    with open(summary_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_splits)

    print(f"Saved split files to: {output_dir}/ and {summary_file}")
    return train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subject-level stratified data splits")
    parser.add_argument("--config", type=str, default="configs/mri_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow environment variable overrides
    meta_path = os.environ.get("MRI_METADATA", cfg["data"]["metadata_path"])
    img_dir = os.environ.get("MRI_DATA_DIR", cfg["data"]["images_dir"])

    create_subject_splits(
        metadata_path=meta_path,
        images_dir=img_dir,
        views=cfg["data"].get("views", ["axial", "coronal", "sagittal"]),
        train_ratio=cfg["split"].get("train_ratio", 0.70),
        val_ratio=cfg["split"].get("val_ratio", 0.15),
        test_ratio=cfg["split"].get("test_ratio", 0.15),
        random_seed=cfg["project"].get("random_seed", 42),
        allow_missing_views=cfg["data"].get("allow_missing_views", True),
        output_dir=cfg["split"].get("splits_dir", "data/splits"),
        summary_file=cfg["split"].get("split_summary_file", "outputs/split.csv"),
    )
