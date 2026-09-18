"""Generate the subject-level MRI split and save the dataset summary."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import create_subject_splits
from src.training.train_mri import load_config


def main() -> None:
    cfg = load_config("configs/mri_config.yaml")
    metadata_path = os.environ.get("MRI_METADATA", cfg["data"]["metadata_path"])
    images_dir = os.environ.get("MRI_DATA_DIR", cfg["data"]["images_dir"])
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
        summary_file=cfg["split"].get("split_summary_file", "outputs/split.csv"),
    )


if __name__ == "__main__":
    main()
