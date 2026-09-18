"""Subject-level MRI dataset for the multi-view ADNI MRI pipeline.

The MRI dataset is built around a strict subject-level abstraction so that all slices
for a given subject remain in one split. We avoid treating each 2D slice as an independent
patient and instead aggregate a small set of representative slices per view.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.splits import extract_subject_id_from_filename, scan_image_subjects

VIEW_NAMES = ("axial", "coronal", "sagittal")
CLASS_TO_INDEX = {"CN": 0, "MCI": 1, "AD": 2}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}


def resolve_dataset_path(raw_path: str) -> str:
    """Resolve a project path while honoring environment variable overrides."""
    if raw_path is None:
        return raw_path
    if raw_path.startswith("$"):
        return os.environ.get(raw_path[1:], raw_path)
    return os.path.expanduser(raw_path)


def extract_slice_index(filename: str) -> int:
    """Extract the slice index from an MRI filename like ..._ax_105.png."""
    match = re.search(r"_(\d+)\.(png|jpg|jpeg)$", filename, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def select_representative_slices(
    file_paths: Sequence[str],
    slices_per_view: int = 3,
    sampling_strategy: str = "equidistant",
) -> List[str]:
    """Select a small set of representative slices per subject/view.

    This handles variable slice counts across subjects without creating patient leakage.
    We use evenly spaced slice indices so that the selected set covers the subject's
    available MRI range while staying computationally lightweight.
    """
    if not file_paths:
        return []

    ordered = sorted(file_paths, key=lambda p: extract_slice_index(os.path.basename(p)))
    if len(ordered) <= slices_per_view:
        return ordered

    if sampling_strategy == "middle":
        center = len(ordered) // 2
        start = max(0, center - slices_per_view // 2)
        end = min(len(ordered), start + slices_per_view)
        if end - start < slices_per_view:
            start = max(0, len(ordered) - slices_per_view)
            end = len(ordered)
        return ordered[start:end]

    # 'equidistant' default: choose roughly evenly spaced indices across the subject's slice stack
    indices = []
    for i in range(slices_per_view):
        if slices_per_view == 1:
            indices.append(0)
        else:
            idx = round(i * (len(ordered) - 1) / (slices_per_view - 1))
            indices.append(idx)
    unique_indices = []
    for idx in indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
    return [ordered[idx] for idx in unique_indices[:slices_per_view]]


def build_subject_records(
    metadata_path: str,
    images_dir: str,
    views: Sequence[str] = VIEW_NAMES,
    allow_missing_views: bool = True,
    slices_per_view: int = 3,
    sampling_strategy: str = "equidistant",
) -> List[Dict[str, object]]:
    """Create subject records by matching CSV metadata with image filenames.

    Each record contains the diagnosis label and the available image paths per view.
    The split will later be assigned at the subject level.
    """
    metadata_path = resolve_dataset_path(metadata_path)
    images_dir = resolve_dataset_path(images_dir)

    csv_df = pd.read_csv(metadata_path)
    csv_df["Subject"] = csv_df["Subject"].astype(str).str.strip()
    subject_df = csv_df.drop_duplicates(subset=["Subject"]).copy()
    subject_meta = subject_df.set_index("Subject")["Group"].to_dict()

    image_index = scan_image_subjects(images_dir, views=list(views))
    records: List[Dict[str, object]] = []

    for subject_id, view_map in image_index.items():
        if subject_id not in subject_meta:
            continue
        group = subject_meta[subject_id].strip()
        if group not in CLASS_TO_INDEX:
            continue
        if not allow_missing_views and not all(len(view_map.get(v, [])) > 0 for v in views):
            continue

        selected_paths = {
            view: select_representative_slices(view_map.get(view, []), slices_per_view, sampling_strategy)
            for view in views
        }

        records.append(
            {
                "subject_id": subject_id,
                "group": group,
                "label": CLASS_TO_INDEX[group],
                "image_paths": selected_paths,
                "view_presence": {view: len(selected_paths.get(view, [])) > 0 for view in views},
            }
        )

    return records


class MRISubjectDataset(Dataset):
    """Dataset yielding one subject sample at a time.

    The sample is a 3-view tensor with a fixed number of slices per view. Missing views are
    zero-filled and marked via a validity mask, which makes the model robust to incomplete data.
    """

    def __init__(
        self,
        subject_records: Sequence[Dict[str, object]],
        images_dir: str,
        views: Sequence[str] = VIEW_NAMES,
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 1,
        slices_per_view: int = 3,
        transform: Optional[transforms.Compose] = None,
    ):
        self.subject_records = list(subject_records)
        self.images_dir = resolve_dataset_path(images_dir)
        self.views = list(views)
        self.image_size = image_size
        self.in_channels = in_channels
        self.slices_per_view = slices_per_view
        self.transform = transform
        self.image_index = scan_image_subjects(self.images_dir, views=self.views)

    def _read_img_tensor(self, path: str) -> torch.Tensor:
        full_path = os.path.join(self.images_dir, path)
        if not os.path.exists(full_path):
            return torch.zeros((self.in_channels, *self.image_size), dtype=torch.float32)
        try:
            with Image.open(full_path) as img:
                img = img.convert("L")
                if self.transform is not None:
                    tensor = self.transform(img)
                else:
                    tensor = transforms.ToTensor()(img)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.shape[0] == 1 and self.in_channels == 3:
                    tensor = tensor.repeat(3, 1, 1)
                if tensor.shape[0] != self.in_channels:
                    tensor = tensor[: self.in_channels]
                return tensor.float()
        except Exception:
            return torch.zeros((self.in_channels, *self.image_size), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.subject_records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.subject_records[idx]
        subject_id = str(record["subject_id"])
        label = int(record["label"])
        view_presence = {view: bool(record.get("view_presence", {}).get(view, False)) for view in self.views}

        sample = torch.zeros((len(self.views), self.slices_per_view, self.in_channels, *self.image_size), dtype=torch.float32)
        valid_views = torch.zeros(len(self.views), dtype=torch.float32)

        for view_idx, view_name in enumerate(self.views):
            paths = self.image_index.get(subject_id, {}).get(view_name, [])
            if not paths:
                continue
            selected = select_representative_slices(paths, slices_per_view=self.slices_per_view, sampling_strategy="equidistant")
            if not selected:
                continue
            for slice_idx, relative_path in enumerate(selected[: self.slices_per_view]):
                tensor = self._read_img_tensor(relative_path)
                sample[view_idx, slice_idx] = tensor
            valid_views[view_idx] = 1.0

        return {
            "subject_id": subject_id,
            "label": torch.tensor(label, dtype=torch.long),
            "sample": sample,
            "view_valid": valid_views,
            "group": record.get("group", ""),
            "split": record.get("split", "unknown"),
        }


def collate_subject_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a list of subject samples into a batch dictionary."""
    subject_ids = [item["subject_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    samples = torch.stack([item["sample"] for item in batch])
    valid_views = torch.stack([item["view_valid"] for item in batch])
    return {
        "subject_id": subject_ids,
        "label": labels,
        "sample": samples,
        "view_valid": valid_views,
    }
