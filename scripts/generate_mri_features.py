"""Wrapper script for generating MRI features and predictions from a saved checkpoint."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_mri import generate_mri_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MRI features from a trained MRI checkpoint.")
    parser.add_argument("--config", type=str, default="configs/mri_config.yaml", help="Path to MRI config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path; defaults to the best MRI model")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint:
        checkpoint = os.path.expanduser(checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint}'. Train the model first with:\n"
                "python scripts/train_mri.py --config configs/mri_config.yaml"
            )

    print(generate_mri_features(args.config, checkpoint_path=checkpoint))


if __name__ == "__main__":
    main()
