"""Wrapper script for training the MRI-only model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_mri import main


if __name__ == "__main__":
    main()
