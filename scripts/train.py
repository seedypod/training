"""Train the pod-state model: `uv run --locked python scripts/train.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedypod_training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
