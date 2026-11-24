"""Split the crops from issue #23 into train/val/test CSVs (issue #8).

Usage, from the repo root: python scripts/prepare_data.py --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `src` imports when run as a plain script

from src.split import count_table, drop_duplicates, load_crops, split_crops, write_splits  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Load crops.csv, de-duplicate, split by photo, write CSVs and print counts.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Overrides data.seed in the config")
    parser.add_argument("--data-dir", type=Path, default=None, help="Overrides data.data_dir in the config")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_dir = args.data_dir if args.data_dir is not None else ROOT / cfg["data_dir"]

    # AGENTS.md rule 4. The split itself is seeded through scikit-learn's random_state.
    random.seed(seed)
    np.random.seed(seed)

    df = load_crops(data_dir / cfg["crops_csv"])
    df, removed, conflicting = drop_duplicates(df, data_dir)
    splits = split_crops(df, cfg["split_fractions"], cfg["train_only_sources"], seed)
    out_dir = data_dir / cfg["splits_subdir"]
    write_splits(splits, out_dir)

    photos = ", ".join(f"{name} {part['image_id'].nunique()}" for name, part in splits.items())
    print(f"Seed: {seed}\nDuplicates removed: {removed}; dropped for conflicting labels: {conflicting}")
    print(f"Photos per split: {photos}\nWrote train/val/test.csv to {out_dir}\n")
    print(count_table(splits).to_string())


if __name__ == "__main__":
    main()
