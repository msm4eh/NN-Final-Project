"""Reproducible train/val/test split of the crops from issue #23 (issue #8).

Input columns: path (relative to the data dir), label, image_id (source photo), source.
All crops of one photo always land in the same split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.labels import CLASSES

SPLITS = ("train", "val", "test")
COLUMNS = ["path", "label", "image_id", "source"]


def load_crops(csv_path: Path) -> pd.DataFrame:
    """Read crops.csv and check its columns and labels.

    Args:
        csv_path: Path to crops.csv.

    Returns:
        DataFrame with columns ``path,label,image_id,source``, sorted by path.

    Raises:
        ValueError: If columns are missing or a label is not a known class.
    """
    df = pd.read_csv(csv_path, dtype=str)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    unknown = set(df["label"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path}: {sorted(unknown)}")
    return df[COLUMNS].sort_values("path", ignore_index=True)


def drop_duplicates(df: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, int, int]:
    """Remove rows whose files are byte-identical (same SHA-256).

    If all copies have the same label, the first by path is kept. If the copies
    disagree on the label, every copy is dropped.

    Args:
        df: Crops DataFrame (see ``load_crops``).
        data_dir: Folder that ``path`` is relative to.

    Returns:
        The de-duplicated DataFrame, the number of duplicate rows removed, and
        the number of rows dropped because duplicates had conflicting labels.
    """
    hashes = df["path"].map(lambda p: hashlib.sha256((data_dir / p).read_bytes()).hexdigest())
    keep, removed, conflicting = [], 0, 0
    for _, idx in df.groupby(hashes, sort=False).groups.items():
        rows = df.loc[idx]
        if rows["label"].nunique() > 1:
            conflicting += len(rows)
            continue
        keep.append(rows.index.min())
        removed += len(rows) - 1
    return df.loc[sorted(keep)].reset_index(drop=True), removed, conflicting


def _take_fold(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off about ``fraction`` of the photos, stratified by label and grouped by photo.

    Args:
        df: Rows to split.
        fraction: Share of rows to split off (e.g. 0.15).
        seed: Random seed for the shuffle.

    Returns:
        ``(rest, taken)`` DataFrames.
    """
    n_splits = max(2, round(1 / fraction))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, taken_idx = next(folds.split(df, df["label"], groups=df["image_id"]))
    return df.iloc[rest_idx], df.iloc[taken_idx]


def split_crops(
    df: pd.DataFrame, fractions: dict[str, float], train_only_sources: list[str], seed: int
) -> dict[str, pd.DataFrame]:
    """Split crops into train/val/test by photo, stratified by label.

    Rows from ``train_only_sources`` always go to train. The other rows are
    split so that val and test each get about their fraction of the photos.

    Args:
        df: Crops DataFrame (see ``load_crops``).
        fractions: Keys ``train``, ``val``, ``test`` summing to 1.
        train_only_sources: Sources that must never be in val or test.
        seed: Random seed, so the split is reproducible.

    Returns:
        Mapping from split name to a DataFrame sorted by path.

    Raises:
        ValueError: If the fractions are malformed.
    """
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-6:
        raise ValueError(f"split_fractions needs keys {SPLITS} summing to 1, got {fractions}")

    df = df.sort_values("path", ignore_index=True)
    train_only = df["source"].isin(train_only_sources)
    pool = df[~train_only].reset_index(drop=True)

    rest, test = _take_fold(pool, fractions["test"], seed)
    rest = rest.reset_index(drop=True)
    train, val = _take_fold(rest, fractions["val"] / (1 - fractions["test"]), seed)
    train = pd.concat([train, df[train_only]])

    return {
        name: part.sort_values("path", ignore_index=True)
        for name, part in zip(SPLITS, (train, val, test))
    }


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write each split to ``out_dir/<split>.csv``.

    Args:
        splits: Output of ``split_crops``.
        out_dir: Folder to write into; created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part[COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)


def count_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count crops per class and split, with totals.

    Args:
        splits: Output of ``split_crops``.

    Returns:
        DataFrame with one row per class plus ``total``, one column per split plus ``total``.
    """
    table = pd.DataFrame({name: part["label"].value_counts() for name, part in splits.items()})
    table = table.reindex(CLASSES).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum()
    return table
