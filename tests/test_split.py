"""Tests for the train/val/test split (issue #8). Uses fake files, no real data."""
import pandas as pd
import pytest

from scripts.prepare_data import main
from src.split import drop_duplicates, load_crops, split_crops

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
LABELS = ["plast", "metal", "glas"]


@pytest.fixture
def data_dir(tmp_path):
    """Fake data dir: 60 photos per class with 1-3 crops each, plus 20 unofficial photos."""
    rows = []
    for li, label in enumerate(LABELS):
        for photo in range(60):
            for crop in range(1 + photo % 3):
                source = "taco_unofficial" if photo < 20 and li == 0 else "taco_official"
                path = f"crops/{label}_{photo}_{crop}.jpg"
                rows.append((path, label, f"{source}:{label}_{photo}", source))
    (tmp_path / "crops").mkdir()
    for path, *_ in rows:
        (tmp_path / path).write_bytes(path.encode())  # unique content per file
    pd.DataFrame(rows, columns=["path", "label", "image_id", "source"]).to_csv(tmp_path / "crops/crops.csv", index=False)
    return tmp_path


def _split(data_dir, seed=42):
    df = load_crops(data_dir / "crops/crops.csv")
    return split_crops(df, FRACTIONS, ["taco_unofficial"], seed)


def test_no_photo_in_two_splits(data_dir):
    ids = [set(part["image_id"]) for part in _split(data_dir).values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_unofficial_only_in_train(data_dir):
    splits = _split(data_dir)
    assert (splits["train"]["source"] == "taco_unofficial").sum() > 0
    for name in ("val", "test"):
        assert not (splits[name]["source"] == "taco_unofficial").any()


def test_every_row_kept_and_fractions_roughly_right(data_dir):
    splits = _split(data_dir)
    assert sum(len(p) for p in splits.values()) == len(load_crops(data_dir / "crops/crops.csv"))
    for name in ("val", "test"):
        share = splits[name]["image_id"].nunique() / pd.concat(splits.values())["image_id"].nunique()
        assert 0.08 < share < 0.22, (name, share)
        assert set(splits[name]["label"]) == set(LABELS)  # every class appears


def test_same_seed_same_csvs(data_dir):
    main(["--seed", "7", "--data-dir", str(data_dir)])
    first = {n: (data_dir / f"splits/{n}.csv").read_text() for n in ("train", "val", "test")}
    main(["--seed", "7", "--data-dir", str(data_dir)])
    second = {n: (data_dir / f"splits/{n}.csv").read_text() for n in ("train", "val", "test")}
    assert first == second


def test_duplicates_removed_and_conflicts_dropped(tmp_path):
    files = {"a.jpg": b"same", "b.jpg": b"same", "c.jpg": b"clash", "d.jpg": b"clash", "e.jpg": b"unique"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    df = pd.DataFrame({
        "path": list(files),
        "label": ["plast", "plast", "metal", "glas", "papir"],
        "image_id": list("abcde"),
        "source": ["taco_official"] * 5,
    })
    kept, removed, conflicting = drop_duplicates(df, tmp_path)
    assert list(kept["path"]) == ["a.jpg", "e.jpg"]
    assert (removed, conflicting) == (1, 2)


def test_unknown_label_raises(tmp_path):
    pd.DataFrame([["x.jpg", "banana", "p1", "taco_official"]], columns=["path", "label", "image_id", "source"]).to_csv(
        tmp_path / "crops.csv", index=False
    )
    with pytest.raises(ValueError, match="banana"):
        load_crops(tmp_path / "crops.csv")
