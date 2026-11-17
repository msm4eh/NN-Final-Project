import pytest

from src.labels import CLASSES, DATASETS, class_to_idx, map_source_label


def test_class_index_is_consistent():
    assert len(set(CLASSES)) == len(CLASSES)
    assert [class_to_idx[c] for c in CLASSES] == list(range(len(CLASSES)))


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_every_source_label_is_mapped_or_dropped(dataset):
    ds = DATASETS[dataset]
    for label in ds["labels"]:
        result = map_source_label(dataset, label)  # raises KeyError if unhandled
        assert result is None or result in CLASSES


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_mapping_and_drop_do_not_overlap_or_invent_labels(dataset):
    ds = DATASETS[dataset]
    mapped, dropped, declared = set(ds["map"]), set(ds.get("drop", [])), set(ds["labels"])
    assert not (mapped & dropped)
    assert (mapped | dropped) <= declared


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_mapping_targets_are_known_classes(dataset):
    assert set(DATASETS[dataset]["map"].values()) <= set(CLASSES)


def test_every_class_has_at_least_one_source():
    used = {t for ds in DATASETS.values() for t in ds["map"].values()}
    assert used == set(CLASSES)


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        map_source_label("trashnet", "banana")
