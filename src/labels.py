"""Copenhagen waste classes and source-label mapping (issue #7).

Everything is loaded from configs/categories.yaml, the single source of truth.
"""
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "categories.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = load_config()

_classes = sorted(_CFG["classes"], key=lambda c: c["id"])
if [c["id"] for c in _classes] != list(range(len(_classes))):
    raise ValueError("Class ids in categories.yaml must be contiguous and start at 0")

# Ordered list of class keys; the index is the model's output id.
CLASSES = [c["key"] for c in _classes]
class_to_idx = {name: i for i, name in enumerate(CLASSES)}
idx_to_class = {i: name for name, i in class_to_idx.items()}
DATASETS = _CFG["datasets"]


def map_source_label(dataset: str, source_label: str) -> Optional[str]:
    """Map a source dataset label to a Copenhagen class key.

    Returns None if the label is explicitly dropped.
    Raises KeyError if the dataset or label is unknown (never silently skip).
    """
    ds = DATASETS[dataset]
    if source_label in ds["map"]:
        return ds["map"][source_label]
    if source_label in ds.get("drop", []):
        return None
    raise KeyError(f"Label {source_label!r} not in mapping or drop list of {dataset!r}")
