"""Basic checks that the project skeleton and environment are set up."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = ["src", "configs", "scripts", "notebooks", "tests", "data", "outputs"]
REQUIRED_PACKAGES = ["torch", "torchvision", "sklearn", "umap", "matplotlib", "pandas", "yaml"]


@pytest.mark.parametrize("folder", REQUIRED_DIRS)
def test_folder_exists(folder):
    assert (ROOT / folder).is_dir()


def test_data_and_outputs_are_gitignored():
    gitignore = (ROOT / ".gitignore").read_text()
    assert "data/*" in gitignore
    assert "outputs/*" in gitignore
    assert "*.pt" in gitignore


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_package_imports(package):
    importlib.import_module(package)
