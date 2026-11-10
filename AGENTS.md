# AGENTS.md

Instructions for coding agents working in this repo. Read this before every task.

## What this project does

A neural network that classifies photos of household waste into Copenhagen's waste sorting categories. We fine-tune an ImageNet-pretrained ResNet-18 (PyTorch), evaluate it, and explain its predictions with Grad-CAM and embedding plots (PCA, t-SNE, UMAP).

## Folder layout

| Folder | Contents |
| --- | --- |
| `src/` | Importable project code (datasets, models, training, evaluation, explainability) |
| `configs/` | YAML config files: categories, hyperparameters. No magic numbers in code |
| `scripts/` | Command-line entry points (data preparation, plotting) |
| `notebooks/` | Exploration only. Code that we keep moves into `src/` |
| `tests/` | pytest tests |
| `data/` | Raw images and split CSVs. **Git-ignored** |
| `outputs/` | Checkpoints, metrics, plots. **Git-ignored** |

## Commands

```bash
# install (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# test
pytest
```

## Coding style and libraries

- Python 3.11. Follow PEP 8, use type hints on function signatures.
- Allowed libraries: PyTorch, torchvision, scikit-learn, umap-learn, matplotlib, pandas, numpy, Pillow, PyYAML, pytest.
- Do not add a new dependency without asking. If one is approved, add it to `requirements.txt` with a pinned version.
- Read settings from `configs/` instead of hard-coding them.
- Keep functions small and testable. Put a test in `tests/` for new logic.

## Rules

1. **Never edit, delete or commit anything in `data/`.**
2. **Never commit model checkpoints (`*.pt`, `*.pth`) or API keys / passwords.**
3. **Only touch files the current issue asks for.** Do not refactor, reformat or "improve" other files.
4. **Set and log random seeds** (Python, NumPy, PyTorch) in anything that trains, splits or samples.
5. **Every new function in `src/` gets a docstring** saying what it does, its arguments and what it returns.
6. Never use the test split for training or tuning. Only touch it when `--split test` is passed explicitly.
7. Plan first. Before writing code, list the files you will change and wait for approval.
8. End AI-assisted commit messages with `Co-authored-by: <model name> <email>`.

When an agent repeats a mistake, add a rule here (through a PR).
