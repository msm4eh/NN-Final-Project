# NN-Final-Project: Trash Classification

An image classifier that sorts photos of household waste into Copenhagen's waste sorting categories, using transfer learning on a pretrained CNN (PyTorch). We also look at what the model has learned with Grad-CAM heatmaps and embedding plots (PCA, t-SNE, UMAP).

Dataset: https://drive.google.com/file/d/1CTpIK4eI_UvPEsTt35dtYKXQNeJCk3TJ/view?usp=sharing

## Setup

Requires **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pip install torch` installs the default build for your machine. To train on a GPU, follow the install command on pytorch.org for your CUDA version.

## Run the tests

```bash
pytest
```

## Data

Download the dataset from the link above and unzip it into `data/`. The `data/` folder is git-ignored, so images are never committed. Scripts for preparing the train/val/test split are added in issue #8.

## Project layout

See [AGENTS.md](AGENTS.md) for the folder layout and the rules for coding agents.

## Workflow

We follow the issue → PR → review workflow described in `# AI-Assisted Project Workflow.md`.
