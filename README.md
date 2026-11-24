# NN-Final-Project: Trash Classification

An image classifier that sorts photos of household waste into Copenhagen's waste sorting categories, using transfer learning on a pretrained CNN (PyTorch). We also look at what the model has learned with Grad-CAM heatmaps and embedding plots (PCA, t-SNE, UMAP).

Access data here: https://drive.google.com/file/d/1CTpIK4eI_UvPEsTt35dtYKXQNeJCk3TJ/view?usp=sharing

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

Classes (7): madaffald, restaffald, papir, pap, plast, metal, glas. Defined in `configs/categories.yaml`, which also maps each source label to a class. 

Sources:
- TACO (https://github.com/pedropro/TACO): the Google Drive link above. Official annotations are used for training, validation and test; unofficial annotations for training only.
- TU Wien "Image Dataset of Domestic Organic Waste and Non-Organic Contaminants" (Wolling and Pal, CC BY 4.0, DOI 10.48436/27k90-dvw73): food waste, also in the Google Drive folder above.

Download the dataset from the link above and unzip it into `data/`. The `data/` folder is git-ignored, so images are never committed. Scripts for preparing the train/val/test split are added in issue #8.

Known limitation: there is no hazardous-waste class. The model cannot recognize batteries, aerosols, paint or bulbs and will sort them into another bin.

## Project layout

See [AGENTS.md](AGENTS.md) for the folder layout and the rules for coding agents.

## Workflow

We follow the issue → PR → review workflow described in `# AI-Assisted Project Workflow.md`.
