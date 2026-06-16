# Isca Emulation Repository

Research code for training and evaluating machine learning emulators of Isca climate model simulations. The repository contains preprocessing pipelines, model definitions, training and evaluation entry points, experiment configuration files, and analysis notebooks used around the paper workflow.

## Repository Layout

- `src/isca_emulation_v2/`: Python package with data processing, models, training, evaluation, plotting, and W&B helpers.
- `config_data/`: preprocessing configurations.
- `config_train/`: Hydra training configurations.
- `config_inference/`: evaluation configurations.
- `tests/`: unit tests for data utilities and model components.
- `notebooks/`: exploratory and paper-analysis notebooks.

Large local files are intentionally not tracked. Raw data, processed data, checkpoints, downloaded artifacts, W&B run state, and result folders are ignored by Git.

## Installation

The project targets Python 3.10.

```bash
conda create -n isca_emulator python=3.10 pip
conda activate isca_emulator
pip install -e ".[dev]"
```

Some dependencies, especially GPU-enabled PyTorch builds and geoscience packages, may require platform-specific installation steps. If `pip install -e ".[dev]"` cannot resolve the CUDA-specific PyTorch build in `requirements.txt`, install the matching PyTorch build for your system first, then rerun the editable install.

## Configuration

Runtime configuration is stored in YAML files:

- preprocessing: `config_data/*.yaml`
- training: `config_train/hydra_config.yaml` and grouped files under `config_train/`
- evaluation: `config_inference/*.yaml`

Training and evaluation use Weights & Biases. Create a local `.env` from `.env.example` and fill in private values locally. Do not commit `.env`.

## Workflows

Preprocess data:

```bash
preprocess_data config_data/cnn3d_preprocess.yaml
```

Train the configured model:

```bash
train run-training hydra_config.yaml
```

Run a sweep:

```bash
train run-sweep
```

Evaluate W&B runs listed in an inference config:

```bash
evaluate config_inference/default.yaml
```

## Tests

After installing the development dependencies:

```bash
pytest -q
```

## Public Release Notes

Before making the repository public, confirm the license, clear or redact notebook outputs containing local paths, and decide whether public inference configs should point at published W&B artifacts, anonymous examples, or DOI-backed model checkpoints.
