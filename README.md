# Isca Emulation Repository

Research code for training and evaluating machine learning emulators of Isca climate model simulations. The repository contains preprocessing pipelines, model definitions, training and evaluation entry points, experiment configuration files, and scripts reproducing the revised paper and Supporting Information.

## Repository Layout

- `src/isca_emulation_v2/`: Python package with data processing, models, training, evaluation, plotting, and W&B helpers.
- `config_data/`: preprocessing configurations.
- `config_train/`: Hydra training configurations.
- `config_inference/`: evaluation configurations.
- `tests/`: unit tests for data utilities and model components.
- `scripts/`: paper and Supporting Information figures and tables.
- `results/paper/`: paper and Supporting Information figures.

Data, checkpoints, W&B working files and populated analysis manifests are excluded
from version control.

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

The analysis scripts require a populated local manifest and matching evaluated data:

```bash
python scripts/reproduce.py all
```

See [scripts/README.md](scripts/README.md) for setup and
[results/paper/README.md](results/paper/README.md) for the figures.

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
