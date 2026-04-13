# CBMC — Continuous Bottleneck Model on Continuous Concepts

CBMC is a research library for experimenting with concept bottleneck models where the concept layer is continuous and architecture-agnostic. You plug `ContinuousBottleneck` between any encoder and any task head.

```python
from cbmc import ContinuousBottleneck

bottleneck = ContinuousBottleneck(in_dim=512, n_concepts=50)
concepts   = bottleneck(encoder_output)   # (B, n_concepts)
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yassine2331/CBMC.git
cd CBMC
```

### 2. Create and activate the environment

```bash
conda create -n cbmc python=3.11 -y
conda activate cbmc
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .        # installs cbmc as an editable package
```

---

## Running training

All training is launched from the repo root with `scripts/train.py`. Configs are loaded from `experiments/configs/`.

```bash
# Train baseline CNN classifier on MNIST
python scripts/train.py --model cnn

# Train baseline vanilla VAE on MNIST
python scripts/train.py --model vae
```

To use a custom config:

```bash
python scripts/train.py --model vae --config experiments/configs/my_vae.json
```

After each run, the exact configs used are saved to `outputs/<model>/` for reproducibility.

For the VAE, sample images (originals, reconstructions, random generations) are saved to `outputs/samples/vae_baseline/` after every epoch.

---

## Project structure

```
CBMC/
├── cbmc/                      # Core library — the installable package
│   ├── bottleneck.py          # ContinuousBottleneck — the main module
│   ├── concepts/              # Concept projection and alignment
│   ├── data/                  # Dataset loaders (e.g. mnist.py)
│   ├── evaluation/            # Interpretability metrics
│   ├── losses/                # Concept-aware losses
│   ├── training/              # Training utilities
│   ├── utils/                 # Shared helpers (sampling, etc.)
│   └── configs.py             # Config dataclasses for all models
│
├── architectures/             # Example backbones showing how to plug cbmc in
│   ├── cnn_baseline.py        # Plain CNN (no bottleneck)
│   ├── vae_baseline.py        # Plain VAE (no bottleneck)
│   ├── cnn_cbmc.py            # CNN + ContinuousBottleneck
│   ├── vae_cbmc.py            # VAE + ContinuousBottleneck (diffusers)
│   └── transformer_cbmc.py    # Transformer + ContinuousBottleneck
│
├── experiments/
│   ├── configs/               # JSON configs for every model and training run
│   │   ├── cnn_baseline.json
│   │   ├── vae_baseline.json
│   │   └── train_default.json
│   └── results/               # Metrics, plots, summaries
│
├── scripts/
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Evaluation entry point
│   └── run_experiment.py      # Full experiment runner
│
├── data/                      # Raw and processed datasets (gitignored)
├── outputs/                   # Checkpoints, logs, samples (gitignored)
├── notebooks/                 # Exploration and visualization
├── tests/                     # Unit tests for cbmc/
├── third_party/               # External repos used with attribution
└── ATTRIBUTION.md             # Credits for all third-party code
```

---

## Config system

Every model and training run is controlled by JSON config files in `experiments/configs/`.

**Model config** (`cnn_baseline.json`):
```json
{
  "in_channels": 1,
  "conv_channels": [32, 64, 128],
  "fc_dims": [256],
  "n_classes": 10,
  "dropout": 0.0
}
```

**Training config** (`train_default.json`):
```json
{
  "epochs": 10,
  "lr": 0.001,
  "batch_size": 128,
  "seed": 42,
  "num_workers": 2
}
```

To run a sweep, create a new config file for each variant and launch:
```bash
python scripts/train.py --model vae --config experiments/configs/vae_latent32.json
```

---

## Adding your own model

To add a new architecture and plug in the CBMC bottleneck:

### 1. Create a config class in `cbmc/configs.py`

```python
@dataclass
class MyModelConfig(BaseConfig):
    in_dim:     int       = 512
    n_concepts: int       = 50
    n_classes:  int       = 10
    # add any hyperparameter you want to sweep
```

### 2. Create a JSON config in `experiments/configs/`

```bash
python -c "
from cbmc.configs import MyModelConfig
MyModelConfig().save('experiments/configs/my_model.json')
"
```

Edit `experiments/configs/my_model.json` to set your values.

### 3. Create the model in `architectures/`

```python
# architectures/my_model.py
import torch.nn as nn
from cbmc import ContinuousBottleneck
from cbmc.configs import MyModelConfig

class MyModel(nn.Module):
    def __init__(self, cfg: MyModelConfig = MyModelConfig()):
        super().__init__()
        self.encoder    = ...                                          # your backbone
        self.bottleneck = ContinuousBottleneck(cfg.in_dim, cfg.n_concepts)
        self.head       = nn.Linear(cfg.n_concepts, cfg.n_classes)

    def forward(self, x):
        z        = self.encoder(x)
        concepts = self.bottleneck(z)
        return self.head(concepts), concepts
```

### 4. Add a training function in `scripts/train.py`

Follow the same pattern as `train_cnn` or `train_vae` — load config, instantiate model, train loop, save config to `outputs/`.

### 5. If you use third-party code

Add an entry to `ATTRIBUTION.md` with the source URL, license, and what you changed.

---

## Reproducibility

Every training run saves its exact configs to `outputs/<model>/`:
```
outputs/vae_baseline/model_config.json
outputs/vae_baseline/train_config.json
```

To reproduce a run exactly:
```bash
python scripts/train.py --model vae --config outputs/vae_baseline/model_config.json
```
