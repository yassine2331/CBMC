# CBMC

**CBMC** is a research project for learning **interpretable latent representations** with **Concept Bottleneck Models (CBMs)** and **Variational Autoencoders (VAEs)**.

The repository combines:
- **Concept-based prediction models**
- **VAE-based representation learning**
- **Experiments on shape datasets**, including **Shapes3D**
- **Training, evaluation, and notebook-based analysis**

## Project goal

The main goal of this project is to study how **high-level concepts** can be used inside generative and predictive models to make learned representations more **interpretable**, **controllable**, and easier to analyze.

In practice, this repository explores:
- learning latent variables with VAEs
- aligning latent dimensions with human-understandable concepts
- testing concept bottleneck approaches on synthetic visual datasets
- comparing different model variants and configurations

## Main components

### Models
The project contains several model implementations, including:
- **CBM-based models**
- **CEM-based models**
- **CBM-VAE variants**
- a full **PyTorchVAE** submodule with multiple VAE baselines

Relevant folders:
- `models/`
- `models/CBMs/`
- `PyTorchVAE/models/`

### Datasets
The experiments use synthetic datasets designed for disentanglement and concept learning, such as:
- **Shapes**
- **3D Shapes**

Relevant folder:
- `datasets/`

### Experiments and notebooks
The repository also includes:
- configuration files for experiments
- notebooks for exploration and visualization
- tests for multiple VAE variants

Relevant folders:
- `configs/`
- `notebooks/`
- `PyTorchVAE/configs/`
- `PyTorchVAE/tests/`

## Repository structure

```text
CBMC/
├── configs/           # Base configuration dataclass (TrainingConfig)
├── datasets/          # Dataset loaders and extracted data
├── experiments/       # One subdirectory per experiment (config + notes)
│   ├── exp_01_baseline_vae/
│   │   ├── config.py  # Overrides for this specific experiment
│   │   └── notes.md   # Hypothesis, results, next steps
│   └── exp_02_deep_encoder/
│       ├── config.py
│       └── notes.md
├── models/            # CBM, CEM, and CBM-VAE implementations
│   └── registry.py    # Model registry — add new architectures here
├── notebooks/         # Experiment notebooks and visual analysis
├── PyTorchVAE/        # VAE framework and baseline implementations
└── run.py             # Universal entry-point — never needs editing
```

## How to run experiments

```bash
# Run a specific experiment (loads experiments/<name>/config.py)
python run.py --experiment exp_01_baseline_vae
python run.py --experiment exp_02_deep_encoder

# Run with the default config (TrainingConfig defaults)
python run.py
```

## How to add a new experiment (without editing any existing file)

```bash
# 1. Copy an existing experiment as starting point
cp -r experiments/exp_01_baseline_vae experiments/exp_03_my_new_idea
touch experiments/exp_03_my_new_idea/__init__.py

# 2. Edit only the config values you want to change
#    (model_name, block_out_channels, latent_channels, etc.)
nano experiments/exp_03_my_new_idea/config.py

# 3. Run it — no other file needs to change
python run.py --experiment exp_03_my_new_idea

# 4. Write up what you found
nano experiments/exp_03_my_new_idea/notes.md
```

### Naming convention for experiments

`exp_<two-digit-number>_<short_slug>`

The number keeps experiments in the order they were tried; the slug reminds
you what the idea was when you come back weeks later.

## How to add a new model architecture

1. Create `models/my_new_arch.py` with your model class.
2. Open `models/registry.py` and add:

```python
from models.my_new_arch import MyNewArch
register_model("my_new_arch", MyNewArch)
```

3. In your experiment config set `model_name = "my_new_arch"`.
4. `run.py` will automatically pick it up — no other changes needed.

## Block type reference

| String in config | Class | Use case |
|-----------------|-------|----------|
| `"DownEncoderBlock2D"` | `DownEncoderBlock2D` | Plain VAE encoder downsampling |
| `"AttnDownEncoderBlock2D"` | `AttnDownEncoderBlock2D` | Self-attention encoder downsampling |
| `"CrossAttnDownBlock2D"` | `CrossAttnDownBlock2D` | Cross-attention encoder (concept conditioning) |
| `"UpDecoderBlock2D"` | `UpDecoderBlock2D` | Plain VAE decoder upsampling |
| `"AttnUpDecoderBlock2D"` | `AttnUpDecoderBlock2D` | Self-attention decoder upsampling |

## Getting started

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd CBMC
```

### 2. Install dependencies
```bash
pip install torch diffusers
```

### 3. Run an experiment
```bash
# Run the baseline experiment
python run.py --experiment exp_01_baseline_vae

# Run the deeper encoder experiment
python run.py --experiment exp_02_deep_encoder

# Run with default settings
python run.py
```

This project is useful for research on:
- **interpretable machine learning**
- **disentangled representation learning**
- **concept supervision**
- **generative modeling**

It provides a practical codebase for testing how concept-aware architectures behave on controlled visual datasets.

## Notes

- Some experiment outputs and logs are stored under `PyTorchVAE/logs/`
- Dataset files such as `3dshapes.h5` are included for local experimentation
- Notebooks can be used to inspect training behavior and learned representations

## License

Add your preferred license information here.