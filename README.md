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
├── configs/           # Project configuration files
├── datasets/          # Dataset loaders and extracted data
├── models/            # CBM, CEM, and CBM-VAE implementations
├── notebooks/         # Experiment notebooks and visual analysis
├── PyTorchVAE/        # VAE framework and baseline implementations
└── test.py            # Example or test entry point
```

## Example experiment result

Add a screenshot or exported figure from your experiments to the repository, for example:

`docs/images/experiments.png`

Then it will appear here:

![Experiment results](docs/images/experiments.png)

## Getting started

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd CBMC
```

### 2. Install dependencies
If you are using the VAE framework inside `PyTorchVAE`:

```bash
cd PyTorchVAE
pip install -r requirements.txt
```

### 3. Run experiments
Depending on the experiment setup, you can use files such as:

```bash
python test.py
```

or inside `PyTorchVAE`:

```bash
python run.py
```

## Why this project matters

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