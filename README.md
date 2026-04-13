# CBMC – Concept Bottleneck Models with Continuous Concepts

A PyTorch library for training and evaluating **Concept Bottleneck Models (CBMs)**
that use *continuous* (real-valued) concept representations as an interpretable
bottleneck between raw inputs and final task predictions.

---

## Overview

Concept Bottleneck Models (Koh et al., 2020) are interpretable neural networks
that first predict human-understandable *concepts* from raw inputs, and then use
those concept predictions to make a final task prediction.  This repository
focuses on the **continuous** variant, where concept values are real numbers
rather than binary flags.

```
x  ──► Concept Encoder ──► ĉ (continuous concepts) ──► Task Predictor ──► ŷ
                              ↑ concept supervision
```

### Training modes

| Mode | Description |
|------|-------------|
| `joint` | Concept encoder + task predictor trained end-to-end with a weighted loss `α · concept_loss + (1-α) · task_loss`. |
| `sequential` | Phase 1: train encoder with `α=1`; Phase 2: freeze encoder, train predictor with `α=0`. |
| `independent` | Encoder and predictor trained independently; the predictor uses raw inputs (no bottleneck). |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

```python
import torch
from cbm import ConceptBottleneckModel, SyntheticContinuousDataset
from cbm.datasets import make_dataloaders
from cbm.training import train
from cbm.evaluate import full_evaluation, intervention_gain

# 1. Create a synthetic dataset with continuous concepts
train_ds, val_ds, test_ds = SyntheticContinuousDataset.splits(
    n_samples=2000,
    in_features=20,
    n_concepts=5,
    task="classification",
    seed=42,
)
train_loader, val_loader, test_loader = make_dataloaders(
    train_ds, val_ds, test_ds, batch_size=64
)

# 2. Build the model
model = ConceptBottleneckModel(
    in_features=20,
    n_concepts=5,
    n_outputs=1,
    mode="joint",           # or "sequential" / "independent"
)

# 3. Train
history = train(
    model, train_loader, val_loader,
    task="classification",
    alpha=0.5,              # concept loss weight
    n_epochs=100,
    patience=15,
)

# 4. Evaluate
metrics = full_evaluation(model, test_loader, task="classification")
print(metrics)
# {'concept_mae': ..., 'concept_r2': ..., 'task_accuracy': ...}

# 5. Test-time concept intervention
gain = intervention_gain(model, test_loader, task="classification")
print(gain)
# {'baseline': ..., 'intervened': ...}
```

### Run the experiment script

```bash
python experiments/train_cbm.py \
    --mode joint \
    --task classification \
    --n_samples 2000 \
    --n_concepts 5 \
    --alpha 0.5 \
    --n_epochs 100 \
    --save_dir results/
```

---

## Repository structure

```
CBMC/
├── cbm/
│   ├── __init__.py      # public API
│   ├── models.py        # ConceptEncoder, TaskPredictor, ConceptBottleneckModel
│   ├── datasets.py      # SyntheticContinuousDataset + DataLoader helpers
│   ├── training.py      # train_epoch, evaluate_epoch, train (with early stopping)
│   ├── evaluate.py      # concept_mae, concept_r2, task_accuracy, task_r2, …
│   └── utils.py         # set_seed, save/load checkpoint, plot_training_curves
├── experiments/
│   └── train_cbm.py     # end-to-end experiment script
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_datasets.py
│   ├── test_training.py
│   └── test_evaluate.py
├── requirements.txt
└── README.md
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Key concepts

* **Continuous concepts** – concept scores are unconstrained real values (or
  optionally bounded via `sigmoid`/`tanh` activation).  This avoids the
  information bottleneck caused by forcing binary concept predictions.

* **Concept alignment (R²)** – `concept_r2` measures how well the model
  recovers the ground-truth concept values.

* **Test-time intervention** – replacing predicted concepts with ground-truth
  values (`intervention_gain`) quantifies how much concept quality limits task
  performance.

---

## Reference

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., &
Liang, P. (2020). *Concept Bottleneck Models*. ICML 2020.
