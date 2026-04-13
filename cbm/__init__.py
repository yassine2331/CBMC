"""
Concept Bottleneck Models (CBM) with continuous concepts.

This package implements concept bottleneck models that use continuous
(real-valued) concept representations as an interpretable bottleneck
layer between raw inputs and final task predictions.

Modules
-------
models     : CBM architectures (sequential, joint, independent)
datasets   : Synthetic and utility dataset classes
training   : Training loops and optimisation helpers
evaluate   : Metric functions (concept, task, and alignment metrics)
utils      : Seed setting, logging, and checkpointing helpers
"""

from cbm.models import (
    ConceptEncoder,
    TaskPredictor,
    ConceptBottleneckModel,
)
from cbm.datasets import SyntheticContinuousDataset
from cbm.training import train_epoch, evaluate_epoch
from cbm.evaluate import (
    concept_mae,
    concept_r2,
    task_accuracy,
    task_r2,
)

__all__ = [
    "ConceptEncoder",
    "TaskPredictor",
    "ConceptBottleneckModel",
    "SyntheticContinuousDataset",
    "train_epoch",
    "evaluate_epoch",
    "concept_mae",
    "concept_r2",
    "task_accuracy",
    "task_r2",
]
