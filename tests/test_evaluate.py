"""Unit tests for cbm.evaluate."""

import pytest
import torch
from torch.utils.data import DataLoader

from cbm.datasets import SyntheticContinuousDataset
from cbm.evaluate import (
    concept_mae,
    concept_r2,
    task_accuracy,
    task_r2,
    full_evaluation,
    intervention_gain,
)
from cbm.models import ConceptBottleneckModel


# ---------------------------------------------------------------------------
# Tensor-level metric tests
# ---------------------------------------------------------------------------

class TestConceptMAE:
    def test_perfect_predictions(self):
        c = torch.randn(50, 5)
        assert concept_mae(c, c) == pytest.approx(0.0, abs=1e-6)

    def test_known_error(self):
        c_pred = torch.zeros(10, 2)
        c_true = torch.ones(10, 2)
        assert concept_mae(c_pred, c_true) == pytest.approx(1.0, abs=1e-5)


class TestConceptR2:
    def test_perfect_r2(self):
        c = torch.randn(100, 4)
        r2 = concept_r2(c, c)
        assert r2 == pytest.approx(1.0, abs=1e-4)

    def test_r2_below_one(self):
        c_true = torch.randn(100, 4)
        c_pred = torch.randn(100, 4)
        r2 = concept_r2(c_pred, c_true)
        assert r2 <= 1.0


class TestTaskAccuracy:
    def test_perfect_binary(self):
        y_true = torch.tensor([0, 1, 1, 0]).float()
        # logits: large positive => predicted 1, large negative => predicted 0
        y_pred = torch.tensor([-5.0, 5.0, 5.0, -5.0]).unsqueeze(-1)
        acc = task_accuracy(y_pred, y_true)
        assert acc == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = torch.tensor([0, 1, 1, 0]).float()
        y_pred = torch.tensor([5.0, -5.0, -5.0, 5.0]).unsqueeze(-1)
        acc = task_accuracy(y_pred, y_true)
        assert acc == pytest.approx(0.0)

    def test_multiclass(self):
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([
            [10.0, -1.0, -1.0],
            [-1.0, 10.0, -1.0],
            [-1.0, -1.0, 10.0],
        ])
        acc = task_accuracy(y_pred, y_true)
        assert acc == pytest.approx(1.0)


class TestTaskR2:
    def test_perfect_r2(self):
        y = torch.randn(100, 1)
        assert task_r2(y, y) == pytest.approx(1.0, abs=1e-4)

    def test_r2_below_one(self):
        y_true = torch.randn(100, 1)
        y_pred = torch.randn(100, 1)
        r2 = task_r2(y_pred, y_true)
        assert r2 <= 1.0


# ---------------------------------------------------------------------------
# Dataloader-level tests
# ---------------------------------------------------------------------------

def _make_loader(task="classification", n_samples=200):
    ds = SyntheticContinuousDataset(
        n_samples=n_samples, in_features=10, n_concepts=3, task=task, seed=0
    )
    return DataLoader(ds, batch_size=32, shuffle=False)


def _make_model():
    return ConceptBottleneckModel(
        in_features=10, n_concepts=3, n_outputs=1,
        encoder_hidden=[32], predictor_hidden=[16],
    )


class TestFullEvaluation:
    def test_classification_keys(self):
        model = _make_model()
        loader = _make_loader("classification")
        metrics = full_evaluation(model, loader, task="classification")
        assert "concept_mae" in metrics
        assert "concept_r2" in metrics
        assert "task_accuracy" in metrics
        assert "task_r2" not in metrics

    def test_regression_keys(self):
        model = _make_model()
        loader = _make_loader("regression")
        metrics = full_evaluation(model, loader, task="regression")
        assert "task_r2" in metrics
        assert "task_accuracy" not in metrics

    def test_accuracy_in_range(self):
        model = _make_model()
        loader = _make_loader("classification")
        metrics = full_evaluation(model, loader, task="classification")
        assert 0.0 <= metrics["task_accuracy"] <= 1.0

    def test_concept_mae_nonneg(self):
        model = _make_model()
        loader = _make_loader("classification")
        metrics = full_evaluation(model, loader, task="classification")
        assert metrics["concept_mae"] >= 0.0


class TestInterventionGain:
    def test_keys(self):
        model = _make_model()
        loader = _make_loader("classification")
        gain = intervention_gain(model, loader, task="classification")
        assert set(gain.keys()) == {"baseline", "intervened"}

    def test_intervention_improves_or_equals_baseline(self):
        """After enough training, intervening with true concepts should not hurt."""
        import torch
        from cbm.training import train

        torch.manual_seed(0)
        model = _make_model()
        train_ds, val_ds, _ = SyntheticContinuousDataset.splits(
            n_samples=600, in_features=10, n_concepts=3, seed=0
        )
        train_l = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_l = DataLoader(val_ds, batch_size=32, shuffle=False)
        train(model, train_l, val_l, n_epochs=30, patience=10, verbose=False)

        loader = _make_loader("classification", n_samples=300)
        gain = intervention_gain(model, loader, task="classification")
        # Intervened accuracy >= baseline - 0.1 (some slack for random init)
        assert gain["intervened"] >= gain["baseline"] - 0.1
