"""Unit tests for cbm.training."""

import pytest
import torch
from torch.utils.data import DataLoader

from cbm.datasets import SyntheticContinuousDataset
from cbm.models import ConceptBottleneckModel
from cbm.training import train_epoch, evaluate_epoch, train


def _make_loader(n_samples=200, batch_size=32, task="classification"):
    ds = SyntheticContinuousDataset(
        n_samples=n_samples, in_features=10, n_concepts=3, task=task, seed=0
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_model(mode="joint", task="classification"):
    return ConceptBottleneckModel(
        in_features=10,
        n_concepts=3,
        n_outputs=1,
        encoder_hidden=[32],
        predictor_hidden=[16],
        mode=mode,
    )


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------

class TestTrainEpoch:
    def test_returns_expected_keys(self):
        model = _make_model()
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_epoch(model, loader, opt)
        assert set(metrics.keys()) == {"loss", "concept_loss", "task_loss"}

    def test_loss_is_positive(self):
        model = _make_model()
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_epoch(model, loader, opt)
        assert metrics["loss"] > 0

    def test_alpha_zero_zeroes_concept_loss_weight(self):
        """When alpha=0 the concept loss should not affect total loss."""
        model = _make_model()
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_epoch(model, loader, opt, alpha=0.0)
        # total = 0 * concept + 1 * task  => total ≈ task_loss
        assert abs(metrics["loss"] - metrics["task_loss"]) < 1e-5

    def test_alpha_one_zeroes_task_loss_weight(self):
        """When alpha=1 the task loss should not affect total loss."""
        model = _make_model()
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_epoch(model, loader, opt, alpha=1.0)
        assert abs(metrics["loss"] - metrics["concept_loss"]) < 1e-5

    def test_regression_task(self):
        model = _make_model(task="regression")
        loader = _make_loader(task="regression")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = train_epoch(model, loader, opt, task="regression")
        assert metrics["loss"] > 0


# ---------------------------------------------------------------------------
# evaluate_epoch
# ---------------------------------------------------------------------------

class TestEvaluateEpoch:
    def test_no_grad_applied(self):
        """Parameters should not receive gradients during evaluation."""
        model = _make_model()
        loader = _make_loader()
        evaluate_epoch(model, loader)
        for p in model.parameters():
            assert p.grad is None

    def test_returns_expected_keys(self):
        model = _make_model()
        loader = _make_loader()
        metrics = evaluate_epoch(model, loader)
        assert set(metrics.keys()) == {"loss", "concept_loss", "task_loss"}


# ---------------------------------------------------------------------------
# train (high-level loop)
# ---------------------------------------------------------------------------

class TestTrain:
    @pytest.mark.parametrize("mode", ["joint", "sequential", "independent"])
    def test_training_reduces_loss(self, mode):
        """Loss at the end should be lower (or similar) than at the start."""
        torch.manual_seed(0)
        model = _make_model(mode=mode)
        train_ds, val_ds, _ = SyntheticContinuousDataset.splits(
            n_samples=400, in_features=10, n_concepts=3, seed=0
        )
        train_l = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_l = DataLoader(val_ds, batch_size=32, shuffle=False)

        history = train(
            model, train_l, val_l,
            n_epochs=10, patience=5, verbose=False
        )

        assert len(history["train_loss"]) > 0
        # Final loss should be lower than initial loss
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_history_keys(self):
        model = _make_model()
        train_ds, val_ds, _ = SyntheticContinuousDataset.splits(
            n_samples=200, in_features=10, n_concepts=3, seed=1
        )
        train_l = DataLoader(train_ds, batch_size=32)
        val_l = DataLoader(val_ds, batch_size=32)
        history = train(model, train_l, val_l, n_epochs=3, patience=3, verbose=False)

        expected_keys = {
            "train_loss", "val_loss",
            "train_concept_loss", "val_concept_loss",
            "train_task_loss", "val_task_loss",
        }
        assert set(history.keys()) == expected_keys

    def test_early_stopping(self):
        """With patience=1 and enough epochs the loop should stop early."""
        torch.manual_seed(0)
        model = _make_model()
        train_ds, val_ds, _ = SyntheticContinuousDataset.splits(
            n_samples=200, in_features=10, n_concepts=3, seed=2
        )
        train_l = DataLoader(train_ds, batch_size=32)
        val_l = DataLoader(val_ds, batch_size=32)
        history = train(
            model, train_l, val_l,
            n_epochs=200, patience=1, verbose=False
        )
        # With patience=1 we expect fewer than 200 epochs
        assert len(history["train_loss"]) < 200
