"""Unit tests for cbm.datasets."""

import pytest
import torch
from torch.utils.data import DataLoader

from cbm.datasets import SyntheticContinuousDataset, make_dataloaders


class TestSyntheticContinuousDataset:
    def test_len(self):
        ds = SyntheticContinuousDataset(n_samples=200)
        assert len(ds) == 200

    def test_item_shapes(self):
        ds = SyntheticContinuousDataset(n_samples=50, in_features=10, n_concepts=4)
        x, c, y = ds[0]
        assert x.shape == (10,)
        assert c.shape == (4,)
        assert y.shape == ()  # scalar label for binary classification

    def test_regression_label_shape(self):
        ds = SyntheticContinuousDataset(
            n_samples=50, in_features=10, n_concepts=4, n_outputs=2, task="regression"
        )
        _, _, y = ds[0]
        assert y.shape == (2,)

    def test_reproducibility(self):
        ds1 = SyntheticContinuousDataset(n_samples=100, seed=0)
        ds2 = SyntheticContinuousDataset(n_samples=100, seed=0)
        x1, c1, y1 = ds1[0]
        x2, c2, y2 = ds2[0]
        assert torch.allclose(x1, x2)
        assert torch.allclose(c1, c2)

    def test_different_seeds(self):
        ds1 = SyntheticContinuousDataset(n_samples=100, seed=1)
        ds2 = SyntheticContinuousDataset(n_samples=100, seed=2)
        x1, _, _ = ds1[0]
        x2, _, _ = ds2[0]
        assert not torch.allclose(x1, x2)

    def test_classification_labels_binary(self):
        ds = SyntheticContinuousDataset(n_samples=500, task="classification")
        labels = torch.stack([ds[i][2] for i in range(len(ds))])
        unique = labels.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_concept_noise_zero(self):
        ds = SyntheticContinuousDataset(n_samples=100, concept_noise=0.0, seed=7)
        # Concepts should be pure linear projections of X; just check shape
        _, c, _ = ds[0]
        assert c.shape == (5,)

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="task"):
            SyntheticContinuousDataset(task="ranking")

    def test_splits_sizes(self):
        total = 1000
        train_ds, val_ds, test_ds = SyntheticContinuousDataset.splits(
            n_samples=total,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        assert len(train_ds) == 700
        assert len(val_ds) == 150
        assert len(test_ds) == 150

    def test_dataloader_batch_shape(self):
        ds = SyntheticContinuousDataset(n_samples=100, in_features=10, n_concepts=3)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        x_batch, c_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (16, 10)
        assert c_batch.shape == (16, 3)


class TestMakeDataloaders:
    def test_returns_three_loaders(self):
        train_ds, val_ds, test_ds = SyntheticContinuousDataset.splits(n_samples=300)
        loaders = make_dataloaders(train_ds, val_ds, test_ds, batch_size=32)
        assert len(loaders) == 3

    def test_loaders_are_iterable(self):
        train_ds, val_ds, test_ds = SyntheticContinuousDataset.splits(n_samples=300)
        train_l, val_l, test_l = make_dataloaders(train_ds, val_ds, test_ds)
        # Should be able to pull at least one batch from each.
        for loader in (train_l, val_l, test_l):
            batch = next(iter(loader))
            assert len(batch) == 3  # (x, c, y)
