"""Unit tests for cbm.models."""

import pytest
import torch

from cbm.models import ConceptBottleneckModel, ConceptEncoder, TaskPredictor, MLP


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_features=10, hidden_dims=[32, 16], out_features=5)
        x = torch.randn(8, 10)
        out = mlp(x)
        assert out.shape == (8, 5)

    def test_no_hidden(self):
        mlp = MLP(in_features=4, hidden_dims=[], out_features=2)
        x = torch.randn(3, 4)
        out = mlp(x)
        assert out.shape == (3, 2)

    def test_dropout(self):
        mlp = MLP(in_features=10, hidden_dims=[16], out_features=4, dropout=0.5)
        mlp.train()
        x = torch.randn(50, 10)
        # With dropout the output should still have the correct shape
        assert mlp(x).shape == (50, 4)


# ---------------------------------------------------------------------------
# ConceptEncoder
# ---------------------------------------------------------------------------

class TestConceptEncoder:
    def test_output_shape(self):
        enc = ConceptEncoder(in_features=20, n_concepts=5)
        x = torch.randn(16, 20)
        c = enc(x)
        assert c.shape == (16, 5)

    @pytest.mark.parametrize("activation", ["none", "sigmoid", "tanh"])
    def test_activations(self, activation):
        enc = ConceptEncoder(in_features=10, n_concepts=3, activation=activation)
        x = torch.randn(4, 10)
        c = enc(x)
        assert c.shape == (4, 3)
        if activation == "sigmoid":
            assert c.min() >= 0.0 and c.max() <= 1.0
        elif activation == "tanh":
            assert c.min() >= -1.0 and c.max() <= 1.0

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="activation"):
            ConceptEncoder(in_features=10, n_concepts=3, activation="relu")


# ---------------------------------------------------------------------------
# TaskPredictor
# ---------------------------------------------------------------------------

class TestTaskPredictor:
    def test_output_shape(self):
        pred = TaskPredictor(in_features=5, n_outputs=1)
        c = torch.randn(8, 5)
        y = pred(c)
        assert y.shape == (8, 1)

    def test_multiclass_output(self):
        pred = TaskPredictor(in_features=5, n_outputs=4)
        c = torch.randn(8, 5)
        y = pred(c)
        assert y.shape == (8, 4)


# ---------------------------------------------------------------------------
# ConceptBottleneckModel
# ---------------------------------------------------------------------------

class TestConceptBottleneckModel:
    @pytest.mark.parametrize("mode", ["joint", "sequential", "independent"])
    def test_forward_output_shapes(self, mode):
        model = ConceptBottleneckModel(
            in_features=20, n_concepts=5, n_outputs=1, mode=mode
        )
        x = torch.randn(8, 20)
        c_pred, y_pred = model(x)
        assert c_pred.shape == (8, 5), f"concept shape mismatch for mode={mode}"
        assert y_pred.shape == (8, 1), f"task shape mismatch for mode={mode}"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            ConceptBottleneckModel(in_features=10, n_concepts=3, mode="bad_mode")

    def test_intervention(self):
        model = ConceptBottleneckModel(in_features=20, n_concepts=5, n_outputs=1)
        x = torch.randn(4, 20)
        c_true = torch.randn(4, 5)
        mask = torch.ones(4, 5, dtype=torch.bool)

        c_pred_no_int, _ = model(x)
        c_pred_int, _ = model(x, intervention=c_true, intervention_mask=mask)

        # With full mask the bottleneck should equal the ground-truth concepts
        assert torch.allclose(c_pred_int, c_true)

    def test_partial_intervention(self):
        model = ConceptBottleneckModel(in_features=20, n_concepts=4, n_outputs=1)
        x = torch.randn(4, 20)
        c_true = torch.randn(4, 4)
        # Only intervene on the first 2 concepts
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:, :2] = True

        c_pred, _ = model(x, intervention=c_true, intervention_mask=mask)
        assert torch.allclose(c_pred[:, :2], c_true[:, :2])

    def test_freeze_unfreeze_encoder(self):
        model = ConceptBottleneckModel(in_features=10, n_concepts=3, n_outputs=1)
        model.freeze_encoder()
        for p in model.concept_encoder.parameters():
            assert not p.requires_grad

        model.unfreeze_encoder()
        for p in model.concept_encoder.parameters():
            assert p.requires_grad

    def test_encode_helper(self):
        model = ConceptBottleneckModel(in_features=10, n_concepts=3, n_outputs=1)
        x = torch.randn(5, 10)
        c = model.encode(x)
        assert c.shape == (5, 3)

    def test_gradient_flow_joint(self):
        """Gradients should flow to both encoder and predictor in joint mode."""
        model = ConceptBottleneckModel(
            in_features=10, n_concepts=3, n_outputs=1, mode="joint"
        )
        x = torch.randn(4, 10)
        c_pred, y_pred = model(x)
        loss = y_pred.sum() + c_pred.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_independent_mode_uses_raw_input(self):
        """In independent mode the predictor input dim equals in_features."""
        model = ConceptBottleneckModel(
            in_features=20, n_concepts=5, n_outputs=1, mode="independent"
        )
        # TaskPredictor's first Linear layer should have in_features=20
        first_layer = list(model.task_predictor.net.net.modules())
        linear_layers = [m for m in first_layer if isinstance(m, torch.nn.Linear)]
        assert linear_layers[0].in_features == 20
