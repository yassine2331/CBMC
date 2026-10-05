"""
concepts/categorical.py
-----------------------
Case 2 — categorical (multi-state) concepts.

Where CEM gives each concept two anchor embeddings and slides between them,
a categorical concept has `n` discrete states and therefore `n` anchors. The
predicted state distribution is a softmax, and the concept's embedding is the
convex combination of its anchors under that distribution:

    phi_{i,s} = anchor embedding for state s of concept i, from h
    logits_i  = g_i( [ phi_{i,1} || phi_{i,2} || ... || phi_{i,n} ] )   in R^{n_i}
    p_i       = softmax(logits_i)
    z_i       = sum_s  p_i[s] * phi_{i,s}(h)

Following CEM, the state logits are predicted from the CONCATENATED anchors of
that concept, not from h directly — the same way CEM predicts its probability
from cat([pos, neg]) rather than from h.

Each concept's vector is then concatenated into one tall vector for the head:

    z = [ z_1 || z_2 || ... || z_k ]     in R^{k * embedding_dim}

Two classes:

    CEMCategorical            every concept has the same number of states
    CEMCategoricalPerConcept  each concept has its own number of states

Binary (Case 1) is simply CEMCategorical with n_classes=2 — the softmax over
two anchors is equivalent to the sigmoid-weighted interpolation of CEM, so this
module subsumes it.

Unlike cem.py there is no positive/negative network pair: a single bank holds
n anchors per concept. Everything else follows CEM, including predicting the
state distribution from the concatenated anchors.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cem import _mlp


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _AnchorBank(nn.Module):
    """
    One embedding network per (concept, state).

        phi_{i,s} : R^input_dim -> R^embedding_dim

    forward(x) returns a list of length n_concepts; entry i has shape
    [B, embedding_dim, n_states_i].  A list is used rather than one tensor
    because concepts may have different numbers of states.
    """

    def __init__(self, input_dim: int, n_states: Sequence[int],
                 hidden_dim: int, embedding_dim: int,
                 depth: int, dropout: float) -> None:
        super().__init__()
        self.banks = nn.ModuleList([
            nn.ModuleList([
                _mlp(input_dim, hidden_dim, embedding_dim, depth, dropout)
                for _ in range(n)
            ])
            for n in n_states
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [torch.stack([net(x) for net in bank], dim=2)  # [B, E, n_i]
                for bank in self.banks]


class _StatePredictor(nn.Module):
    """
    One network per concept mapping that concept's CONCATENATED anchors to its
    state logits, mirroring CEM's concept predictor which reads cat([pos, neg]).

        input  : concept i's anchors [B, embedding_dim, n_i]
        output : [B, n_i] logits
    """

    def __init__(self, embedding_dim: int, n_states: Sequence[int],
                 hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.nets = nn.ModuleList([
            _mlp(embedding_dim * n, hidden_dim, n, depth, dropout)
            for n in n_states
        ])

    def forward(self, anchors: list[torch.Tensor]) -> list[torch.Tensor]:
        # anchors[i] is [B, E, n_i] -> flatten to [B, E*n_i] -> [B, n_i]
        return [net(a.flatten(1)) for a, net in zip(anchors, self.nets)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CEMCategoricalPerConcept(nn.Module):
    """
    Categorical concept bottleneck where each concept has its own state count.

    Parameters
    ----------
    input_dim : int
        Width of the backbone output.
    n_states : sequence of int
        Number of states for each concept. Its length is the concept count.
        Every entry must be >= 2.
    embedding_dim : int
        Size of each anchor embedding.
    hidden_dim, depth, dropout
        Internal MLP settings, as in CEM.
    normalize : bool
        Apply LayerNorm to the concatenated output (default True, matching CEM).

    Inputs
    ------
    x : Tensor [B, input_dim]
    interventions : LongTensor [B, n_concepts] or None
        Ground-truth state index per concept. Where used, the softmax is
        replaced by a one-hot on that state, so the concept contributes exactly
        its anchor embedding.
    intervention_mask : Tensor [B, n_concepts] or None
        1 = use the intervention, 0 = keep the prediction. Defaults to all ones
        when interventions are given.

    Outputs
    -------
    embeddings : Tensor [B, embedding_dim * n_concepts]
        Concatenation z_1 || ... || z_k, concept-major (concept i occupies
        columns [i*embedding_dim, (i+1)*embedding_dim)).
    logits : list of Tensor
        Entry i has shape [B, n_states_i]; supervise with cross-entropy.
    """

    def __init__(
        self,
        input_dim: int,
        n_states: Sequence[int],
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.2,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        n_states = list(n_states)
        if len(n_states) == 0:
            raise ValueError("n_states must name at least one concept")
        if any(n < 2 for n in n_states):
            raise ValueError(f"every concept needs >= 2 states, got {n_states}")

        self.input_dim = input_dim
        self.n_states = n_states
        self.n_concepts = len(n_states)
        self.embedding_dim = embedding_dim

        self.anchor_bank = _AnchorBank(
            input_dim, n_states, hidden_dim, embedding_dim, depth, dropout)
        self.state_predictor = _StatePredictor(
            embedding_dim, n_states, hidden_dim, depth, dropout)
        self.output_norm = (nn.LayerNorm(embedding_dim * self.n_concepts)
                            if normalize else nn.Identity())

    @property
    def output_dim(self) -> int:
        return self.embedding_dim * self.n_concepts

    def forward(
        self,
        x: torch.Tensor,
        interventions: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
    ):
        anchors = self.anchor_bank(x)            # list of [B, E, n_i]
        logits = self.state_predictor(anchors)   # list of [B, n_i]

        parts = []
        for i, (anc, lg) in enumerate(zip(anchors, logits)):
            weights = F.softmax(lg, dim=1)       # [B, n_i]

            if interventions is not None:
                true_state = interventions[:, i].long()
                one_hot = F.one_hot(true_state, num_classes=self.n_states[i])
                one_hot = one_hot.to(weights.dtype)
                if intervention_mask is None:
                    weights = one_hot
                else:
                    m = intervention_mask[:, i].unsqueeze(1).to(weights.dtype)
                    weights = m * one_hot + (1.0 - m) * weights

            # convex combination of this concept's anchors -> [B, E]
            parts.append(torch.bmm(anc, weights.unsqueeze(2)).squeeze(2))

        # concatenate, concept-major
        flat = self.output_norm(torch.cat(parts, dim=1))
        return flat, logits

    # ------------------------------------------------------------------
    @staticmethod
    def concept_loss(logits: list[torch.Tensor], targets: torch.Tensor):
        """
        Mean cross-entropy over concepts.

        targets : LongTensor [B, n_concepts] of true state indices.
        """
        losses = [F.cross_entropy(lg, targets[:, i].long())
                  for i, lg in enumerate(logits)]
        return torch.stack(losses).mean()

    @staticmethod
    def predicted_states(logits: list[torch.Tensor]) -> torch.Tensor:
        """Argmax state per concept -> LongTensor [B, n_concepts]."""
        return torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)


class CEMCategorical(CEMCategoricalPerConcept):
    """
    Categorical concept bottleneck with the same number of states for every
    concept. A thin wrapper over CEMCategoricalPerConcept.

    n_classes=2 recovers the binary case (Case 1): a softmax over two anchors
    is the sigmoid-weighted interpolation of CEM.

    >>> block = CEMCategorical(input_dim=128, n_concepts=8, n_classes=5)
    >>> z = torch.randn(4, 128)
    >>> emb, logits = block(z)
    >>> emb.shape, len(logits), logits[0].shape
    (torch.Size([4, 128]), 8, torch.Size([4, 5]))
    """

    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        n_classes: int = 2,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.2,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            n_states=[n_classes] * n_concepts,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            normalize=normalize,
        )
        self.n_classes = n_classes

    def forward(self, x, interventions=None, intervention_mask=None):
        """As the parent, but logits are stacked into one [B, C, n] tensor."""
        flat, logits = super().forward(x, interventions, intervention_mask)
        return flat, torch.stack(logits, dim=1)               # [B, C, n]

    @staticmethod
    def concept_loss(logits: torch.Tensor, targets: torch.Tensor):
        """logits [B, C, n], targets [B, C] of state indices."""
        B, C, n = logits.shape
        return F.cross_entropy(logits.reshape(B * C, n),
                               targets.reshape(B * C).long())

    @staticmethod
    def predicted_states(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=2)
