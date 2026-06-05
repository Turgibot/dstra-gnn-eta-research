"""MetaTTE-GRU: trajectory GRU + residual FC travel-time estimator.

Implements the inner model of Wang et al. (IEEE T-ITS 2022) — the MetaTTE-GRU
variant — trained on Porto only (standard supervised training, no Reptile
meta-learning across cities). Architecture follows Section IV and Table VI
of the paper: embedding dim D=64, GRU units n_r=64, residual widths
1024/512/256/64.

Input representation: same delta-coordinate + 100m resampling as DeepTTE
for consistency across baselines.
"""
import math
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.baselines.deeptte import (
    PortoTrajDataset,   # reuse data loading / resampling
    collate_fn,
)


# ── Residual block ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc    = nn.Linear(in_dim, out_dim)
        self.bn    = nn.BatchNorm1d(out_dim)
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.fc(x))) + self.proj(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class MetaTTEGRU(nn.Module):
    """MetaTTE-GRU inner model (Porto-only training).

    Attribute component is identical to DeepTTE for fair comparison.
    The spatio-temporal component replaces GeoConv+LSTM with a plain GRU
    over the same per-step delta-coordinate features.
    """

    def __init__(
        self,
        n_taxis:    int   = 500,
        embed_dim:  int   = 64,    # D in paper
        gru_hidden: int   = 64,    # n_r in paper
        gru_layers: int   = 2,
        attr_size:  int   = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # Attribute component (same as DeepTTE)
        self.taxi_embed = nn.Embedding(n_taxis + 1, 16, padding_idx=0)
        self.hour_embed = nn.Embedding(24, 8)
        self.dow_embed  = nn.Embedding(7, 4)
        self.attr_fc    = nn.Sequential(
            nn.Linear(16 + 8 + 4 + 1, attr_size),
            nn.ReLU(),
        )

        # Trajectory encoder: GRU over delta-coordinate steps
        # Input per step: [delta_lat, delta_lon, step_dist_norm, cum_dist_frac]
        gru_in = 3 + 1 + attr_size   # coords(3) + cum_dist(1) + attr broadcast
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Attention (same mechanism as DeepTTE)
        self.att_proj = nn.Linear(attr_size, gru_hidden)

        # Residual FC head: paper Table VI widths 1024/512/256/64
        head_in = gru_hidden + attr_size
        self.head = nn.Sequential(
            ResidualBlock(head_in, 1024, dropout),
            ResidualBlock(1024,    512,  dropout),
            ResidualBlock(512,     256,  dropout),
            ResidualBlock(256,     64,   dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        coords:    torch.Tensor,   # (B, T, 3)
        cum_dist:  torch.Tensor,   # (B, T, 1)
        attr_ids:  torch.Tensor,   # (B, 3)
        dist_norm: torch.Tensor,   # (B, 1)
        lengths:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:

        # Attribute vector
        attr = self.attr_fc(torch.cat([
            self.taxi_embed(attr_ids[:, 0]),
            self.hour_embed(attr_ids[:, 1]),
            self.dow_embed(attr_ids[:, 2]),
            dist_norm,
        ], dim=1))                                              # (B, attr_size)

        T = coords.shape[1]
        attr_exp  = attr.unsqueeze(1).expand(-1, T, -1)
        gru_input = torch.cat([coords, cum_dist, attr_exp], dim=2)  # (B, T, gru_in)

        # Pack + GRU
        packed   = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, H)

        # Attention
        query  = self.att_proj(attr)                            # (B, H)
        scores = torch.bmm(out, query.unsqueeze(2)).squeeze(2)  # (B, T)
        mask   = torch.arange(T, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        alpha  = torch.softmax(scores, dim=1)
        h_att  = torch.bmm(alpha.unsqueeze(1), out).squeeze(1) # (B, H)

        pred = self.head(torch.cat([h_att, attr], dim=1)).squeeze(1)  # (B,)
        return pred
