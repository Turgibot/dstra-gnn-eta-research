"""DSTRA-GNN model: GATv2 encoder + Transformer/GRU temporal + route encoder + sparse MoE head."""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from models.dstra_gnn.config import DSTRAConfig


# ── MoE head ─────────────────────────────────────────────────────────────────

class _ExpertMLP(nn.Module):
    """Single MoE expert: two-layer residual MLP."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.fc1(x)))
        h = self.drop(self.act(self.fc2(h)))
        return self.out(h + x)   # residual skip before output projection


class SparseMoEHead(nn.Module):
    """Sparse Top-k MoE output head with load-balancing support.

    forward() returns (predictions, importance, load):
      predictions : (N,)  per-vehicle ETA in target space
      importance  : (E,)  mean router probability per expert
      load        : (E,)  fraction of tokens routed to each expert
    """

    def __init__(
        self,
        in_dim:    int,
        n_experts: int   = 6,
        top_k:     int   = 2,
        dropout:   float = 0.1,
        noise_std: float = 0.15,
        temp:      float = 1.2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.noise_std = noise_std
        self.temp      = temp
        self.router    = nn.Linear(in_dim, n_experts)
        self.experts   = nn.ModuleList([_ExpertMLP(in_dim, dropout) for _ in range(n_experts)])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        scores = self.router(x)  # (N, E)

        # Noise + temperature during training (encourages exploration)
        if self.training and self.noise_std > 0:
            scores = scores + torch.randn_like(scores) * self.noise_std
        scores = scores / self.temp

        probs = F.softmax(scores, dim=-1)         # (N, E)  — for load stats

        topk_v, topk_i = torch.topk(scores, self.top_k, dim=-1)   # (N, k)
        gate = F.softmax(topk_v, dim=-1)                           # (N, k) renormalised

        out = torch.zeros(N, device=x.device)
        for rank in range(self.top_k):
            expert_idx = topk_i[:, rank]          # (N,)
            weight     = gate[:, rank]             # (N,)
            for e_idx in range(self.n_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e_idx](x[mask]).squeeze(1)

        # Load-balancing statistics (Switch Transformer formulation)
        importance = probs.mean(dim=0)            # (E,)  mean router prob per expert

        load = torch.zeros(self.n_experts, device=x.device)
        for rank in range(self.top_k):
            load.scatter_add_(0, topk_i[:, rank], torch.ones(N, device=x.device))
        load = load / N                            # (E,)  fraction of tokens per expert

        return out, importance, load


# ── Graph Encoder (GATv2) ─────────────────────────────────────────────────────

class DSTRAGraphEncoder(nn.Module):
    """2-layer heterogeneous GATv2 encoder."""

    def __init__(self, cfg: DSTRAConfig, j_in: int = 10, v_in: int = 22, re_in: int = 7):
        super().__init__()
        H    = cfg.gat_heads
        hdim = cfg.hidden_dim
        out  = hdim // H

        self.j_proj = nn.Linear(j_in, hdim)
        self.v_proj = nn.Linear(v_in, hdim)

        def _make_conv(edge_dim=None):
            return GATv2Conv(
                in_channels=(hdim, hdim),
                out_channels=out,
                heads=H,
                edge_dim=edge_dim,
                concat=True,
                dropout=cfg.dropout,
                add_self_loops=False,
            )

        self.use_dyn = cfg.use_dynamic_edges

        conv1_dict = {("junction", "road", "junction"): _make_conv(edge_dim=re_in)}
        if self.use_dyn:
            conv1_dict.update({
                ("junction", "j2v", "vehicle"): _make_conv(),
                ("vehicle",  "v2j", "junction"): _make_conv(),
                ("vehicle",  "vv",  "vehicle"):  _make_conv(),
            })
        self.conv1   = HeteroConv(conv1_dict, aggr="sum")
        self.norm1_j = nn.LayerNorm(hdim)
        self.norm1_v = nn.LayerNorm(hdim)

        conv2_dict = {("junction", "road", "junction"): _make_conv(edge_dim=re_in)}
        if self.use_dyn:
            conv2_dict.update({
                ("junction", "j2v", "vehicle"): _make_conv(),
                ("vehicle",  "v2j", "junction"): _make_conv(),
                ("vehicle",  "vv",  "vehicle"):  _make_conv(),
            })
        self.conv2   = HeteroConv(conv2_dict, aggr="sum")
        self.norm2_j = nn.LayerNorm(hdim)
        self.norm2_v = nn.LayerNorm(hdim)

        self.act = nn.GELU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict) -> dict:
        hj = self.j_proj(x_dict["junction"])
        hv = self.v_proj(x_dict["vehicle"])
        h  = {"junction": hj, "vehicle": hv}

        ea = {"junction__road__junction": edge_attr_dict.get("junction__road__junction")}

        h1 = self.conv1(h, edge_index_dict, ea)
        h["junction"] = self.norm1_j(self.act(h1.get("junction", h["junction"]) + h["junction"]))
        h["vehicle"]  = self.norm1_v(self.act(h1.get("vehicle",  h["vehicle"])  + h["vehicle"]))

        h2 = self.conv2(h, edge_index_dict, ea)
        h["junction"] = self.norm2_j(self.act(h2.get("junction", h["junction"]) + h["junction"]))
        h["vehicle"]  = self.norm2_v(self.act(h2.get("vehicle",  h["vehicle"])  + h["vehicle"]))

        return h


# ── Temporal Aggregator ───────────────────────────────────────────────────────

class TemporalAggregator(nn.Module):
    """Per-edge Transformer over H-1 time steps → mean-pooled global context vector."""

    def __init__(self, cfg: DSTRAConfig, feat_in: int = 4):
        super().__init__()
        hdim = cfg.hidden_dim
        self.input_proj = nn.Linear(feat_in, hdim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hdim,
            nhead=cfg.temporal_heads,
            dim_feedforward=hdim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.temporal_layers)
        self.out_proj    = nn.Linear(hdim, hdim)

    def forward(self, temporal_feats: torch.Tensor) -> torch.Tensor:
        """
        temporal_feats: (N_are, H-1, feat_in)
        Returns: (hidden_dim,) global context vector
        """
        N, T, _ = temporal_feats.shape
        if N == 0 or T == 0:
            return torch.zeros(self.input_proj.out_features, device=temporal_feats.device)

        x = self.input_proj(temporal_feats)   # (N, T, hdim)
        x = self.transformer(x)               # (N, T, hdim)
        x = x.mean(dim=1)                     # (N, hdim) — pool over time
        x = self.out_proj(x).mean(dim=0)      # (hdim,)   — pool over edges
        return x


# ── GRU Temporal Aggregator ──────────────────────────────────────────────────

class GRUTemporalAggregator(nn.Module):
    """Per-edge GRU over H-1 time steps → mean-pooled global context vector.

    Drop-in replacement for TemporalAggregator with identical interface.
    The GRU is unidirectional (causal): each snapshot only attends to past
    snapshots, which matches the temporal ordering of the sliding window.
    """

    def __init__(self, cfg: DSTRAConfig, feat_in: int = 4):
        super().__init__()
        hdim = cfg.hidden_dim
        self.input_proj = nn.Linear(feat_in, hdim)
        self.gru = nn.GRU(
            input_size=hdim,
            hidden_size=hdim,
            num_layers=cfg.temporal_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.temporal_layers > 1 else 0.0,
        )
        self.norm     = nn.LayerNorm(hdim)
        self.out_proj = nn.Linear(hdim, hdim)

    def forward(self, temporal_feats: torch.Tensor) -> torch.Tensor:
        """
        temporal_feats: (N_edges, H-1, feat_in)
        Returns: (hidden_dim,) global context vector
        """
        N, T, _ = temporal_feats.shape
        if N == 0 or T == 0:
            return torch.zeros(self.input_proj.out_features, device=temporal_feats.device)

        x = self.input_proj(temporal_feats)      # (N, T, hdim)
        out, _ = self.gru(x)                     # (N, T, hdim)
        x = self.norm(out[:, -1, :])             # (N, hdim) — last time step
        x = self.out_proj(x).mean(dim=0)         # (hdim,)   — pool over edges
        return x


# ── Route Encoder ─────────────────────────────────────────────────────────────

class RouteEncoder(nn.Module):
    """Mean-pool edge-ID embeddings over vehicle's route_left."""

    def __init__(self, n_road_edges: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_road_edges + 1, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, route_left_idxs: List[List[int]], device) -> torch.Tensor:
        """
        route_left_idxs: list of N_v lists of int (road-edge indices, 0-based)
        Returns: (N_v, embed_dim)
        """
        N_v = len(route_left_idxs)
        out = torch.zeros(N_v, self.embed_dim, device=device)
        for i, rl in enumerate(route_left_idxs):
            if rl:
                ids = torch.tensor([r + 1 for r in rl], dtype=torch.long, device=device)
                out[i] = self.embed(ids).mean(dim=0)
        return out


# ── Full DSTRA-GNN ────────────────────────────────────────────────────────────

class DSTRAGNN(nn.Module):
    def __init__(self, cfg: DSTRAConfig, n_road_edges: int, j_in: int = 10, v_in: int = 22):
        super().__init__()
        self.cfg  = cfg
        hdim      = cfg.hidden_dim
        re_in     = 7  # 3 static + 4 dynamic edge features

        self.encoder  = DSTRAGraphEncoder(cfg, j_in=j_in, v_in=v_in, re_in=re_in)
        if cfg.use_temporal:
            if cfg.temporal_type == "gru":
                self.temporal = GRUTemporalAggregator(cfg, feat_in=4)
            else:
                self.temporal = TemporalAggregator(cfg, feat_in=4)
        else:
            self.temporal = None
        self.route    = RouteEncoder(n_road_edges, cfg.route_embed_dim) if cfg.use_route else None

        fuse_in = hdim
        if cfg.use_temporal: fuse_in += hdim
        if cfg.use_route:    fuse_in += cfg.route_embed_dim

        # Two-layer fusion MLP: fuse_in → 256 → fusion_out
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, cfg.fusion_out),
        )
        self.head = SparseMoEHead(
            cfg.fusion_out, cfg.moe_experts, cfg.moe_top_k, cfg.dropout,
            noise_std=cfg.moe_noise, temp=cfg.moe_temp,
        )

    def forward(
        self, graph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_log_z  : (N_v,)  predicted log-normalised ETA for ALL vehicles
            labeled_mask: (N_v,)  bool — which vehicles have labels
            importance  : (E,)    mean router probability per expert
            load        : (E,)    fraction of tokens routed to each expert
        """
        device = graph["vehicle"].x.device

        x_dict = {
            "junction": graph["junction"].x,
            "vehicle":  graph["vehicle"].x,
        }
        edge_index_dict: dict = {}
        edge_attr_dict:  dict = {}
        for et in graph.edge_types:
            key = "__".join(et)
            edge_index_dict[et] = graph[et].edge_index
            if hasattr(graph[et], "edge_attr"):
                edge_attr_dict[key] = graph[et].edge_attr

        h     = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        v_emb = h["vehicle"]                        # (N_v, hdim)

        parts = [v_emb]
        if self.temporal is not None:
            tf    = graph["junction"].temporal_feats.to(device)
            t_ctx = self.temporal(tf)               # (hdim,)
            parts.append(t_ctx.unsqueeze(0).expand(v_emb.size(0), -1))

        if self.route is not None:
            r_emb = self.route(graph["vehicle"].route_left_idx, device)
            parts.append(r_emb)

        fused = self.fusion(torch.cat(parts, dim=1))           # (N_v, fusion_out)
        pred, importance, load = self.head(fused)              # (N_v,), (E,), (E,)

        return pred, graph["vehicle"].labeled_mask, importance, load
