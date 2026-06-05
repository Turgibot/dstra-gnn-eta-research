"""DSTRA-GNN model: GATv2 encoder + Transformer temporal + route encoder + MoE head."""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from models.dstra_gnn.config import DSTRAConfig


# ── MoE head ─────────────────────────────────────────────────────────────────

class SparseMoEHead(nn.Module):
    """Sparse Top-k Mixture-of-Experts output head."""

    def __init__(self, in_dim: int, n_experts: int = 6, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.router    = nn.Linear(in_dim, n_experts)
        self.experts   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim // 2, 1),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        scores  = self.router(x)                                     # (N, E)
        topk_v, topk_i = torch.topk(scores, self.top_k, dim=-1)     # (N, k)
        gate    = F.softmax(topk_v, dim=-1)                          # (N, k)

        out = torch.zeros(x.shape[0], device=x.device)
        for rank in range(self.top_k):
            expert_idx = topk_i[:, rank]                             # (N,)
            weight     = gate[:, rank]                               # (N,)
            for e_idx in range(self.n_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e_idx](x[mask]).squeeze(1)
        return out


# ── Graph Encoder (GATv2) ─────────────────────────────────────────────────────

class DSTRAGraphEncoder(nn.Module):
    """2-layer heterogeneous GATv2 encoder."""

    def __init__(self, cfg: DSTRAConfig, j_in: int = 4, v_in: int = 5, re_in: int = 6):
        super().__init__()
        H    = cfg.gat_heads
        hdim = cfg.hidden_dim
        out  = hdim // H   # per-head output

        # Layer 1 projections (input dims differ per node type)
        self.j_proj = nn.Linear(j_in, hdim)
        self.v_proj = nn.Linear(v_in, hdim)

        def _make_conv(src_dim, dst_dim, edge_dim=None):
            return GATv2Conv(
                in_channels=(src_dim, dst_dim),
                out_channels=out,
                heads=H,
                edge_dim=edge_dim,
                concat=True,
                dropout=cfg.dropout,
                add_self_loops=False,
            )

        self.use_dyn = cfg.use_dynamic_edges

        # Layer 1
        conv1_dict = {
            ("junction", "road", "junction"): _make_conv(hdim, hdim, edge_dim=re_in),
        }
        if self.use_dyn:
            conv1_dict.update({
                ("junction", "j2v", "vehicle"): _make_conv(hdim, hdim),
                ("vehicle",  "v2j", "junction"): _make_conv(hdim, hdim),
                ("vehicle",  "vv",  "vehicle"):  _make_conv(hdim, hdim),
            })
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")

        # Layer-1 norms
        self.norm1_j = nn.LayerNorm(hdim)
        self.norm1_v = nn.LayerNorm(hdim)

        # Layer 2 (same dims — hdim after concat)
        conv2_dict = {
            ("junction", "road", "junction"): _make_conv(hdim, hdim, edge_dim=re_in),
        }
        if self.use_dyn:
            conv2_dict.update({
                ("junction", "j2v", "vehicle"): _make_conv(hdim, hdim),
                ("vehicle",  "v2j", "junction"): _make_conv(hdim, hdim),
                ("vehicle",  "vv",  "vehicle"):  _make_conv(hdim, hdim),
            })
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")
        self.norm2_j = nn.LayerNorm(hdim)
        self.norm2_v = nn.LayerNorm(hdim)

        self.act = nn.GELU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict) -> dict:
        # Project inputs to hidden_dim
        hj = self.j_proj(x_dict["junction"])
        hv = self.v_proj(x_dict["vehicle"])
        h  = {"junction": hj, "vehicle": hv}

        # Layer 1
        ea = {"junction__road__junction": edge_attr_dict.get("junction__road__junction")}
        h1 = self.conv1(h, edge_index_dict, ea)
        h["junction"] = self.norm1_j(self.act(h1.get("junction", h["junction"]) + h["junction"]))
        h["vehicle"]  = self.norm1_v(self.act(h1.get("vehicle",  h["vehicle"])  + h["vehicle"]))

        # Layer 2
        h2 = self.conv2(h, edge_index_dict, ea)
        h["junction"] = self.norm2_j(self.act(h2.get("junction", h["junction"]) + h["junction"]))
        h["vehicle"]  = self.norm2_v(self.act(h2.get("vehicle",  h["vehicle"])  + h["vehicle"]))

        return h


# ── Temporal Aggregator ───────────────────────────────────────────────────────

class TemporalAggregator(nn.Module):
    """Per-edge Transformer over H-1 time steps → mean-pooled global context."""

    def __init__(self, cfg: DSTRAConfig, feat_in: int = 3):
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
        temporal_feats: (N_are, H-1, 3)
        Returns: (hidden_dim,) global context vector
        """
        N, T, _ = temporal_feats.shape
        if N == 0 or T == 0:
            return torch.zeros(self.input_proj.out_features,
                               device=temporal_feats.device)

        x = self.input_proj(temporal_feats)          # (N, T, hdim)
        x = self.transformer(x)                      # (N, T, hdim)
        x = x.mean(dim=1)                            # (N, hdim)  — pool over time
        x = self.out_proj(x).mean(dim=0)             # (hdim,)    — pool over edges
        return x


# ── Route Encoder ────────────────────────────────────────────────────────────

class RouteEncoder(nn.Module):
    """Mean-pool edge-ID embeddings over vehicle's route_left."""

    def __init__(self, n_road_edges: int, embed_dim: int = 64):
        super().__init__()
        # +1 for padding/unknown edge
        self.embed = nn.Embedding(n_road_edges + 1, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, route_left_idxs: List[List[int]], device) -> torch.Tensor:
        """
        route_left_idxs: list of N_v lists of int (road edge indices, 0-based)
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
    def __init__(self, cfg: DSTRAConfig, n_road_edges: int):
        super().__init__()
        self.cfg  = cfg
        hdim      = cfg.hidden_dim

        self.encoder  = DSTRAGraphEncoder(cfg)
        self.temporal = TemporalAggregator(cfg) if cfg.use_temporal else None
        self.route    = RouteEncoder(n_road_edges, cfg.route_embed_dim) if cfg.use_route else None

        # Fusion MLP before MoE
        fuse_in = hdim                                      # vehicle embedding
        if cfg.use_temporal: fuse_in += hdim               # temporal context
        if cfg.use_route:    fuse_in += cfg.route_embed_dim

        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, hdim), nn.GELU(), nn.LayerNorm(hdim)
        )
        self.head = SparseMoEHead(hdim, cfg.moe_experts, cfg.moe_top_k, cfg.dropout)

    def forward(self, graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (pred_log_z, labeled_mask)
            pred_log_z: (N_v,) predicted log-normalised ETA for ALL vehicles
            labeled_mask: (N_v,) bool — which vehicles have labels
        """
        device = graph["vehicle"].x.device

        # ── Graph Encoder ──────────────────────────────────────────────────
        x_dict = {
            "junction": graph["junction"].x,
            "vehicle":  graph["vehicle"].x,
        }
        edge_index_dict = {}
        edge_attr_dict  = {}

        for et in graph.edge_types:
            key = "__".join(et)
            edge_index_dict[et] = graph[et].edge_index
            if hasattr(graph[et], "edge_attr"):
                edge_attr_dict[key] = graph[et].edge_attr

        h = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        v_emb = h["vehicle"]                                # (N_v, hdim)

        # ── Temporal Aggregator ────────────────────────────────────────────
        parts = [v_emb]
        if self.temporal is not None:
            tf     = graph["junction"].temporal_feats.to(device)
            t_ctx  = self.temporal(tf)                      # (hdim,)
            parts.append(t_ctx.unsqueeze(0).expand(v_emb.size(0), -1))

        # ── Route Encoder ──────────────────────────────────────────────────
        if self.route is not None:
            r_emb = self.route(graph["vehicle"].route_left_idx, device)  # (N_v, r_dim)
            parts.append(r_emb)

        # ── Fusion + MoE ───────────────────────────────────────────────────
        fused    = self.fusion(torch.cat(parts, dim=1))     # (N_v, hdim)
        pred     = self.head(fused)                         # (N_v,)

        return pred, graph["vehicle"].labeled_mask
