"""DeepTTE: trajectory CNN + LSTM + attention travel-time estimator.

Faithful reimplementation of Wang et al. (AAAI 2018) for Porto-T.
Key deviation: no multi-task local-path loss (Porto GPS is at fixed 15s
intervals so segment targets would be trivial constants); global prediction
only. All other architectural components are preserved.
"""
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ── Geo helpers ───────────────────────────────────────────────────────────────

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(a, 0.0)))


def resample_polyline(
    pts: List[List[float]],
    step_m: float = 300.0,
    max_pts: int = 120,
) -> List[List[float]]:
    """Resample a [lon, lat] polyline to fixed distance intervals.

    Avoids the trivial leakage where raw point count ∝ travel time in
    Porto's 15-second-sampled GPS data.
    """
    if len(pts) < 2:
        return pts

    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + _haversine_m(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1]))
    total = cum[-1]

    if total < step_m:
        return pts

    result = [pts[0]]
    target, j = step_m, 1
    while target < total and len(result) < max_pts:
        while j < len(pts) and cum[j] < target:
            j += 1
        if j >= len(pts):
            break
        seg = cum[j] - cum[j - 1]
        t = (target - cum[j - 1]) / seg if seg > 0 else 0.0
        result.append([
            pts[j-1][0] + t * (pts[j][0] - pts[j-1][0]),
            pts[j-1][1] + t * (pts[j][1] - pts[j-1][1]),
        ])
        target += step_m

    return result if len(result) >= 2 else pts[:2]


# ── Dataset ───────────────────────────────────────────────────────────────────

class PortoTrajDataset(Dataset):
    """Porto-T trajectory dataset for DeepTTE.

    Each sample:
      coords    (T, 2)  float32  lat/lon sequence (padded)
      cum_dist  (T, 1)  float32  cumulative distance at each resampled point (m)
      attr_ids  (3,)    int64    [taxi_idx, hour, dow]
      dist_norm float32          total trip distance / 10_000 (normalised)
      label     float32          travel_time_s
      length    int              actual sequence length (before padding)
    """

    STEP_M = 100.0

    def __init__(self, df, taxi_vocab: dict):
        import json

        self.samples = []
        for row in df.iter_rows(named=True):
            pts_raw = json.loads(row["polyline"])
            if len(pts_raw) < 2:
                continue
            pts = resample_polyline(pts_raw, step_m=self.STEP_M)
            if len(pts) < 2:
                continue

            # Cumulative distances at resampled points
            cdists = [0.0]
            for i in range(1, len(pts)):
                cdists.append(
                    cdists[-1] + _haversine_m(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
                )
            total_dist = cdists[-1]

            ts  = int(row["timestamp"])
            hour = (ts % 86400) // 3600
            dow  = (ts // 86400 + 3) % 7

            taxi_idx = taxi_vocab.get(row["taxi_id"], 0)

            # Per-step features: [delta_lat, delta_lon, step_dist_norm]
            # Delta from previous point captures direction + local speed proxy.
            # Scale deltas by 1e4 to bring degree-differences into ~[0,1] range.
            SCALE = 1e4
            feats = []
            for i in range(len(pts)):
                if i == 0:
                    dlat, dlon = 0.0, 0.0
                else:
                    dlat = (pts[i][1] - pts[i-1][1]) * SCALE
                    dlon = (pts[i][0] - pts[i-1][0]) * SCALE
                step_d = (cdists[i] - cdists[i-1]) / self.STEP_M if i > 0 else 0.0
                feats.append([dlat, dlon, step_d])

            self.samples.append({
                "coords":    feats,                  # [delta_lat, delta_lon, step_dist]
                "cum_dist":  [c / max(total_dist, 1.0) for c in cdists],  # fractional progress
                "taxi_idx":  taxi_idx,
                "hour":      int(hour),
                "dow":       int(dow),
                "dist_norm": total_dist / 10_000.0,
                "label":     float(row["travel_time_s"]),
                "length":    len(pts),
            })

    def __len__(self):
        return len(self.samples)

    def set_label_stats(self, mean: float, std: float) -> None:
        self.label_mean = mean
        self.label_std  = std

    def __getitem__(self, idx):
        s = self.samples[idx]
        coords   = torch.tensor(s["coords"],   dtype=torch.float32)
        cum_dist = torch.tensor(
            [c / 10_000.0 for c in s["cum_dist"]], dtype=torch.float32
        ).unsqueeze(1)
        attr_ids = torch.tensor([s["taxi_idx"], s["hour"], s["dow"]], dtype=torch.long)
        dist_n   = torch.tensor([s["dist_norm"]], dtype=torch.float32)
        raw      = s["label"]
        label    = torch.tensor(
            (raw - self.label_mean) / self.label_std, dtype=torch.float32
        )
        return coords, cum_dist, attr_ids, dist_n, label, s["length"], raw


def collate_fn(batch):
    coords_list, cdist_list, attr_ids, dist_ns, labels, lengths, raw_labels = zip(*batch)
    max_T = max(lengths)

    coords_pad = torch.zeros(len(batch), max_T, 3)
    cdist_pad  = torch.zeros(len(batch), max_T, 1)
    for i, (c, d, l) in enumerate(zip(coords_list, cdist_list, lengths)):
        coords_pad[i, :l] = c
        cdist_pad[i, :l]  = d

    return (
        coords_pad,
        cdist_pad,
        torch.stack(attr_ids),
        torch.stack(dist_ns),
        torch.stack(labels),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(raw_labels, dtype=torch.float32),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class GeoConvLayer(nn.Module):
    def __init__(self, kernel_size: int = 3, n_filters: int = 64):
        super().__init__()
        self.k = kernel_size
        # Input: [delta_lat, delta_lon, step_dist_norm] — 3 features per step
        self.loc_embed = nn.Linear(3, 16)
        self.conv      = nn.Conv1d(16, n_filters, kernel_size=kernel_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, T, 3) — [delta_lat*1e4, delta_lon*1e4, step_dist_norm]
        B, T, _ = coords.shape
        loc = torch.tanh(self.loc_embed(coords.view(B * T, 3))).view(B, T, 16)
        out = torch.relu(self.conv(loc.permute(0, 2, 1)))   # (B, n_filters, T-k+1)
        return out.permute(0, 2, 1)                          # (B, T-k+1, n_filters)


class DeepTTE(nn.Module):
    def __init__(
        self,
        n_taxis:    int = 500,
        n_filters:  int = 64,
        kernel_size: int = 3,
        hidden_size: int = 64,
        n_lstm_layers: int = 2,
        attr_size:  int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.kernel_size = kernel_size

        # Attribute component
        self.taxi_embed = nn.Embedding(n_taxis + 1, 16, padding_idx=0)
        self.hour_embed = nn.Embedding(24, 8)
        self.dow_embed  = nn.Embedding(7, 4)
        self.attr_fc    = nn.Sequential(
            nn.Linear(16 + 8 + 4 + 1, attr_size),
            nn.ReLU(),
        )

        # Spatio-temporal component
        self.geo_conv = GeoConvLayer(kernel_size=kernel_size, n_filters=n_filters)
        lstm_in = n_filters + 1 + attr_size        # geo_feat + cum_dist + attr
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Attention
        self.att_proj = nn.Linear(attr_size, hidden_size)

        # Global prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size + attr_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        coords:    torch.Tensor,   # (B, T, 2)
        cum_dist:  torch.Tensor,   # (B, T, 1)
        attr_ids:  torch.Tensor,   # (B, 3)
        dist_norm: torch.Tensor,   # (B, 1)
        lengths:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        B = coords.shape[0]
        k = self.kernel_size

        # Attribute vector
        attr = self.attr_fc(torch.cat([
            self.taxi_embed(attr_ids[:, 0]),
            self.hour_embed(attr_ids[:, 1]),
            self.dow_embed(attr_ids[:, 2]),
            dist_norm,
        ], dim=1))                                          # (B, attr_size)

        # Geo-conv
        geo = self.geo_conv(coords)                         # (B, T-k+1, n_filters)
        T_conv = geo.shape[1]
        cd  = cum_dist[:, :T_conv, :]                       # (B, T-k+1, 1)

        # Inject attr at every step
        attr_exp = attr.unsqueeze(1).expand(-1, T_conv, -1)
        lstm_in  = torch.cat([geo, cd, attr_exp], dim=2)   # (B, T_conv, lstm_in)

        # Pack + LSTM
        conv_lens = (lengths - k + 1).clamp(min=1)
        packed    = nn.utils.rnn.pack_padded_sequence(
            lstm_in, conv_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T_conv, H)

        # Attention
        query  = self.att_proj(attr)                        # (B, H)
        scores = torch.bmm(out, query.unsqueeze(2)).squeeze(2)  # (B, T_conv)
        mask   = torch.arange(T_conv, device=lengths.device).unsqueeze(0) >= conv_lens.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        alpha  = torch.softmax(scores, dim=1)               # (B, T_conv)
        h_att  = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)  # (B, H)

        pred = self.head(torch.cat([h_att, attr], dim=1)).squeeze(1)  # (B,)
        return pred
