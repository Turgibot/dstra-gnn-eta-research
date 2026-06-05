"""Sliding-window dataset for DSTRA-GNN.

Each item = a window of H consecutive snapshots.
Label = ETA for each vehicle present in the LAST snapshot of the window.
Temporal aggregator sees road-edge features from the first H-1 snapshots.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from models.dstra_gnn.data_utils import (
    build_hetero_graph,
    build_static_index,
    extract_road_dyn_features,
    load_snapshot_raw,
)


class SlidingWindowDataset(Dataset):
    """Sliding window dataset over graph snapshots.

    For SUMO (small): pre-loads all snapshots into RAM at init.
    For Porto-G (large): loads from disk on demand.
    """

    # Threshold: if n_snapshots <= this, pre-load into RAM.
    _PRELOAD_THRESHOLD = 50_000

    def __init__(
        self,
        split_dir:         Path,
        static_index:      dict,
        window_size:       int   = 30,
        use_dynamic_edges: bool  = True,
        label_log_mean:    Optional[float] = None,
        label_log_std:     Optional[float] = None,
    ):
        self.si        = static_index
        self.H         = window_size
        self.use_dyn   = use_dynamic_edges
        self.log_mean  = label_log_mean
        self.log_std   = label_log_std

        snap_dir  = split_dir / "snapshots"
        label_dir = split_dir / "labels"

        self.snap_files  = sorted(snap_dir.iterdir())
        self.label_files = {
            f.stem.split("_")[1]: f
            for f in label_dir.iterdir()
        }

        # Filter out empty/corrupt files
        valid = []
        for f in self.snap_files:
            if f.stat().st_size > 10:
                valid.append(f)
        self.snap_files = valid

        self.n_snaps = len(self.snap_files)
        self.n_windows = max(0, self.n_snaps - self.H + 1)

        # Pre-load into RAM if dataset is small enough
        self._cache: Optional[Dict[str, dict]] = None
        if self.n_snaps <= self._PRELOAD_THRESHOLD:
            print(f"    Pre-loading {self.n_snaps:,} snapshots into RAM …")
            self._cache = {}
            for f in self.snap_files:
                raw = load_snapshot_raw(f)
                if raw is not None:
                    self._cache[f.stem] = raw
            print(f"    Cached {len(self._cache):,} snapshots.")

    def _load_snap(self, path: Path) -> Optional[dict]:
        if self._cache is not None:
            return self._cache.get(path.stem)
        return load_snapshot_raw(path)

    def _load_labels(self, ts_str: str) -> Dict[str, float]:
        lf = self.label_files.get(ts_str)
        if lf is None:
            return {}
        raw = lf.read_text()
        if not raw.strip():
            return {}
        lbl = json.loads(raw)
        return {entry["id"]: float(entry["eta"]) for entry in lbl.get("labels", [])}

    def set_label_stats(self, log_mean: float, log_std: float) -> None:
        self.log_mean = log_mean
        self.log_std  = log_std

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        # Window: snap[idx] ... snap[idx + H - 1]
        # Current (labelled) snapshot = snap[idx + H - 1]
        window_files = self.snap_files[idx: idx + self.H]
        curr_file    = window_files[-1]
        ts_str       = curr_file.stem.split("_")[1]

        # Load current snapshot + labels
        curr_snap = self._load_snap(curr_file)
        if curr_snap is None or not curr_snap.get("nodes"):
            return None

        labels_map = self._load_labels(ts_str)
        if not labels_map:
            return None

        # Build graph from current snapshot
        graph = build_hetero_graph(
            snap=curr_snap,
            labels_map=labels_map,
            si=self.si,
            use_dynamic_edges=self.use_dyn,
        )
        if graph is None:
            return None

        # Temporal features: road-edge dynamic feats from first H-1 snapshots
        # Shape: (N_re_total, H-1, 3)
        H_minus1 = self.H - 1
        N_re = self.si["n_road_edges"]
        temporal_feats = np.zeros((N_re, H_minus1, 3), dtype=np.float32)
        for t, snap_file in enumerate(window_files[:-1]):
            snap = self._load_snap(snap_file)
            if snap is None:
                continue
            temporal_feats[:, t, :] = extract_road_dyn_features(snap, self.si)

        # Slice to active road edges only (saves memory + compute)
        active_re = graph["junction"].active_re_idxs
        temporal_active = temporal_feats[active_re]   # (N_are, H-1, 3)

        graph["junction"].temporal_feats = torch.from_numpy(temporal_active)

        return graph


def collate_windows(batch):
    """Keep valid items only; return list (batch_size=1 or mini-batch as list)."""
    return [item for item in batch if item is not None]


def compute_label_stats(dataset: SlidingWindowDataset) -> Tuple[float, float]:
    """Compute log-ETA mean and std over all labeled vehicles in the dataset.
    Samples a subset for efficiency on large datasets.
    """
    import random
    all_log_etas = []
    indices = list(range(len(dataset)))
    if len(indices) > 5000:
        indices = random.sample(indices, 5000)

    for idx in indices:
        item = dataset[idx]
        if item is None:
            continue
        y_raw = item["vehicle"].y_raw
        mask  = item["vehicle"].labeled_mask
        if mask.sum() == 0:
            continue
        y_valid = y_raw[mask]
        log_eta = torch.log1p(y_valid)
        all_log_etas.extend(log_eta.tolist())

    if not all_log_etas:
        return 0.0, 1.0
    arr = np.array(all_log_etas)
    return float(arr.mean()), float(arr.std())
