"""Sliding-window dataset for DSTRA-GNN.

Each item = a window of H consecutive snapshots.
Label = ETA for each vehicle present in the LAST snapshot of the window.
Temporal aggregator sees road-edge features from the first H-1 snapshots.
"""
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from models.dstra_gnn.data_utils import (
    build_hetero_graph,
    build_static_index,
    extract_road_dyn_features,
    load_snapshot_raw,
    slim_snapshot,
)


class SlidingWindowDataset(Dataset):
    """Sliding window dataset over graph snapshots.

    For small datasets (≤ _PRELOAD_THRESHOLD snapshots): pre-loads all snapshots
    into RAM at init time.  For large datasets: loads from disk on demand.
    """

    _PRELOAD_THRESHOLD = 10_000

    def __init__(
        self,
        split_dir:         Path,
        static_index:      dict,
        window_size:       int   = 30,
        use_dynamic_edges: bool  = True,
        label_log_mean:    Optional[float] = None,
        label_log_std:     Optional[float] = None,
        demand_stats:      Optional[Dict[str, float]] = None,
        max_windows:       Optional[int]   = None,
        force_preload:     bool            = False,
    ):
        self.si           = static_index
        self.H            = window_size
        self.use_dyn      = use_dynamic_edges
        self.log_mean     = label_log_mean
        self.log_std      = label_log_std
        self.demand_stats = demand_stats
        self.trip_filter_ids: Optional[Set[str]] = None
        self.label_filter_range: Optional[Tuple[float, float]] = None

        snap_dir  = split_dir / "snapshots"
        label_dir = split_dir / "labels"

        self.snap_files  = sorted(snap_dir.iterdir())
        self.label_files = {
            f.stem.split("_")[1]: f
            for f in label_dir.iterdir()
        }

        # Filter out empty/corrupt files
        self.snap_files = [f for f in self.snap_files if f.stat().st_size > 10]

        self.n_snaps   = len(self.snap_files)
        self.n_windows = max(0, self.n_snaps - self.H + 1)

        if max_windows is not None and max_windows < self.n_windows:
            self.n_windows  = max_windows
            self.snap_files = self.snap_files[:self.n_windows + self.H - 1]
            self.n_snaps    = len(self.snap_files)

        self._cache: Optional[Dict[str, dict]] = None
        if self.n_snaps <= self._PRELOAD_THRESHOLD or force_preload:
            print(f"    Pre-loading {self.n_snaps:,} snapshots into RAM …")
            self._cache = {}
            re_id2idx = self.si["re_id2idx"]
            for f in self.snap_files:
                raw = load_snapshot_raw(f)
                if raw is not None:
                    self._cache[f.stem] = slim_snapshot(raw, re_id2idx)
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

    def set_demand_stats(self, demand_stats: Dict[str, float]) -> None:
        self.demand_stats = demand_stats

    def set_trip_filter(self, in_range_ids: Optional[Set[str]]) -> None:
        """Restrict learning/metrics to vehicles whose whole trip's total
        duration (ETA at trip start) falls in a target range — i.e. the
        vehicle's trip population matches MetaTTE's "typical trip" Rule-1
        selection (see compute_trip_durations). Intersected with the existing
        labeled_mask, so the model still sees the full graph context — only
        which vehicles' predictions contribute to the loss/metrics is filtered."""
        self.trip_filter_ids = in_range_ids

    def set_label_filter(self, label_range: Optional[Tuple[float, float]]) -> None:
        """Restrict learning/metrics to individual (vehicle, snapshot)
        predictions whose own remaining-ETA target (y_raw) falls in
        [lo, hi] seconds — independent of which trip they belong to. Contrast
        with set_trip_filter, which keeps every snapshot of an in-range trip
        regardless of that snapshot's own remaining ETA. Intersected with the
        existing labeled_mask; can be combined with a trip filter (AND)."""
        self.label_filter_range = label_range

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        window_files = self.snap_files[idx: idx + self.H]
        curr_file    = window_files[-1]
        ts_str       = curr_file.stem.split("_")[1]

        curr_snap = self._load_snap(curr_file)
        if curr_snap is None or not curr_snap.get("nodes"):
            return None

        labels_map = self._load_labels(ts_str)
        if not labels_map:
            return None

        graph = build_hetero_graph(
            snap=curr_snap,
            labels_map=labels_map,
            si=self.si,
            use_dynamic_edges=self.use_dyn,
            demand_stats=self.demand_stats,
        )
        if graph is None:
            return None

        if self.trip_filter_ids is not None:
            in_trip = torch.tensor(
                [vid in self.trip_filter_ids for vid in graph["vehicle"].v_ids],
                dtype=torch.bool,
            )
            graph["vehicle"].labeled_mask = graph["vehicle"].labeled_mask & in_trip

        if self.label_filter_range is not None:
            lo, hi = self.label_filter_range
            y_raw = graph["vehicle"].y_raw
            in_label_range = (y_raw >= lo) & (y_raw <= hi)
            graph["vehicle"].labeled_mask = graph["vehicle"].labeled_mask & in_label_range

        # Temporal features: dynamic road-edge feats from first H-1 snapshots
        # Shape: (N_re_total, H-1, 4)  [avg_speed, density, demand_z, occ_norm]
        H_minus1 = self.H - 1
        N_re     = self.si["n_road_edges"]
        temporal_feats = np.zeros((N_re, H_minus1, 4), dtype=np.float32)
        for t, snap_file in enumerate(window_files[:-1]):
            snap = self._load_snap(snap_file)
            if snap is None:
                continue
            temporal_feats[:, t, :] = extract_road_dyn_features(snap, self.si, self.demand_stats)

        active_re = graph["junction"].active_re_idxs
        graph["junction"].temporal_feats = torch.from_numpy(temporal_feats[active_re])

        return graph


def collate_windows(batch):
    """Keep valid items only; return list (batch_size or mini-batch as list)."""
    return [item for item in batch if item is not None]


def compute_label_stats(
    dataset: "SlidingWindowDataset",
    vehicle_filter: Optional[Set[str]] = None,
    label_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Compute log-ETA mean and std directly from the dataset's label files.

    If vehicle_filter is given, only labels for vehicle IDs in that set are
    used (see SlidingWindowDataset.set_trip_filter). If label_range is given,
    only labels whose own ETA falls in [lo, hi] are used (see
    SlidingWindowDataset.set_label_filter). Either or both can be supplied —
    the normalisation then matches the population the model is actually
    trained on.
    """
    all_log_etas = []
    # Restrict to label files covered by dataset.snap_files — keeps this
    # consistent with any --max-train-windows cap (dataset.label_files is
    # built from the full split directory listing, which can span a wider
    # range than the possibly-truncated snap_files).
    snap_ts     = {f.stem.split("_")[1] for f in dataset.snap_files}
    label_files = [lf for ts, lf in dataset.label_files.items() if ts in snap_ts]
    if len(label_files) > 5000:
        label_files = random.sample(label_files, 5000)

    for lf in label_files:
        raw = lf.read_text()
        if not raw.strip():
            continue
        lbl = json.loads(raw)
        for entry in lbl.get("labels", []):
            if vehicle_filter is not None and entry.get("id") not in vehicle_filter:
                continue
            eta = float(entry.get("eta", float("nan")))
            if label_range is not None and not (label_range[0] <= eta <= label_range[1]):
                continue
            if eta > 0:
                all_log_etas.append(math.log1p(eta))

    if not all_log_etas:
        return 0.0, 1.0
    arr = np.array(all_log_etas)
    return float(arr.mean()), float(arr.std())


def compute_trip_durations(
    dataset: "SlidingWindowDataset",
    max_workers: int = 8,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Total trip duration per vehicle ID = ETA at its chronologically-first
    labeled snapshot (remaining-time-to-destination at trip start equals the
    trip's total duration — same definition the eval script uses for its
    once-per-trip MetaTTE-style metric).

    Requires a full chronological scan of every label file (first appearance
    can't be determined any other way), so label loading is parallelised with
    a thread pool while the chronological merge stays sequential.
    """
    ts_strs = [f.stem.split("_")[1] for f in dataset.snap_files]
    durations: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        label_maps = ex.map(dataset._load_labels, ts_strs)
        for labels in tqdm(label_maps, total=len(ts_strs),
                           desc="trip-durations", unit="snap", disable=not show_progress):
            for vid, eta in labels.items():
                if vid not in durations:
                    durations[vid] = eta
    return durations


def trip_ids_in_range(durations: Dict[str, float], lo: float, hi: float) -> Set[str]:
    """Vehicle IDs whose total trip duration falls in [lo, hi] seconds."""
    return {vid for vid, dur in durations.items() if lo <= dur <= hi}


def compute_demand_stats(
    dataset: "SlidingWindowDataset",
    max_samples: int = 3000,
) -> Dict[str, float]:
    """Compute edge_demand mean/std and occupancy 99th-pct from training snapshots.

    Used to z-score demand and cap-normalise occupancy features.
    """
    snap_files = list(dataset.snap_files)
    if len(snap_files) > max_samples:
        snap_files = random.sample(snap_files, max_samples)

    demands, occs = [], []
    for f in snap_files:
        snap = dataset._load_snap(f)
        if snap is None:
            continue
        for e in snap.get("road_edges_dynamic", []):
            d = e.get("edge_demand", 0.0)
            if d > 0:
                demands.append(float(d))
            occ = e.get("vehicles_on_road", 0)
            if isinstance(occ, list):
                occ = len(occ)
            if occ > 0:
                occs.append(float(occ))

    if not demands:
        return {"demand_mean": 0.0, "demand_std": 1.0, "occ_max": 30.0}

    d_arr = np.array(demands)
    o_arr = np.array(occs) if occs else np.array([30.0])
    return {
        "demand_mean": float(d_arr.mean()),
        "demand_std":  float(max(d_arr.std(), 1e-6)),
        "occ_max":     float(np.percentile(o_arr, 99)),
    }
