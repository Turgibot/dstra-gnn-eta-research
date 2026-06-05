import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.utils.metrics import mae, mape, rmse


@dataclass
class RouteSumConfig:
    n_workers: int = 8
    min_speed_ms: float = 0.5  # clamp to avoid division by near-zero


class RouteSumBaseline:
    """Route-sum baseline on Porto-G graph data (B4).

    Predicts ETA = sum(edge_length / hist_speed) over each vehicle's
    route_left. hist_speed per edge is the mean observed avg_speed from
    train snapshots (avg_speed > 0); falls back to the static design speed
    for edges never observed in training.
    """

    def __init__(self, config: Optional[RouteSumConfig] = None) -> None:
        self.config = config or RouteSumConfig()
        self._edge_length: Dict[str, float] = {}
        self._edge_speed: Dict[str, float] = {}
        self._global_speed: float = 8.33  # ~30 km/h, overwritten after fit

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_dir: Path) -> None:
        """Build per-edge speed lookup from train split snapshots."""
        static = json.loads((train_dir / "static.json").read_text())

        for e in static["road_edges"]:
            self._edge_length[e["id"]] = float(e["length"])

        # Accumulate observed speeds across all train snapshots.
        speed_sum: Dict[str, float] = {}
        speed_cnt: Dict[str, int] = {}

        snap_files = sorted((train_dir / "snapshots").iterdir())

        def _extract_speeds(f: Path):
            raw = f.read_text()
            if not raw.strip():
                return []
            snap = json.loads(raw)
            pairs = []
            for e in snap["road_edges_dynamic"]:
                if e["avg_speed"] > 0:
                    pairs.append((e["id"], float(e["avg_speed"])))
            return pairs

        print(f"  Scanning {len(snap_files):,} train snapshots …")
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as pool:
            for pairs in pool.map(_extract_speeds, snap_files, chunksize=500):
                for eid, spd in pairs:
                    if eid in speed_sum:
                        speed_sum[eid] += spd
                        speed_cnt[eid] += 1
                    else:
                        speed_sum[eid] = spd
                        speed_cnt[eid] = 1

        observed = len(speed_cnt)
        n_edges = len(self._edge_length)
        print(f"  Observed speeds for {observed:,} / {n_edges:,} edges "
              f"({100*observed/n_edges:.1f}%); rest use global mean fallback.")

        # Global fallback = mean of observed-only speeds (not speed limits).
        observed_means = [speed_sum[e] / speed_cnt[e] for e in speed_cnt]
        if observed_means:
            self._global_speed = float(np.mean(observed_means))

        for eid in self._edge_length:
            if eid in speed_cnt:
                self._edge_speed[eid] = speed_sum[eid] / speed_cnt[eid]
            else:
                # Speed limit is NOT a travel speed; use observed mean as fallback.
                self._edge_speed[eid] = self._global_speed

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_vehicle(self, route_left: List[str]) -> float:
        if not route_left:
            return 0.0
        total = 0.0
        for eid in route_left:
            length = self._edge_length.get(eid, 50.0)
            speed = max(self._edge_speed.get(eid, self._global_speed),
                        self.config.min_speed_ms)
            total += length / speed
        return total

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, split_dir: Path, split_name: str) -> dict:
        """Evaluate on all (snapshot, vehicle) pairs in split_dir."""
        label_files = sorted((split_dir / "labels").iterdir())
        snap_dir = split_dir / "snapshots"

        y_true_all: List[float] = []
        y_pred_all: List[float] = []

        def _process(lbl_file: Path):
            ts_str = lbl_file.stem.split("_")[1]
            snap_file = snap_dir / f"step_{ts_str}.json"
            if not snap_file.exists():
                return [], []
            raw_snap = snap_file.read_text()
            raw_lbl  = lbl_file.read_text()
            if not raw_snap.strip() or not raw_lbl.strip():
                return [], []
            lbl = json.loads(raw_lbl)
            snap = json.loads(raw_snap)
            node_map = {n["id"]: n for n in snap["nodes"]}
            ys, yp = [], []
            for entry in lbl["labels"]:
                node = node_map.get(entry["id"])
                if node is None or not node["route_left"]:
                    continue
                ys.append(float(entry["eta"]))
                yp.append(self._predict_vehicle(node["route_left"]))
            return ys, yp

        print(f"  Evaluating {len(label_files):,} snapshots ({split_name}) …")
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as pool:
            for ys, yp in pool.map(_process, label_files, chunksize=200):
                y_true_all.extend(ys)
                y_pred_all.extend(yp)

        y_true = np.array(y_true_all, dtype=np.float64)
        y_pred = np.array(y_pred_all, dtype=np.float64)

        return {
            "split":    split_name,
            "n_pairs":  int(len(y_true)),
            "mae_s":    round(mae(y_true, y_pred), 4),
            "rmse_s":   round(rmse(y_true, y_pred), 4),
            "mape_pct": round(mape(y_true, y_pred), 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "global_speed": self._global_speed,
            "edge_speed":   self._edge_speed,
            "edge_length":  self._edge_length,
        }
        (model_dir / "state.json").write_text(json.dumps(state))

    @classmethod
    def load(cls, model_dir: Path) -> "RouteSumBaseline":
        state = json.loads((model_dir / "state.json").read_text())
        model = cls()
        model._global_speed = state["global_speed"]
        model._edge_speed   = state["edge_speed"]
        model._edge_length  = state["edge_length"]
        return model
