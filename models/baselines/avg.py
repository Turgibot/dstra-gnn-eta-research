import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from models.utils.data import extract_features
from models.utils.metrics import mae, mape, rmse


@dataclass
class AVGConfig:
    grid_resolution: float = 0.01  # degrees per OD cell (~1 km in Porto)
    n_time_slots: int = 24          # hour-of-day buckets (0–23)


class AVGBaseline:
    """Historical OD-style average baseline (MetaTTE Table IV, B1).

    Predicts travel time by looking up the mean over training trips that share
    the same origin grid cell, destination grid cell, and hour-of-day.
    Falls back to OD-only mean, then global mean when lookup misses.
    """

    def __init__(self, config: Optional[AVGConfig] = None) -> None:
        self.config = config or AVGConfig()
        self._lookup_od_hour: Optional[pl.DataFrame] = None
        self._lookup_od: Optional[pl.DataFrame] = None
        self._global_mean: Optional[float] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame) -> None:
        feats = extract_features(train_df, self.config.grid_resolution, self.config.n_time_slots)

        key_od_hour = ["origin_cx", "origin_cy", "dest_cx", "dest_cy", "hour"]
        key_od      = ["origin_cx", "origin_cy", "dest_cx", "dest_cy"]

        self._lookup_od_hour = (
            feats.group_by(key_od_hour)
            .agg(pl.col("travel_time_s").mean().alias("mean_tt"))
        )
        self._lookup_od = (
            feats.group_by(key_od)
            .agg(pl.col("travel_time_s").mean().alias("mean_tt"))
        )
        self._global_mean = float(feats["travel_time_s"].mean())

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_full(self, df: pl.DataFrame) -> dict:
        """Return predictions array plus per-level coverage counts."""
        feats = extract_features(df, self.config.grid_resolution, self.config.n_time_slots)

        # Level 1: OD + hour
        joined = feats.join(
            self._lookup_od_hour,
            on=["origin_cx", "origin_cy", "dest_cx", "dest_cy", "hour"],
            how="left",
        ).rename({"mean_tt": "pred_l1"})

        # Level 2: OD only
        joined = joined.join(
            self._lookup_od.rename({"mean_tt": "pred_l2"}),
            on=["origin_cx", "origin_cy", "dest_cx", "dest_cy"],
            how="left",
        )

        # Coalesce L1 → L2 → global
        joined = joined.with_columns(
            pl.coalesce([pl.col("pred_l1"), pl.col("pred_l2")]).alias("pred")
        )

        n = len(joined)
        n_l1 = int(joined["pred_l1"].is_not_null().sum())
        n_l2 = int((joined["pred_l1"].is_null() & joined["pred_l2"].is_not_null()).sum())
        n_l3 = n - n_l1 - n_l2

        predictions = joined["pred"].fill_null(self._global_mean).to_numpy()

        return {
            "predictions": predictions,
            "coverage": {
                "level1_od_hour_pct": round(100.0 * n_l1 / n, 2),
                "level2_od_pct":      round(100.0 * n_l2 / n, 2),
                "level3_global_pct":  round(100.0 * n_l3 / n, 2),
            },
        }

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self._predict_full(df)["predictions"]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pl.DataFrame, split_name: str) -> dict:
        result   = self._predict_full(df)
        y_pred   = result["predictions"]
        y_true   = df["travel_time_s"].to_numpy()

        return {
            "split":    split_name,
            "n_trips":  len(df),
            "mae_s":    round(mae(y_true, y_pred), 4),
            "rmse_s":   round(rmse(y_true, y_pred), 4),
            "mape_pct": round(mape(y_true, y_pred), 4),
            "coverage": result["coverage"],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self._lookup_od_hour.write_parquet(model_dir / "lookup_od_hour.parquet")
        self._lookup_od.write_parquet(model_dir / "lookup_od.parquet")
        state = {
            "global_mean": self._global_mean,
            "config": {
                "grid_resolution": self.config.grid_resolution,
                "n_time_slots":    self.config.n_time_slots,
            },
        }
        (model_dir / "state.json").write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> "AVGBaseline":
        state  = json.loads((model_dir / "state.json").read_text())
        config = AVGConfig(**state["config"])
        model  = cls(config)
        model._lookup_od_hour = pl.read_parquet(model_dir / "lookup_od_hour.parquet")
        model._lookup_od      = pl.read_parquet(model_dir / "lookup_od.parquet")
        model._global_mean    = state["global_mean"]
        return model
