import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

from models.utils.data import extract_lr_features
from models.utils.metrics import mae, mape, rmse


@dataclass
class LRConfig:
    pass  # sklearn LinearRegression has no meaningful hyperparameters to tune


class LRBaseline:
    """Linear regression on hand-crafted geometry + temporal features (B2).

    Features: OD straight-line distance, polyline distance, tortuosity ratio,
    OD bearing (sin/cos), hour-of-day (sin/cos), day-of-week (sin/cos),
    weekend flag, holiday flag, log OD distance.
    Target: travel_time_s in seconds (no log transform — Rule 1 bounds
    the range to [315, 945 s], so the distribution is mild enough for OLS).
    """

    def __init__(self, config: Optional[LRConfig] = None) -> None:
        self.config = config or LRConfig()
        self._scaler = StandardScaler()
        self._model  = LinearRegression()
        self._feature_names = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame) -> None:
        df  = train_df.filter(pl.col("polyline") != "[]")
        X, self._feature_names = extract_lr_features(df)
        y   = df["travel_time_s"].to_numpy()
        self.fit_xy(X, y, self._feature_names)

    def fit_xy(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> None:
        """Fit directly on a pre-extracted feature matrix (e.g. SUMO features)."""
        self._feature_names = feature_names
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X), y)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        df_clean = df.filter(pl.col("polyline") != "[]")
        X, _     = extract_lr_features(df_clean)
        preds    = self._model.predict(self._scaler.transform(X))
        return np.clip(preds, 0.0, None)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pl.DataFrame, split_name: str) -> dict:
        df_clean = df.filter(pl.col("polyline") != "[]")
        X, _     = extract_lr_features(df_clean)
        y_true   = df_clean["travel_time_s"].to_numpy()
        return self.evaluate_xy(X, y_true, split_name)

    def evaluate_xy(self, X: np.ndarray, y_true: np.ndarray, split_name: str) -> dict:
        """Like `evaluate`, but on a pre-extracted feature matrix (e.g. SUMO features)."""
        y_pred = np.clip(self._model.predict(self._scaler.transform(X)), 0.0, None)

        coefs = {
            name: round(float(c), 6)
            for name, c in zip(self._feature_names, self._model.coef_)
        }
        return {
            "split":        split_name,
            "n_trips":      len(y_true),
            "mae_s":        round(mae(y_true, y_pred), 4),
            "rmse_s":       round(rmse(y_true, y_pred), 4),
            "mape_pct":     round(mape(y_true, y_pred), 4),
            "intercept":    round(float(self._model.intercept_), 4),
            "coefficients": coefs,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, model_dir / "scaler.joblib")
        joblib.dump(self._model,  model_dir / "model.joblib")
        (model_dir / "feature_names.json").write_text(
            json.dumps(self._feature_names, indent=2)
        )

    @classmethod
    def load(cls, model_dir: Path) -> "LRBaseline":
        model = cls()
        model._scaler        = joblib.load(model_dir / "scaler.joblib")
        model._model         = joblib.load(model_dir / "model.joblib")
        model._feature_names = json.loads((model_dir / "feature_names.json").read_text())
        return model
