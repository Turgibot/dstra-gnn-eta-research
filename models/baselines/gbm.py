import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import polars as pl

from models.utils.data import extract_gbm_features
from models.utils.metrics import mae, mape, rmse


@dataclass
class GBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 50
    grid_resolution: float = 0.01


class GBMBaseline:
    """LightGBM gradient-boosted trees on geometry + temporal + OD features (B3).

    Uses raw integer hour/dow/month and OD grid cells instead of cyclical
    encodings — trees learn non-linear splits without normalisation.
    Trains with early stopping on validation MAE.
    Never uses n_points (= travel_time_s / 15, pure leakage).
    """

    def __init__(self, config: Optional[GBMConfig] = None) -> None:
        self.config = config or GBMConfig()
        self._booster: Optional[lgb.Booster] = None
        self._feature_names = None
        self._best_iteration: Optional[int] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
        tr = train_df.filter(pl.col("polyline") != "[]")
        va = val_df.filter(pl.col("polyline") != "[]")

        X_tr, self._feature_names = extract_gbm_features(tr, self.config.grid_resolution)
        X_va, _                   = extract_gbm_features(va, self.config.grid_resolution)
        y_tr = tr["travel_time_s"].to_numpy()
        y_va = va["travel_time_s"].to_numpy()

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self._feature_names)
        dval   = lgb.Dataset(X_va, label=y_va, feature_name=self._feature_names, reference=dtrain)

        params = {
            "objective":         "regression_l1",   # MAE loss
            "metric":            "mae",
            "learning_rate":     self.config.learning_rate,
            "num_leaves":        self.config.num_leaves,
            "min_child_samples": self.config.min_child_samples,
            "subsample":         self.config.subsample,
            "colsample_bytree":  self.config.colsample_bytree,
            "reg_alpha":         self.config.reg_alpha,
            "reg_lambda":        self.config.reg_lambda,
            "verbosity":         -1,
        }

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(100),
        ]

        self._booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )
        self._best_iteration = self._booster.best_iteration

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        df_clean = df.filter(pl.col("polyline") != "[]")
        X, _     = extract_gbm_features(df_clean, self.config.grid_resolution)
        return np.clip(self._booster.predict(X), 0.0, None)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pl.DataFrame, split_name: str) -> dict:
        df_clean = df.filter(pl.col("polyline") != "[]")
        X, _     = extract_gbm_features(df_clean, self.config.grid_resolution)
        y_pred   = np.clip(self._booster.predict(X), 0.0, None)
        y_true   = df_clean["travel_time_s"].to_numpy()

        importances = self._booster.feature_importance(importance_type="gain")
        top10 = sorted(
            zip(self._feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "split":              split_name,
            "n_trips":            len(df_clean),
            "mae_s":              round(mae(y_true, y_pred), 4),
            "rmse_s":             round(rmse(y_true, y_pred), 4),
            "mape_pct":           round(mape(y_true, y_pred), 4),
            "best_iteration":     self._best_iteration,
            "feature_importance": {name: round(val, 2) for name, val in top10},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(model_dir / "model.lgb"))
        state = {
            "config":          asdict(self.config),
            "best_iteration":  self._best_iteration,
            "feature_names":   self._feature_names,
        }
        (model_dir / "state.json").write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> "GBMBaseline":
        state  = json.loads((model_dir / "state.json").read_text())
        model  = cls(GBMConfig(**state["config"]))
        model._booster        = lgb.Booster(model_file=str(model_dir / "model.lgb"))
        model._best_iteration = state["best_iteration"]
        model._feature_names  = state["feature_names"]
        return model
