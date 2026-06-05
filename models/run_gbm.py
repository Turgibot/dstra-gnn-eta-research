"""Run the GBM baseline (LightGBM on geometry + temporal + OD features).

Usage (from repo root):
    python models/run_gbm.py
    python models/run_gbm.py --num-leaves 127 --output-dir results/gbm_deep
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines.gbm import GBMBaseline, GBMConfig
from models.utils.data import load_split

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SPLITS_DIR = _REPO_ROOT / "datasets/real/porto_taxi_traj/processed/splits"
_OUTPUT_DIR = _REPO_ROOT / "results/gbm"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GBM baseline — LightGBM on Porto-T")
    p.add_argument("--splits-dir",     type=Path,  default=_SPLITS_DIR)
    p.add_argument("--output-dir",     type=Path,  default=_OUTPUT_DIR)
    p.add_argument("--n-estimators",   type=int,   default=1000)
    p.add_argument("--learning-rate",  type=float, default=0.05)
    p.add_argument("--num-leaves",     type=int,   default=63)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading splits ...")
    train_df = load_split(args.splits_dir, "train")
    val_df   = load_split(args.splits_dir, "val")
    test_df  = load_split(args.splits_dir, "test")
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    config = GBMConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    model = GBMBaseline(config)

    print(f"\nFitting (n_estimators={args.n_estimators}, lr={args.learning_rate}, "
          f"num_leaves={args.num_leaves}) ...")
    model.fit(train_df, val_df)

    print("\nEvaluating ...")
    val_metrics  = model.evaluate(val_df,  "val")
    test_metrics = model.evaluate(test_df, "test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model")

    results = {"val": val_metrics, "test": test_metrics}
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print("\n" + "=" * 56)
    print(f"{'Split':<6}  {'N':>8}  {'MAE (s)':>9}  {'RMSE (s)':>10}  {'MAPE (%)':>9}")
    print("-" * 56)
    for m in (val_metrics, test_metrics):
        print(
            f"{m['split']:<6}  {m['n_trips']:>8,}  "
            f"{m['mae_s']:>9.2f}  {m['rmse_s']:>10.2f}  {m['mape_pct']:>9.2f}"
        )
    print("=" * 56)
    print(f"\nBest iteration: {val_metrics['best_iteration']}")

    print("\nTop feature importances (gain):")
    for name, val in val_metrics["feature_importance"].items():
        print(f"  {name:<20s}  {val:>12,.0f}")

    print(f"\nResults → {metrics_path}")
    print(f"Model   → {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
