"""Run the LR baseline (linear regression on geometry + temporal features).

Usage (from repo root):
    python models/run_lr.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines.lr import LRBaseline, LRConfig
from models.utils.data import load_split

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SPLITS_DIR = _REPO_ROOT / "datasets/real/porto_taxi_traj/processed/splits"
_OUTPUT_DIR = _REPO_ROOT / "results/lr"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LR baseline — geometry + temporal features")
    p.add_argument("--splits-dir", type=Path, default=_SPLITS_DIR)
    p.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading splits ...")
    train_df = load_split(args.splits_dir, "train")
    val_df   = load_split(args.splits_dir, "val")
    test_df  = load_split(args.splits_dir, "test")
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    model = LRBaseline()

    print("\nFitting ...")
    model.fit(train_df)

    print("Evaluating ...")
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

    # Top coefficients by magnitude
    coefs = val_metrics["coefficients"]
    ranked = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nCoefficients (scaled, by magnitude):")
    for name, val in ranked:
        print(f"  {name:<20s}  {val:+.2f}")

    print(f"\nResults → {metrics_path}")
    print(f"Model   → {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
