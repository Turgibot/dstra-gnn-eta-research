"""Run the AVG baseline (MetaTTE-style OD + hour lookup).

Usage (from repo root):
    python models/run_avg.py
    python models/run_avg.py --grid-resolution 0.005 --output-dir results/avg_fine
"""

import argparse
import json
import sys
from pathlib import Path

# Allow invocation as `python models/run_avg.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines.avg import AVGBaseline, AVGConfig
from models.utils.data import load_split

_REPO_ROOT    = Path(__file__).resolve().parent.parent
_SPLITS_DIR   = _REPO_ROOT / "datasets/real/porto_taxi_traj/processed/splits"
_OUTPUT_DIR   = _REPO_ROOT / "results/avg"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVG baseline — MetaTTE-style Porto-T evaluation")
    p.add_argument("--splits-dir",      type=Path,  default=_SPLITS_DIR)
    p.add_argument("--output-dir",      type=Path,  default=_OUTPUT_DIR)
    p.add_argument("--grid-resolution", type=float, default=0.01,
                   help="Degrees per OD grid cell (default 0.01 ≈ 1 km)")
    p.add_argument("--n-time-slots",    type=int,   default=24,
                   help="Hour-of-day buckets (default 24)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading splits ...")
    train_df = load_split(args.splits_dir, "train")
    val_df   = load_split(args.splits_dir, "val")
    test_df  = load_split(args.splits_dir, "test")
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    config = AVGConfig(
        grid_resolution=args.grid_resolution,
        n_time_slots=args.n_time_slots,
    )
    model = AVGBaseline(config)

    print(f"\nFitting on train (grid={args.grid_resolution}°, slots={args.n_time_slots}) ...")
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

    for m in (val_metrics, test_metrics):
        cov = m["coverage"]
        print(
            f"{m['split']}  coverage — "
            f"L1(OD+hr) {cov['level1_od_hour_pct']:.1f}%  "
            f"L2(OD) {cov['level2_od_pct']:.1f}%  "
            f"L3(global) {cov['level3_global_pct']:.1f}%"
        )

    print(f"\nResults → {metrics_path}")
    print(f"Model   → {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
