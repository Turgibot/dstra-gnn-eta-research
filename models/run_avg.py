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
from models.utils.sumo_features import build_sumo_trip_table, extract_sumo_avg_features

_REPO_ROOT    = Path(__file__).resolve().parent.parent
_SPLITS_DIR   = _REPO_ROOT / "datasets/real/porto_taxi_traj/processed/splits"
_OUTPUT_DIR   = _REPO_ROOT / "results/avg"
_SUMO_DIR     = _REPO_ROOT / "datasets/simulated/sumo"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVG baseline — MetaTTE-style Porto-T / SUMO evaluation")
    p.add_argument("--dataset",         choices=["porto", "sumo"], default="porto")
    p.add_argument("--splits-dir",      type=Path,  default=_SPLITS_DIR)
    p.add_argument("--sumo-dir",        type=Path,  default=_SUMO_DIR)
    p.add_argument("--output-dir",      type=Path,  default=None)
    p.add_argument("--grid-resolution", type=float, default=0.01,
                   help="Degrees per OD grid cell, Porto only (default 0.01 ≈ 1 km)")
    p.add_argument("--grid-m",          type=float, default=1000.0,
                   help="Meters per OD grid cell, SUMO only (default 1000)")
    p.add_argument("--n-time-slots",    type=int,   default=24,
                   help="Hour-of-day buckets (default 24)")
    p.add_argument("--train-stride",    type=int,   default=10,
                   help="SUMO only: subsample train label files")
    p.add_argument("--eval-stride",     type=int,   default=1,
                   help="SUMO only: subsample val/test label files")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = _REPO_ROOT / ("results/avg_sumo" if args.dataset == "sumo" else "results/avg")
    return args


def main() -> None:
    args = _parse_args()
    config = AVGConfig(n_time_slots=args.n_time_slots)
    model = AVGBaseline(config)

    if args.dataset == "sumo":
        model.config.grid_resolution = args.grid_m  # record-keeping only (meters, not degrees)
        print("Building SUMO trip tables ...")
        train_trips = build_sumo_trip_table(args.sumo_dir / "train", stride=args.train_stride)
        val_trips   = build_sumo_trip_table(args.sumo_dir / "val",   stride=args.eval_stride)
        test_trips  = build_sumo_trip_table(args.sumo_dir / "test",  stride=args.eval_stride)
        print(f"  train {len(train_trips):,}  val {len(val_trips):,}  test {len(test_trips):,}")

        train_feats = extract_sumo_avg_features(train_trips, args.grid_m, args.n_time_slots)
        val_feats   = extract_sumo_avg_features(val_trips,   args.grid_m, args.n_time_slots)
        test_feats  = extract_sumo_avg_features(test_trips,  args.grid_m, args.n_time_slots)

        print(f"\nFitting on train (grid={args.grid_m}m, slots={args.n_time_slots}) ...")
        model.fit_prefeaturized(train_feats)

        print("Evaluating ...")
        val_metrics  = model.evaluate_prefeaturized(val_feats,  "val")
        test_metrics = model.evaluate_prefeaturized(test_feats, "test")
    else:
        print("Loading splits ...")
        train_df = load_split(args.splits_dir, "train")
        val_df   = load_split(args.splits_dir, "val")
        test_df  = load_split(args.splits_dir, "test")
        print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

        model.config.grid_resolution = args.grid_resolution

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
