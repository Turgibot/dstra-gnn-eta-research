"""Run the route-sum baseline on Porto-G graph data (B4).

Predicts per-vehicle ETA = sum(edge_length / hist_speed) over route_left.
Historical speed is the train-split mean observed dynamic speed per edge,
falling back to static design speed for unobserved edges.

Usage (from repo root):
    python models/run_route_sum.py
    python models/run_route_sum.py --graph-dir datasets/real/porto_taxi_graph --output-dir results/route_sum
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines.route_sum import RouteSumBaseline, RouteSumConfig

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_GRAPH_DIR  = _REPO_ROOT / "datasets/real/porto_taxi_graph"
_OUTPUT_DIR = _REPO_ROOT / "results/route_sum"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route-sum baseline — Porto-G evaluation")
    p.add_argument("--graph-dir",   type=Path, default=_GRAPH_DIR)
    p.add_argument("--output-dir",  type=Path, default=_OUTPUT_DIR)
    p.add_argument("--n-workers",   type=int,  default=8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    train_dir = args.graph_dir / "train"
    val_dir   = args.graph_dir / "val"
    test_dir  = args.graph_dir / "test"

    config = RouteSumConfig(n_workers=args.n_workers)
    model  = RouteSumBaseline(config)

    print("Fitting on train split …")
    t0 = time.time()
    model.fit(train_dir)
    print(f"  Fit done in {time.time()-t0:.1f}s  |  global fallback speed: {model._global_speed:.2f} m/s")

    print("\nEvaluating …")
    val_metrics  = model.evaluate(val_dir,  "val")
    test_metrics = model.evaluate(test_dir, "test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model")

    results = {"val": val_metrics, "test": test_metrics}
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print(f"{'Split':<6}  {'N pairs':>10}  {'MAE (s)':>9}  {'RMSE (s)':>10}  {'MAPE (%)':>9}")
    print("-" * 60)
    for m in (val_metrics, test_metrics):
        print(
            f"{m['split']:<6}  {m['n_pairs']:>10,}  "
            f"{m['mae_s']:>9.2f}  {m['rmse_s']:>10.2f}  {m['mape_pct']:>9.2f}"
        )
    print("=" * 60)
    print(f"\nResults → {metrics_path}")
    print(f"Model   → {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
