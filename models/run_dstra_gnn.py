"""Train and evaluate DSTRA-GNN on SUMO or Porto-G.

Usage:
    # Full model (A6) on SUMO:
    .venv-deeptte/bin/python models/run_dstra_gnn.py --dataset sumo --variant A6

    # All ablations on Porto-G:
    for V in A1 A2 A3 A4 A5 A6; do
        .venv-deeptte/bin/python models/run_dstra_gnn.py --dataset porto --variant $V
    done

    # Specify seed:
    .venv-deeptte/bin/python models/run_dstra_gnn.py --dataset sumo --variant A6 --seed 43
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dstra_gnn.config import VARIANTS, DSTRAConfig
from models.dstra_gnn.data_utils import build_static_index
from models.dstra_gnn.dataset import (
    SlidingWindowDataset,
    collate_windows,
    compute_label_stats,
)
from models.dstra_gnn.model import DSTRAGNN
from models.utils.metrics import mae, mape, rmse

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATHS = {
    "sumo":  _REPO_ROOT / "datasets/simulated/sumo",
    "porto": _REPO_ROOT / "datasets/real/porto_taxi_graph",
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=["sumo", "porto"], default="sumo")
    p.add_argument("--variant",    choices=list(VARIANTS),    default="A6")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--epochs",     type=int,   default=None,  help="Override cfg epochs")
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--output-dir", type=Path,  default=None)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--workers",    type=int,   default=2)
    return p.parse_args()


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _log_normalise(y_sec: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (torch.log1p(y_sec) - mean) / max(std, 1e-6)


def _invert_log_normalise(z: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.expm1(z * std + mean).clamp(min=0)


@torch.no_grad()
def evaluate(model, loader, device, log_mean, log_std):
    model.eval()
    all_true, all_pred = [], []

    for batch in loader:
        for graph in batch:
            graph = graph.to(device)
            pred_z, mask = model(graph)
            if mask.sum() == 0:
                continue
            y_raw  = graph["vehicle"].y_raw[mask]
            pred_s = _invert_log_normalise(pred_z[mask], log_mean, log_std)
            all_true.extend(y_raw.cpu().numpy())
            all_pred.extend(pred_s.cpu().numpy())

    y_t = np.array(all_true, dtype=np.float64)
    y_p = np.array(all_pred, dtype=np.float64)
    if len(y_t) == 0:
        return {"mae_s": float("nan"), "rmse_s": float("nan"), "mape_pct": float("nan"), "n": 0}
    return {
        "mae_s":    round(float(mae(y_t,  y_p)), 4),
        "rmse_s":   round(float(rmse(y_t, y_p)), 4),
        "mape_pct": round(float(mape(y_t, y_p)), 4),
        "n":        len(y_t),
    }


def main():
    args   = _parse_args()
    cfg    = VARIANTS[args.variant]
    if args.epochs:   cfg.epochs   = args.epochs
    if args.patience: cfg.patience = args.patience
    _set_seed(args.seed)

    dataset_root = DATASET_PATHS[args.dataset]
    out_dir = args.output_dir or (
        _REPO_ROOT / "results" / f"dstra_{args.dataset}_{args.variant}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Variant: {args.variant}  |  Seed: {args.seed}")

    # Static index (shared across all splits)
    print("Building static index …")
    si = build_static_index(dataset_root / "train" / "static.json")
    print(f"  Junctions: {si['n_junctions']:,}  Road edges: {si['n_road_edges']:,}")

    # Datasets
    print("Building datasets …")
    t0 = time.time()
    train_ds = SlidingWindowDataset(dataset_root / "train", si, cfg.window_size, cfg.use_dynamic_edges)
    val_ds   = SlidingWindowDataset(dataset_root / "val",   si, cfg.window_size, cfg.use_dynamic_edges)
    test_ds  = SlidingWindowDataset(dataset_root / "test",  si, cfg.window_size, cfg.use_dynamic_edges)
    print(f"  Windows — train {len(train_ds):,}  val {len(val_ds):,}  test {len(test_ds):,}  ({time.time()-t0:.0f}s)")

    # Label statistics from training set
    print("Computing label statistics …")
    log_mean, log_std = compute_label_stats(train_ds)
    print(f"  log-ETA  mean={log_mean:.4f}  std={log_std:.4f}")
    train_ds.set_label_stats(log_mean, log_std)
    val_ds.set_label_stats(log_mean, log_std)
    test_ds.set_label_stats(log_mean, log_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_windows,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_windows)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_windows)

    # Model
    model  = DSTRAGNN(cfg, n_road_edges=si["n_road_edges"]).to(device)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DSTRA-GNN ({args.variant}) params: {n_par:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Linear warmup + cosine annealing
    total_steps   = cfg.epochs * len(train_loader)
    warmup_steps  = min(cfg.warmup_steps, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.HuberLoss(delta=1.0)

    best_val_mae, best_state, patience_cnt = float("inf"), None, 0
    global_step = 0

    print(f"\nTraining for {cfg.epochs} epochs (patience={cfg.patience}) …")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_veh = 0.0, 0

        for batch in train_loader:
            for graph in batch:
                graph = graph.to(device)
                pred_z, mask = model(graph)
                if mask.sum() == 0:
                    continue
                y_raw  = graph["vehicle"].y_raw[mask].to(device)
                y_z    = _log_normalise(y_raw, log_mean, log_std)
                loss   = loss_fn(pred_z[mask], y_z)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                total_loss += loss.item() * mask.sum().item()
                n_veh      += mask.sum().item()

        avg_loss = total_loss / max(n_veh, 1)
        val_m    = evaluate(model, val_loader, device, log_mean, log_std)
        improved = val_m["mae_s"] < best_val_mae

        if improved:
            best_val_mae = val_m["mae_s"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(f"  ep {epoch:3d}  loss {avg_loss:.4f}  val MAE {val_m['mae_s']:.2f}s"
              f"  n={val_m['n']:,}{' *' if improved else ''}")

        if patience_cnt >= cfg.patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    # Final evaluation on best model
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")

    val_metrics  = {**evaluate(model, val_loader,  device, log_mean, log_std), "split": "val"}
    test_metrics = {**evaluate(model, test_loader, device, log_mean, log_std), "split": "test"}

    results = {
        "val":     val_metrics,
        "test":    test_metrics,
        "variant": args.variant,
        "dataset": args.dataset,
        "seed":    args.seed,
        "log_mean": log_mean,
        "log_std":  log_std,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 62)
    print(f"  {args.variant} on {args.dataset.upper()} (seed {args.seed})")
    print(f"{'Split':<6}  {'N':>8}  {'MAE (s)':>9}  {'RMSE (s)':>10}  {'MAPE (%)':>9}")
    print("-" * 62)
    for m in (val_metrics, test_metrics):
        print(f"{m['split']:<6}  {m['n']:>8,}  {m['mae_s']:>9.2f}  {m['rmse_s']:>10.2f}  {m['mape_pct']:>9.2f}")
    print("=" * 62)
    print(f"\nResults → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
