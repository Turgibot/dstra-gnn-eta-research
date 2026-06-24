"""Train and evaluate DSTRA-GNN on SUMO or Porto-G.

Usage:
    # Full model (A6) on SUMO:
    python models/run_dstra_gnn.py --dataset sumo --variant A6

    # All ablations on Porto-G:
    for V in A1 A2 A3 A4 A5 A6; do
        python models/run_dstra_gnn.py --dataset porto --variant $V
    done

    # Specify seed:
    python models/run_dstra_gnn.py --dataset sumo --variant A6 --seed 43
"""
import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dstra_gnn.config import VARIANTS, DSTRAConfig
from models.dstra_gnn.data_utils import build_static_index
from models.dstra_gnn.dataset import (
    SlidingWindowDataset,
    collate_windows,
    compute_demand_stats,
    compute_label_stats,
    compute_trip_durations,
    trip_ids_in_range,
)
from models.dstra_gnn.model import DSTRAGNN
from models.utils.metrics import mae, mape, rmse

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATHS = {
    "sumo":       _REPO_ROOT / "datasets/simulated/sumo",
    "sumo_small": _REPO_ROOT / "datasets/simulated/sumo_small",
    "porto":      _REPO_ROOT / "datasets/real/porto_taxi_graph",
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=list(DATASET_PATHS), default="sumo")
    p.add_argument("--variant",    choices=list(VARIANTS),    default="A6")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--epochs",     type=int,   default=None,  help="Override cfg epochs")
    p.add_argument("--patience",   type=int,   default=None)
    p.add_argument("--moe-experts", type=int,  default=None,  help="Override cfg moe_experts")
    p.add_argument("--moe-top-k",   type=int,  default=None,  help="Override cfg moe_top_k")
    p.add_argument("--output-dir", type=Path,  default=None)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Cap gradient steps per epoch (subsamples training data)")
    p.add_argument("--val-steps",  type=int,   default=None,
                   help="Cap val windows per epoch")
    p.add_argument("--max-train-windows", type=int, default=None,
                   help="Limit training set size for quick sanity checks")
    p.add_argument("--preload-train", action="store_true", default=False,
                   help="Force-preload train snapshots into RAM")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from checkpoint.pt in --output-dir")
    p.add_argument("--no-amp", action="store_true", default=False,
                   help="Disable automatic mixed precision")
    p.add_argument("--filter-trip-duration", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="Restrict training/validation/test to vehicles whose "
                        "whole trip's total duration (ETA at trip start) falls "
                        "in [LO, HI] seconds — i.e. only 'typical trips' matching "
                        "MetaTTE's Rule-1 population selection contribute to the "
                        "loss/metrics (every snapshot of an in-range trip is kept, "
                        "from trip start down to arrival). The model architecture "
                        "and per-snapshot mechanism are unchanged — only which "
                        "vehicles' predictions contribute is filtered (label "
                        "normalisation stats are recomputed on the filtered "
                        "population too). E.g. '--filter-trip-duration 315 945' "
                        "trains/evaluates on MetaTTE's Porto 'typical trip' "
                        "population for a population-matched comparison "
                        "(see METATTE_PORTO_RANGE).")
    p.add_argument("--filter-labels", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="Restrict training/validation/test to individual "
                        "(vehicle, snapshot) predictions whose own remaining-"
                        "ETA target falls in [LO, HI] seconds — independent of "
                        "which trip they belong to (contrast with "
                        "--filter-trip-duration, which keeps every snapshot of "
                        "an in-range trip regardless of that snapshot's own "
                        "remaining ETA). Snapshots/labels outside the range are "
                        "cleared from the loss/metrics; label normalisation "
                        "stats are recomputed on the filtered population too. "
                        "Can be combined with --filter-trip-duration (the two "
                        "are AND'ed) for a population matched on both whole-"
                        "trip duration and per-snapshot remaining time. "
                        "E.g. '--filter-labels 315 945'.")
    return p.parse_args()


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _log_normalise(y_sec: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (torch.log1p(y_sec) - mean) / max(std, 1e-6)


def _invert_log_normalise(z: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.expm1(z * std + mean).clamp(min=0)


# MetaTTE's Porto "typical trip" range (Rule 1 filter — see
# Papers/Springer/METATTE_EXPERIMENT_PROTOCOL.md §3): travel time in [315s, 945s],
# derived from the CDF 10%-80% band of their Porto trip-duration distribution.
METATTE_PORTO_RANGE = (315.0, 945.0)


@torch.no_grad()
def evaluate(model, loader, device, log_mean, log_std,
             use_amp: bool = False, max_windows: int = None, return_raw: bool = False,
             show_progress: bool = False):
    """Evaluate and report two views of the same predictions:
      - "full": every labeled (vehicle, snapshot) prediction — DSTRA-GNN's
        native per-snapshot remaining-ETA task.
      - "mtte_trip": one prediction per trip — each vehicle's chronologically-
        first labeled prediction, where remaining-ETA == total trip duration —
        restricted to trips whose total duration falls in METATTE_PORTO_RANGE.
        This reproduces MetaTTE's evaluation methodology exactly (one
        total-duration prediction per "typical" trip), so mape_mtte_pct is
        directly comparable to their reported Porto MAPE.

    Requires the loader to be chronologically ordered (shuffle=False) so that
    each vehicle ID's first appearance corresponds to its trip start.

    If return_raw, also includes the raw "y_true"/"y_pred" arrays (every
    per-snapshot sample) so callers can dump them to disk and recompute
    arbitrary filtered metrics offline later without re-running the
    (expensive) forward pass.
    """
    model.eval()
    all_true, all_pred = [], []
    trip_true, trip_pred = [], []
    seen_vehicle_ids = set()
    n_windows = 0

    total = min(len(loader), max_windows) if max_windows else len(loader)
    pbar = tqdm(total=total, desc="eval", unit="win", disable=not show_progress)

    for batch in loader:
        for graph in batch:
            graph = graph.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_z, mask, _imp, _load = model(graph)
            if mask.sum() == 0:
                continue
            y_raw  = graph["vehicle"].y_raw[mask]
            pred_s = _invert_log_normalise(pred_z[mask].float(), log_mean, log_std)
            y_np = y_raw.cpu().numpy()
            p_np = pred_s.cpu().numpy()
            all_true.extend(y_np)
            all_pred.extend(p_np)

            masked_ids = [vid for vid, m in zip(graph["vehicle"].v_ids, mask.cpu().numpy()) if m]
            for vid, yt, yp in zip(masked_ids, y_np, p_np):
                if vid not in seen_vehicle_ids:
                    seen_vehicle_ids.add(vid)
                    trip_true.append(yt)
                    trip_pred.append(yp)
            n_windows += 1
            pbar.update(1)
        if max_windows and n_windows >= max_windows:
            break
    pbar.close()

    y_t = np.array(all_true, dtype=np.float64)
    y_p = np.array(all_pred, dtype=np.float64)

    def _empty():
        d = {"mae_s": float("nan"), "rmse_s": float("nan"), "mape_pct": float("nan"), "n": 0,
             "mae_mtte_s": float("nan"), "rmse_mtte_s": float("nan"),
             "mape_mtte_pct": float("nan"), "n_mtte": 0}
        if return_raw:
            d.update({"y_true": y_t, "y_pred": y_p})
        return d

    if len(y_t) == 0:
        return _empty()

    out = {
        "mae_s":    round(float(mae(y_t,  y_p)), 4),
        "rmse_s":   round(float(rmse(y_t, y_p)), 4),
        "mape_pct": round(float(mape(y_t, y_p)), 4),
        "n":        len(y_t),
    }

    # "mtte_trip": one (first-snapshot) prediction per trip, restricted to
    # trips whose total duration (== remaining ETA at trip start) is in
    # MetaTTE's "typical trip" range — exactly their evaluation population.
    tt = np.array(trip_true, dtype=np.float64)
    tp = np.array(trip_pred, dtype=np.float64)
    lo, hi = METATTE_PORTO_RANGE
    in_range = (tt >= lo) & (tt <= hi)
    tt_r, tp_r = tt[in_range], tp[in_range]
    if in_range.any():
        out.update({
            "mae_mtte_s":    round(float(mae(tt_r,  tp_r)), 4),
            "rmse_mtte_s":   round(float(rmse(tt_r, tp_r)), 4),
            "mape_mtte_pct": round(float(mape(tt_r, tp_r)), 4),
            "n_mtte":        int(in_range.sum()),
        })
    else:
        out.update({"mae_mtte_s": float("nan"), "rmse_mtte_s": float("nan"),
                    "mape_mtte_pct": float("nan"), "n_mtte": 0})

    if return_raw:
        out.update({"y_true": y_t, "y_pred": y_p})
    return out


def main():
    args = _parse_args()
    cfg  = VARIANTS[args.variant]
    if args.epochs:     cfg.epochs      = args.epochs
    if args.patience:   cfg.patience    = args.patience
    if args.moe_experts: cfg.moe_experts = args.moe_experts
    if args.moe_top_k:   cfg.moe_top_k   = args.moe_top_k
    _set_seed(args.seed)

    trip_range  = tuple(args.filter_trip_duration) if args.filter_trip_duration else None
    label_range = tuple(args.filter_labels)        if args.filter_labels        else None

    dataset_root = DATASET_PATHS[args.dataset]
    dir_name = f"dstra_{args.dataset}_{args.variant}_s{args.seed}"
    if trip_range:
        dir_name += f"_trip{int(trip_range[0])}-{int(trip_range[1])}"
    if label_range:
        dir_name += f"_lbl{int(label_range[0])}-{int(label_range[1])}"
    out_dir = args.output_dir or (_REPO_ROOT / "results" / dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device(args.device if args.device else
                           ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp

    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Variant: {args.variant}  |  Seed: {args.seed}")
    if device.type == "cuda":
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB)  |  AMP: {use_amp}")

    print("Building static index …")
    si = build_static_index(dataset_root / "train" / "static.json")
    print(f"  Junctions: {si['n_junctions']:,}  Road edges: {si['n_road_edges']:,}"
          f"  j_in: {si['j_in']}  v_in: {si['v_in']}")

    print("Building datasets …")
    t0 = time.time()
    train_ds = SlidingWindowDataset(dataset_root / "train", si, cfg.window_size, cfg.use_dynamic_edges,
                                    max_windows=args.max_train_windows,
                                    force_preload=args.preload_train)
    val_ds   = SlidingWindowDataset(dataset_root / "val",   si, cfg.window_size, cfg.use_dynamic_edges)
    test_ds  = SlidingWindowDataset(dataset_root / "test",  si, cfg.window_size, cfg.use_dynamic_edges)
    print(f"  Windows — train {len(train_ds):,}  val {len(val_ds):,}  test {len(test_ds):,}  ({time.time()-t0:.0f}s)")

    # Trip-duration population filter (optional): restrict which vehicles'
    # predictions contribute to the loss/metrics to those whose whole trip's
    # total duration falls in a target range — i.e. only "typical trips"
    # matching MetaTTE's Rule-1 population selection. Every snapshot of an
    # in-range trip is kept (from trip start down to arrival); the graph
    # structure and per-snapshot mechanism are untouched — only labeled_mask
    # is intersected with trip membership (see set_trip_filter).
    train_filter_ids = None
    if trip_range:
        lo, hi = trip_range
        print(f"Applying trip-duration population filter [{lo:.0f}, {hi:.0f}]s "
              f"to train/val/test ('typical trips' only — every snapshot of an "
              f"in-range trip is kept):")
        for split_name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            t1 = time.time()
            durations    = compute_trip_durations(ds)
            in_range_ids = trip_ids_in_range(durations, lo, hi)
            ds.set_trip_filter(in_range_ids)
            print(f"  {split_name}: {len(in_range_ids):,}/{len(durations):,} vehicles "
                  f"({100.0*len(in_range_ids)/max(len(durations),1):.1f}%) belong to "
                  f"'typical trips'  ({time.time()-t1:.0f}s)")
        train_filter_ids = train_ds.trip_filter_ids

    # Per-snapshot label-range filter (optional, independent of and AND'able
    # with the trip-duration filter above): clears individual (vehicle,
    # snapshot) predictions whose own remaining-ETA target falls outside the
    # range — so e.g. a 20-minute trip's early snapshots (remaining ETA still
    # > range) are dropped even though the trip itself may be in-range under
    # --filter-trip-duration (and vice versa).
    if label_range:
        lo, hi = label_range
        print(f"Applying per-snapshot label-range filter [{lo:.0f}, {hi:.0f}]s "
              f"to train/val/test (only predictions whose own remaining-ETA "
              f"target falls in range contribute to loss/metrics):")
        for ds in (train_ds, val_ds, test_ds):
            ds.set_label_filter(label_range)

    # Label statistics
    print("Computing label statistics …")
    log_mean, log_std = compute_label_stats(train_ds, vehicle_filter=train_filter_ids,
                                             label_range=label_range)
    print(f"  log-ETA  mean={log_mean:.4f}  std={log_std:.4f}")
    for ds in (train_ds, val_ds, test_ds):
        ds.set_label_stats(log_mean, log_std)

    # Demand / occupancy statistics (for edge feature normalisation)
    print("Computing demand statistics …")
    demand_stats = compute_demand_stats(train_ds)
    print(f"  demand  mean={demand_stats['demand_mean']:.4f}  std={demand_stats['demand_std']:.4f}"
          f"  occ_max={demand_stats['occ_max']:.1f}")
    for ds in (train_ds, val_ds, test_ds):
        ds.set_demand_stats(demand_stats)

    n_workers  = 0 if train_ds._cache is not None else args.workers
    if n_workers != args.workers:
        print(f"  DataLoader workers: {args.workers} → 0 (data is preloaded)")
    val_workers = 0 if val_ds._cache is not None else min(n_workers, 2)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=n_workers, collate_fn=collate_windows,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=val_workers, collate_fn=collate_windows)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=val_workers, collate_fn=collate_windows)

    model = DSTRAGNN(cfg, n_road_edges=si["n_road_edges"],
                     j_in=si["j_in"], v_in=si["v_in"]).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DSTRA-GNN ({args.variant}) params: {n_par:,}")

    spe_est  = args.steps_per_epoch or len(train_loader)
    secs_est = cfg.epochs * spe_est * 0.18
    print(f"  Est. max training time: {secs_est/3600:.1f} h "
          f"({cfg.epochs} ep × {spe_est:,} steps × ~0.18 s)"
          f"{'  ← use --steps-per-epoch to cap' if secs_est > 7200 else ''}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Linear warmup (epoch-based) then cosine annealing
    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 0.2)   # start at 20% lr
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_mae, best_state, patience_cnt = float("inf"), None, 0
    global_step = 0
    start_epoch = 1

    ckpt_path = out_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt.get("global_step", 0)
        best_val_mae = ckpt["best_val_mae"]
        best_state   = ckpt["model_state"]
        patience_cnt = ckpt.get("patience_cnt", 0)
        print(f"  Resumed from epoch {ckpt['epoch']}  "
              f"best val MAE {best_val_mae:.2f}s  patience {patience_cnt}/{cfg.patience}")

    spe_msg = f", {steps_per_epoch:,} steps/epoch" if args.steps_per_epoch else ""
    print(f"\nTraining for {cfg.epochs} epochs (patience={cfg.patience}{spe_msg}) …")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss, n_veh = 0.0, 0
        epoch_t0 = time.time()

        batch_iter = itertools.islice(train_loader, steps_per_epoch)
        for batch in batch_iter:
            for graph in batch:
                graph = graph.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_z, mask, importance, load = model(graph)
                    if mask.sum() == 0:
                        continue

                    y_raw  = graph["vehicle"].y_raw[mask].to(device)
                    y_z    = _log_normalise(y_raw, log_mean, log_std)

                    # MAE regression loss in log-normalised space
                    reg_loss = F.l1_loss(pred_z[mask], y_z)

                    # MoE load-balancing loss (Switch Transformer)
                    lb_loss  = (cfg.moe_experts * (importance * load).sum())

                    # Router KL divergence from uniform (entropy regularisation)
                    kl_loss  = (importance * torch.log(importance * cfg.moe_experts + 1e-8)).sum()

                    loss = reg_loss + cfg.lb_weight * lb_loss + cfg.entropy_weight * kl_loss

                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                global_step += 1
                total_loss  += reg_loss.item() * mask.sum().item()
                n_veh       += mask.sum().item()

        avg_loss  = total_loss / max(n_veh, 1)
        val_m     = evaluate(model, val_loader, device, log_mean, log_std,
                             use_amp, max_windows=args.val_steps)
        improved  = val_m["mae_s"] < best_val_mae
        epoch_sec = time.time() - epoch_t0

        if improved:
            best_val_mae = val_m["mae_s"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            torch.save({
                "epoch":           epoch,
                "global_step":     global_step,
                "model_state":     best_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict() if scaler else None,
                "best_val_mae":    best_val_mae,
                "patience_cnt":    0,
                "log_mean":        log_mean,
                "log_std":         log_std,
                "demand_stats":    demand_stats,
            }, ckpt_path)
        else:
            patience_cnt += 1

        print(f"  ep {epoch:3d}  loss {avg_loss:.4f}  val MAE {val_m['mae_s']:.2f}s"
              f"  n={val_m['n']:,}  ({epoch_sec:.0f}s){' *' if improved else ''}")

        if patience_cnt >= cfg.patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    # Final evaluation on best model
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")

    print("\nRunning final evaluation …")
    val_metrics  = {**evaluate(model, val_loader,  device, log_mean, log_std,
                               use_amp, max_windows=args.val_steps), "split": "val"}
    test_metrics = {**evaluate(model, test_loader, device, log_mean, log_std,
                               use_amp, max_windows=args.val_steps), "split": "test"}

    results = {
        "val":          val_metrics,
        "test":         test_metrics,
        "variant":      args.variant,
        "dataset":      args.dataset,
        "seed":         args.seed,
        "log_mean":     log_mean,
        "log_std":      log_std,
        "demand_stats": demand_stats,
        "filter_trip_duration_range": list(trip_range) if trip_range else None,
        "filter_labels_range": list(label_range) if label_range else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 100)
    print(f"  {args.variant} on {args.dataset.upper()} (seed {args.seed})")
    print(f"{'Split':<6}  {'N':>8}  {'MAE (s)':>9}  {'RMSE (s)':>10}  {'MAPE (%)':>9}"
          f"  |  {'N trips (mtte)':>14}  {'MAE-mtte (s)':>12}  {'RMSE-mtte (s)':>13}  {'MAPE-mtte (%)':>13}")
    print("-" * 100)
    for m in (val_metrics, test_metrics):
        print(f"{m['split']:<6}  {m['n']:>8,}  {m['mae_s']:>9.2f}  {m['rmse_s']:>10.2f}  {m['mape_pct']:>9.2f}"
              f"  |  {m['n_mtte']:>14,}  {m['mae_mtte_s']:>12.2f}  {m['rmse_mtte_s']:>13.2f}  {m['mape_mtte_pct']:>13.2f}")
    print("=" * 100)
    print(f"\nResults → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
