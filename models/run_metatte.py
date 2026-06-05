"""Train and evaluate MetaTTE-GRU on Porto-T (Porto-only, no meta-learning).

Usage:
    .venv-deeptte/bin/python models/run_metatte.py
    .venv-deeptte/bin/python models/run_metatte.py --seed 43 --output-dir results/metatte_s43
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from models.baselines.deeptte import PortoTrajDataset, collate_fn
from models.baselines.metatte import MetaTTEGRU
from models.utils.metrics import mae, mape, rmse

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SPLITS_DIR = _REPO_ROOT / "datasets/real/porto_taxi_traj/processed/splits"
_OUTPUT_DIR = _REPO_ROOT / "results/metatte"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir",  type=Path,  default=_SPLITS_DIR)
    p.add_argument("--output-dir",  type=Path,  default=_OUTPUT_DIR)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--device",      type=str,   default=None)
    return p.parse_args()


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _build_taxi_vocab(df) -> dict:
    return {t: i+1 for i, t in enumerate(sorted(df["taxi_id"].unique().to_list()))}


@torch.no_grad()
def _evaluate(model, loader, device, label_mean, label_std):
    model.eval()
    y_true, y_pred = [], []
    for coords, cdist, attr_ids, dist_n, labels, lengths, raw_labels in loader:
        coords   = coords.to(device); cdist    = cdist.to(device)
        attr_ids = attr_ids.to(device); dist_n = dist_n.to(device)
        lengths  = lengths.to(device)
        preds = model(coords, cdist, attr_ids, dist_n, lengths).cpu().numpy()
        preds = preds * label_std + label_mean
        y_pred.extend(preds); y_true.extend(raw_labels.numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return {
        "mae_s":    round(float(mae(y_true, y_pred)), 4),
        "rmse_s":   round(float(rmse(y_true, y_pred)), 4),
        "mape_pct": round(float(mape(y_true, y_pred)), 4),
        "n_trips":  len(y_true),
    }


def main():
    args = _parse_args()
    _set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {args.seed}")

    print("Loading splits …")
    train_df = pl.read_parquet(args.splits_dir / "train.parquet").filter(pl.col("polyline") != "[]")
    val_df   = pl.read_parquet(args.splits_dir / "val.parquet").filter(pl.col("polyline") != "[]")
    test_df  = pl.read_parquet(args.splits_dir / "test.parquet").filter(pl.col("polyline") != "[]")
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    taxi_vocab = _build_taxi_vocab(train_df)

    print("Building datasets …")
    t0 = time.time()
    train_ds = PortoTrajDataset(train_df, taxi_vocab)
    val_ds   = PortoTrajDataset(val_df,   taxi_vocab)
    test_ds  = PortoTrajDataset(test_df,  taxi_vocab)
    print(f"  done in {time.time()-t0:.1f}s")

    label_mean = float(np.mean([s["label"] for s in train_ds.samples]))
    label_std  = float(np.std( [s["label"] for s in train_ds.samples]))
    print(f"  label stats: mean={label_mean:.1f}s  std={label_std:.1f}s")
    for ds in (train_ds, val_ds, test_ds):
        ds.set_label_stats(label_mean, label_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    model = MetaTTEGRU(n_taxis=len(taxi_vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MetaTTE-GRU params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.HuberLoss(delta=1.0)

    best_val_mae, best_state, patience_cnt = float("inf"), None, 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}) …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for coords, cdist, attr_ids, dist_n, labels, lengths, _ in train_loader:
            coords   = coords.to(device); cdist    = cdist.to(device)
            attr_ids = attr_ids.to(device); dist_n = dist_n.to(device)
            labels   = labels.to(device); lengths  = lengths.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(coords, cdist, attr_ids, dist_n, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        val_m = _evaluate(model, val_loader, device, label_mean, label_std)
        improved = val_m["mae_s"] < best_val_mae
        if improved:
            best_val_mae = val_m["mae_s"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(f"  ep {epoch:3d}  loss {total_loss/max(n_batches,1):.3f}"
              f"  val MAE {val_m['mae_s']:.2f}s{' *' if improved else ''}")

        if patience_cnt >= args.patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, args.output_dir / "model.pt")

    val_metrics  = {**_evaluate(model, val_loader,  device, label_mean, label_std), "split": "val"}
    test_metrics = {**_evaluate(model, test_loader, device, label_mean, label_std), "split": "test"}

    results = {"val": val_metrics, "test": test_metrics, "config": vars(args)}
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print(f"{'Split':<6}  {'N':>8}  {'MAE (s)':>9}  {'RMSE (s)':>10}  {'MAPE (%)':>9}")
    print("-" * 60)
    for m in (val_metrics, test_metrics):
        print(f"{m['split']:<6}  {m['n_trips']:>8,}  "
              f"{m['mae_s']:>9.2f}  {m['rmse_s']:>10.2f}  {m['mape_pct']:>9.2f}")
    print("=" * 60)
    print(f"\nResults → {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
