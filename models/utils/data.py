import datetime as dt
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import polars as pl


# ── Geo helpers ──────────────────────────────────────────────────────────────

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bearing_rad(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlam = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlam) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlam)
    return math.atan2(x, y)


def load_split(splits_dir: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(splits_dir / f"{split}.parquet")


def extract_features(
    df: pl.DataFrame,
    grid_resolution: float = 0.01,
    n_time_slots: int = 24,
) -> pl.DataFrame:
    """Add OD grid-cell columns and time-slot column to df.

    Parses polyline once per row to extract origin (first point) and
    destination (last point) coordinates, then floors each into a
    grid_resolution-degree cell. Time slot is derived from Unix timestamp.
    """
    slot_seconds = 86400 // n_time_slots

    df = df.filter(pl.col("polyline") != "[]")
    parsed = [json.loads(p) for p in df["polyline"].to_list()]
    origin_lon = pl.Series("origin_lon", [pts[0][0] for pts in parsed], dtype=pl.Float64)
    origin_lat = pl.Series("origin_lat", [pts[0][1] for pts in parsed], dtype=pl.Float64)
    dest_lon   = pl.Series("dest_lon",   [pts[-1][0] for pts in parsed], dtype=pl.Float64)
    dest_lat   = pl.Series("dest_lat",   [pts[-1][1] for pts in parsed], dtype=pl.Float64)

    res = grid_resolution
    origin_cx = pl.Series("origin_cx", np.floor(origin_lon.to_numpy() / res).astype("int32"))
    origin_cy = pl.Series("origin_cy", np.floor(origin_lat.to_numpy() / res).astype("int32"))
    dest_cx   = pl.Series("dest_cx",   np.floor(dest_lon.to_numpy()   / res).astype("int32"))
    dest_cy   = pl.Series("dest_cy",   np.floor(dest_lat.to_numpy()   / res).astype("int32"))

    return df.with_columns([
        origin_lon,
        origin_lat,
        dest_lon,
        dest_lat,
        origin_cx,
        origin_cy,
        dest_cx,
        dest_cy,
        ((pl.col("timestamp") % 86400) // slot_seconds).cast(pl.Int32).alias("hour"),
    ])


# ── LR / GBM feature matrix ──────────────────────────────────────────────────

LR_FEATURE_NAMES: List[str] = [
    "od_dist_m",
    "polyline_dist_m",
    "dist_ratio",
    "bearing_sin",
    "bearing_cos",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
    "log_od_dist_m",
]


def extract_lr_features(df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build the feature matrix used by LR (and GBM) baselines.

    Caller is responsible for pre-filtering empty polylines.
    Features:
      od_dist_m       straight-line O→D haversine distance
      polyline_dist_m total GPS-trace distance (from parquet)
      dist_ratio      polyline / od distance (tortuosity)
      bearing_sin/cos cyclical direction O→D
      hour_sin/cos    cyclical hour-of-day (UTC)
      dow_sin/cos     cyclical day-of-week (0=Mon)
      is_weekend      Saturday or Sunday flag
      is_holiday      day_type == 'C' flag
      log_od_dist_m   log1p(od_dist_m)
    """
    parsed = [json.loads(p) for p in df["polyline"].to_list()]
    origins = [(pts[0][0], pts[0][1]) for pts in parsed]   # (lon, lat)
    dests   = [(pts[-1][0], pts[-1][1]) for pts in parsed]

    od_dist = np.array([
        _haversine_m(o[0], o[1], d[0], d[1]) for o, d in zip(origins, dests)
    ])
    poly_dist  = df["distance_m"].to_numpy()
    dist_ratio = poly_dist / np.maximum(od_dist, 1.0)

    bearings    = np.array([_bearing_rad(o[0], o[1], d[0], d[1]) for o, d in zip(origins, dests)])
    bearing_sin = np.sin(bearings)
    bearing_cos = np.cos(bearings)

    # Unix epoch 0 = Thursday; +3 shifts so 0 = Monday
    ts      = df["timestamp"].to_numpy()
    hours   = (ts % 86400) // 3600
    dow     = (ts // 86400 + 3) % 7
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dow_sin  = np.sin(2 * np.pi * dow / 7)
    dow_cos  = np.cos(2 * np.pi * dow / 7)

    is_weekend = (dow >= 5).astype(np.float32)
    is_holiday = np.array([1.0 if dt == "C" else 0.0 for dt in df["day_type"].to_list()], dtype=np.float32)

    X = np.column_stack([
        od_dist, poly_dist, dist_ratio,
        bearing_sin, bearing_cos,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        is_weekend, is_holiday,
        np.log1p(od_dist),
    ])
    return X, LR_FEATURE_NAMES


# ── GBM feature matrix ───────────────────────────────────────────────────────

_CALL_TYPE_MAP = {"A": 0, "B": 1, "C": 2}

GBM_FEATURE_NAMES: List[str] = [
    "od_dist_m",
    "polyline_dist_m",
    "dist_ratio",
    "bearing_sin",
    "bearing_cos",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "call_type_enc",
    "origin_cx",
    "origin_cy",
    "dest_cx",
    "dest_cy",
]


def extract_gbm_features(
    df: pl.DataFrame,
    grid_resolution: float = 0.01,
) -> Tuple[np.ndarray, List[str]]:
    """Build the feature matrix for the GBM baseline.

    Caller is responsible for pre-filtering empty polylines.
    Uses raw integer hour/dow/month instead of sin/cos — trees learn
    arbitrary non-linear splits without needing cyclical encodings.
    Adds call_type and OD grid cells as categorical integers.
    Never includes n_points (= travel_time_s / 15, pure leakage).
    """
    parsed = [json.loads(p) for p in df["polyline"].to_list()]
    origins = [(pts[0][0], pts[0][1]) for pts in parsed]
    dests   = [(pts[-1][0], pts[-1][1]) for pts in parsed]

    od_dist    = np.array([_haversine_m(o[0], o[1], d[0], d[1]) for o, d in zip(origins, dests)])
    poly_dist  = df["distance_m"].to_numpy()
    dist_ratio = poly_dist / np.maximum(od_dist, 1.0)

    bearings    = np.array([_bearing_rad(o[0], o[1], d[0], d[1]) for o, d in zip(origins, dests)])
    bearing_sin = np.sin(bearings)
    bearing_cos = np.cos(bearings)

    ts    = df["timestamp"].to_numpy()
    hours = (ts % 86400) // 3600                   # 0–23
    dow   = (ts // 86400 + 3) % 7                  # 0=Mon … 6=Sun
    month = np.array(
        [dt.date.fromtimestamp(int(t)).month for t in ts], dtype=np.int32
    )                                               # 1–12

    is_weekend    = (dow >= 5).astype(np.float32)
    day_types     = df["day_type"].to_list()
    is_holiday    = np.array([1.0 if d == "C" else 0.0 for d in day_types], dtype=np.float32)
    call_type_enc = np.array(
        [_CALL_TYPE_MAP.get(c, 0) for c in df["call_type"].to_list()], dtype=np.int32
    )

    res       = grid_resolution
    origin_cx = np.floor(np.array([o[0] for o in origins]) / res).astype(np.int32)
    origin_cy = np.floor(np.array([o[1] for o in origins]) / res).astype(np.int32)
    dest_cx   = np.floor(np.array([d[0] for d in dests])   / res).astype(np.int32)
    dest_cy   = np.floor(np.array([d[1] for d in dests])   / res).astype(np.int32)

    X = np.column_stack([
        od_dist, poly_dist, dist_ratio,
        bearing_sin, bearing_cos,
        hours, dow, month,
        is_weekend, is_holiday, call_type_enc,
        origin_cx, origin_cy, dest_cx, dest_cy,
    ])
    return X, GBM_FEATURE_NAMES
