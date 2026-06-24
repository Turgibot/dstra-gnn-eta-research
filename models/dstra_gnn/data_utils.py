"""Snapshot loading, feature extraction, and graph construction for DSTRA-GNN."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Slimming: fields retained in RAM cache (~90% memory reduction vs full JSON)
_KEEP_NODE = frozenset({
    "id", "vehicle_type",
    "speed", "acceleration",
    "current_x", "current_y",
    "destination_x", "destination_y",
    "route_length", "route_length_left",
    "current_zone", "current_edge",
    "route_left",
})
_KEEP_RE = frozenset({"id", "avg_speed", "density", "edge_demand", "vehicles_on_road"})
_KEEP_DE = frozenset({"edge_type", "from", "to"})

# Fixed vocabulary orders (must be consistent across train/val/test)
_VTYPES = ["bus", "passenger", "truck"]           # vehicle type one-hot (3)
_JTYPES = ["priority", "traffic_light", "internal"]  # junction type one-hot (3)

_SECONDS_PER_DAY  = 86400.0
_SECONDS_PER_WEEK = 604800.0


# ── Static index ──────────────────────────────────────────────────────────────

def build_static_index(static_path: Path) -> dict:
    """Load static.json and return lookup tables + pre-computed feature arrays.

    Junction features  (N_j, j_in):
        x_norm, y_norm,
        is_priority, is_traffic_light, is_internal,
        degree_norm,
        zone_oh × n_zones

    Road edge static features (N_re, 3): length_norm, speed_norm, lanes_norm
    """
    static = json.loads(static_path.read_text())

    junction_ids   = [j["id"] for j in static["junctions"]]
    j_id2idx       = {jid: i for i, jid in enumerate(junction_ids)}
    road_edge_ids  = [e["id"] for e in static["road_edges"]]
    re_id2idx      = {eid: i for i, eid in enumerate(road_edge_ids)}
    road_edge_info = {e["id"]: e for e in static["road_edges"]}

    # Zone vocabulary: enumerate from junctions (sorted → reproducible)
    zones    = sorted(set(j.get("zone", "") for j in static["junctions"]) - {""})
    zone2idx = {z: i for i, z in enumerate(zones)}
    n_zones  = len(zones)

    # Junction features -------------------------------------------------------
    xs = np.array([j["x"] for j in static["junctions"]], dtype=np.float32)
    ys = np.array([j["y"] for j in static["junctions"]], dtype=np.float32)
    x_min   = float(xs.min()); x_scale = float(max(xs.max() - xs.min(), 1.0))
    y_min   = float(ys.min()); y_scale = float(max(ys.max() - ys.min(), 1.0))
    x_norm  = (xs - x_min) / x_scale
    y_norm  = (ys - y_min) / y_scale

    j_type_oh = np.zeros((len(junction_ids), len(_JTYPES)), dtype=np.float32)
    for i, j in enumerate(static["junctions"]):
        jt = j.get("type", "priority")
        if jt in _JTYPES:
            j_type_oh[i, _JTYPES.index(jt)] = 1.0

    degree = np.array(
        [len(j.get("incoming", [])) + len(j.get("outgoing", [])) for j in static["junctions"]],
        dtype=np.float32,
    )
    degree_norm = degree / max(degree.max(), 1.0)

    j_zone_oh = np.zeros((len(junction_ids), n_zones), dtype=np.float32)
    for i, j in enumerate(static["junctions"]):
        z = j.get("zone", "")
        if z in zone2idx:
            j_zone_oh[i, zone2idx[z]] = 1.0

    junction_feats = np.concatenate(
        [x_norm[:, None], y_norm[:, None], j_type_oh, degree_norm[:, None], j_zone_oh],
        axis=1,
    )  # (N_j, 2 + n_jtypes + 1 + n_zones)
    j_in = junction_feats.shape[1]

    # Road edge static features -----------------------------------------------
    max_len   = max(e["length"] for e in static["road_edges"])
    max_speed = max(e["speed"]  for e in static["road_edges"])
    re_static_feats = np.array(
        [[e["length"]    / max(max_len,   1.0),
          e["speed"]     / max(max_speed, 1.0),
          e["num_lanes"] / 3.0]
         for e in static["road_edges"]],
        dtype=np.float32,
    )  # (N_re, 3)

    re_src = np.array([j_id2idx.get(e["from"], 0) for e in static["road_edges"]], dtype=np.int64)
    re_dst = np.array([j_id2idx.get(e["to"],   0) for e in static["road_edges"]], dtype=np.int64)

    # v_in: vehicle feature dimension = 18 + n_zones
    # (3 vtype + 1 speed + 1 accel + 2 cur_xy + 2 dest_xy + 1 progress +
    #  1 rl_norm + 4 sin/cos + 1 demand + 1 occ + 1 lanes + n_zones)
    v_in = 18 + n_zones

    return {
        "junction_ids":    junction_ids,
        "j_id2idx":        j_id2idx,
        "junction_feats":  junction_feats,      # (N_j, j_in)
        "j_in":            j_in,
        "v_in":            v_in,
        "road_edge_ids":   road_edge_ids,
        "re_id2idx":       re_id2idx,
        "road_edge_info":  road_edge_info,
        "re_static_feats": re_static_feats,     # (N_re, 3)
        "re_src":          re_src,
        "re_dst":          re_dst,
        "n_junctions":     len(junction_ids),
        "n_road_edges":    len(road_edge_ids),
        "max_len":         max_len,
        "max_speed":       max_speed,
        "x_min":           x_min,
        "x_scale":         x_scale,
        "y_min":           y_min,
        "y_scale":         y_scale,
        "zones":           zones,
        "zone2idx":        zone2idx,
        "n_zones":         n_zones,
    }


# ── Per-snapshot helpers ──────────────────────────────────────────────────────

def slim_snapshot(raw: dict, re_id2idx: Optional[Dict[str, int]] = None) -> dict:
    """Retain only fields used by graph construction (~90% memory reduction).

    Converts route_left strings → int indices when re_id2idx provided.
    Converts vehicles_on_road list → count int to avoid retaining lists.
    Preserves top-level step field for time encoding.
    """
    nodes = []
    for n in raw.get("nodes", []):
        slim_n = {k: v for k, v in n.items() if k in _KEEP_NODE}
        if re_id2idx is not None and "route_left" in slim_n:
            slim_n["route_left"] = [re_id2idx[e] for e in slim_n["route_left"]
                                    if e in re_id2idx]
        nodes.append(slim_n)

    road_edges = []
    for e in raw.get("road_edges_dynamic", []):
        slim_e = {k: v for k, v in e.items() if k in _KEEP_RE}
        if "vehicles_on_road" in slim_e:
            vor = slim_e["vehicles_on_road"]
            slim_e["vehicles_on_road"] = len(vor) if isinstance(vor, list) else int(vor)
        road_edges.append(slim_e)

    return {
        "step":               raw.get("step", 0),
        "nodes":              nodes,
        "road_edges_dynamic": road_edges,
        "dynamic_edges":      [{k: v for k, v in de.items() if k in _KEEP_DE}
                               for de in raw.get("dynamic_edges", [])],
    }


def load_snapshot_raw(path: Path) -> Optional[dict]:
    """Load a snapshot JSON; return None if empty or corrupt."""
    raw = path.read_text()
    if not raw.strip():
        return None
    return json.loads(raw)


def extract_road_dyn_features(
    snap: dict,
    si: dict,
    demand_stats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Return (N_re, 4) dynamic road-edge features: [avg_speed, density, demand_z, occ_norm]."""
    N_re      = si["n_road_edges"]
    re_id2idx = si["re_id2idx"]
    max_spd   = si["max_speed"]

    d_mean = demand_stats["demand_mean"] if demand_stats else 0.0
    d_std  = max(demand_stats["demand_std"], 1e-6) if demand_stats else 1.0
    o_max  = max(demand_stats.get("occ_max", 30.0), 1.0) if demand_stats else 30.0

    feats = np.zeros((N_re, 4), dtype=np.float32)
    for e in snap.get("road_edges_dynamic", []):
        idx = re_id2idx.get(e["id"])
        if idx is None:
            continue
        feats[idx, 0] = e["avg_speed"] / max(max_spd, 1.0)
        feats[idx, 1] = e["density"]
        feats[idx, 2] = (e["edge_demand"] - d_mean) / d_std
        occ = e.get("vehicles_on_road", 0)
        if isinstance(occ, list):
            occ = len(occ)
        feats[idx, 3] = min(float(occ), o_max) / o_max
    return feats


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_hetero_graph(
    snap:              dict,
    labels_map:        Dict[str, float],
    si:                dict,
    use_dynamic_edges: bool = True,
    demand_stats:      Optional[Dict[str, float]] = None,
) -> Optional[HeteroData]:
    """Build a PyG HeteroData from a single snapshot.

    Vehicle features (18 + n_zones):
        [0:3]       vehicle_type one-hot [bus, passenger, truck]
        [3]         speed_norm
        [4]         accel_norm
        [5]         current_x_norm
        [6]         current_y_norm
        [7]         destination_x_norm
        [8]         destination_y_norm
        [9]         trip_progress  (1 - route_length_left / route_length)
        [10]        route_length_left_norm
        [11]        sin_hour
        [12]        cos_hour
        [13]        sin_day
        [14]        cos_day
        [15]        current_edge_demand_z
        [16]        current_edge_occ_norm
        [17]        current_edge_lanes_norm
        [18:18+nz]  zone one-hot
    """
    j_id2idx       = si["j_id2idx"]
    re_id2idx      = si["re_id2idx"]
    max_spd        = si["max_speed"]
    x_min          = si["x_min"];  x_scale = si["x_scale"]
    y_min          = si["y_min"];  y_scale = si["y_scale"]
    zone2idx       = si["zone2idx"]
    n_zones        = si["n_zones"]
    road_edge_info = si["road_edge_info"]

    d_mean = demand_stats["demand_mean"] if demand_stats else 0.0
    d_std  = max(demand_stats["demand_std"], 1e-6) if demand_stats else 1.0
    o_max  = max(demand_stats.get("occ_max", 30.0), 1.0) if demand_stats else 30.0

    # Dynamic edge lookup keyed by edge ID
    edge_dyn_lut: Dict[str, dict] = {}
    for e in snap.get("road_edges_dynamic", []):
        occ = e.get("vehicles_on_road", 0)
        if isinstance(occ, list):
            occ = len(occ)
        edge_dyn_lut[e["id"]] = {"demand": e.get("edge_demand", 0.0), "occ": float(occ)}

    # Time encoding from snapshot step (seconds since simulation start)
    step     = snap.get("step", 0)
    hour_frac = (step % _SECONDS_PER_DAY) / _SECONDS_PER_DAY
    day_frac  = ((step // _SECONDS_PER_DAY) % 7) / 7.0
    sin_h = math.sin(2 * math.pi * hour_frac)
    cos_h = math.cos(2 * math.pi * hour_frac)
    sin_d = math.sin(2 * math.pi * day_frac)
    cos_d = math.cos(2 * math.pi * day_frac)

    # ── Vehicle nodes ──────────────────────────────────────────────────────
    nodes = snap.get("nodes", [])
    if not nodes:
        return None

    v_ids    = [n["id"] for n in nodes]
    v_id2idx = {vid: i for i, vid in enumerate(v_ids)}
    N_v      = len(nodes)

    max_rl = max((n.get("route_length_left", 0) for n in nodes), default=1.0)
    max_rl = max(float(max_rl), 1.0)

    n_vfeats = 18 + n_zones
    v_feats  = np.zeros((N_v, n_vfeats), dtype=np.float32)
    route_left_idx: List[List[int]] = []

    for i, n in enumerate(nodes):
        col = 0

        # vehicle type one-hot
        vt = n.get("vehicle_type", "passenger")
        if vt in _VTYPES:
            v_feats[i, col + _VTYPES.index(vt)] = 1.0
        col += len(_VTYPES)  # → 3

        v_feats[i, col]     = n.get("speed", 0.0) / max(max_spd, 1.0)
        v_feats[i, col + 1] = n.get("acceleration", 0.0) / 10.0
        col += 2  # → 5

        v_feats[i, col]     = (n.get("current_x", 0.0) - x_min) / x_scale
        v_feats[i, col + 1] = (n.get("current_y", 0.0) - y_min) / y_scale
        col += 2  # → 7

        v_feats[i, col]     = (n.get("destination_x", 0.0) - x_min) / x_scale
        v_feats[i, col + 1] = (n.get("destination_y", 0.0) - y_min) / y_scale
        col += 2  # → 9

        rl   = float(n.get("route_length_left", 0.0))
        rtot = max(float(n.get("route_length", rl)), rl, 1.0)
        v_feats[i, col]     = 1.0 - rl / rtot  # trip_progress
        v_feats[i, col + 1] = rl / max_rl       # rl_norm
        col += 2  # → 11

        v_feats[i, col]     = sin_h
        v_feats[i, col + 1] = cos_h
        v_feats[i, col + 2] = sin_d
        v_feats[i, col + 3] = cos_d
        col += 4  # → 15

        ce_id  = n.get("current_edge")
        ce_dyn = edge_dyn_lut.get(ce_id, {}) if ce_id else {}
        raw_d  = ce_dyn.get("demand", 0.0)
        raw_o  = ce_dyn.get("occ",    0.0)
        v_feats[i, col]     = (raw_d - d_mean) / d_std
        v_feats[i, col + 1] = min(raw_o, o_max) / o_max
        col += 2  # → 17

        ce_info = road_edge_info.get(ce_id) if ce_id else None
        v_feats[i, col] = ce_info["num_lanes"] / 3.0 if ce_info else 0.0
        col += 1  # → 18

        # zone one-hot
        z = n.get("current_zone", "")
        if z in zone2idx:
            v_feats[i, col + zone2idx[z]] = 1.0
        # col += n_zones  (final)

        # route_left: may already be int indices (from preloaded cache)
        rl_raw = n.get("route_left", [])
        if rl_raw and isinstance(rl_raw[0], int):
            route_left_idx.append(rl_raw)
        else:
            route_left_idx.append([re_id2idx[eid] for eid in rl_raw if eid in re_id2idx])

    # ── Road edges (junction → junction) ──────────────────────────────────
    active_re_ids = set(e["id"] for e in snap.get("road_edges_dynamic", []))
    if not active_re_ids:
        active_re_idxs = np.arange(si["n_road_edges"], dtype=np.int64)
    else:
        active_re_idxs = np.array(
            [re_id2idx[eid] for eid in active_re_ids if eid in re_id2idx], dtype=np.int64
        )

    re_src_all = si["re_src"]
    re_dst_all = si["re_dst"]
    active_j_set = set(re_src_all[active_re_idxs].tolist()) | set(re_dst_all[active_re_idxs].tolist())
    active_j_idxs   = np.array(sorted(active_j_set), dtype=np.int64)
    j_global2local  = {g: l for l, g in enumerate(active_j_idxs)}

    j_feats = si["junction_feats"][active_j_idxs]  # (N_j, j_in)

    re_src_local = np.array([j_global2local[g] for g in re_src_all[active_re_idxs]], dtype=np.int64)
    re_dst_local = np.array([j_global2local[g] for g in re_dst_all[active_re_idxs]], dtype=np.int64)

    # Road edge features: static(3) + dynamic(4) = 7
    re_static  = si["re_static_feats"][active_re_idxs]              # (N_are, 3)
    re_dyn_all = extract_road_dyn_features(snap, si, demand_stats)
    re_dyn     = re_dyn_all[active_re_idxs]                         # (N_are, 4)
    re_feats   = np.concatenate([re_static, re_dyn], axis=1)         # (N_are, 7)

    # ── Dynamic edges (junction ↔ vehicle, vehicle ↔ vehicle) ─────────────
    j2v_src, j2v_dst = [], []
    v2j_src, v2j_dst = [], []
    vv_src,  vv_dst  = [], []

    if use_dynamic_edges:
        for de in snap.get("dynamic_edges", []):
            etype = de.get("edge_type", 0)
            frm, to = de["from"], de["to"]
            if etype == 1:
                j_g = j_id2idx.get(frm)
                v_l = v_id2idx.get(to)
                if j_g is not None and j_g in j_global2local and v_l is not None:
                    j2v_src.append(j_global2local[j_g]); j2v_dst.append(v_l)
            elif etype == 2:
                v_l = v_id2idx.get(frm)
                j_g = j_id2idx.get(to)
                if v_l is not None and j_g is not None and j_g in j_global2local:
                    v2j_src.append(v_l); v2j_dst.append(j_global2local[j_g])
            elif etype == 3:
                v_s = v_id2idx.get(frm); v_d = v_id2idx.get(to)
                if v_s is not None and v_d is not None:
                    vv_src.append(v_s); vv_dst.append(v_d)

    # ── Build HeteroData ───────────────────────────────────────────────────
    data = HeteroData()
    data["junction"].x  = torch.from_numpy(j_feats)   # (N_j, j_in)
    data["vehicle"].x   = torch.from_numpy(v_feats)   # (N_v, v_in)

    data["junction", "road", "junction"].edge_index = torch.tensor(
        np.stack([re_src_local, re_dst_local]), dtype=torch.long
    )
    data["junction", "road", "junction"].edge_attr = torch.from_numpy(re_feats)  # (N_are, 7)

    data["vehicle"].route_left_idx  = route_left_idx
    data["vehicle"].v_ids           = v_ids
    data["junction"].active_j_idxs  = active_j_idxs
    data["junction"].active_re_idxs = active_re_idxs

    if use_dynamic_edges:
        def _ei(src, dst):
            return (torch.tensor([src, dst], dtype=torch.long) if src
                    else torch.zeros((2, 0), dtype=torch.long))
        data["junction", "j2v", "vehicle"].edge_index  = _ei(j2v_src, j2v_dst)
        data["vehicle",  "v2j", "junction"].edge_index = _ei(v2j_src, v2j_dst)
        data["vehicle",  "vv",  "vehicle"].edge_index  = _ei(vv_src, vv_dst)

    y_raw        = torch.tensor(
        [labels_map.get(vid, float("nan")) for vid in v_ids], dtype=torch.float32
    )
    labeled_mask = ~torch.isnan(y_raw)
    data["vehicle"].y_raw        = y_raw
    data["vehicle"].labeled_mask = labeled_mask

    return data
