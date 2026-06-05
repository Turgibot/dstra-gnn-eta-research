"""Snapshot loading, feature extraction, and graph construction for DSTRA-GNN."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


# ── Static index builders ─────────────────────────────────────────────────────

def build_static_index(static_path: Path) -> dict:
    """Load static.json and return lookup dicts for junctions and road edges."""
    static = json.loads(static_path.read_text())

    junction_ids = [j["id"] for j in static["junctions"]]
    j_id2idx     = {jid: i for i, jid in enumerate(junction_ids)}

    road_edge_ids    = [e["id"] for e in static["road_edges"]]
    re_id2idx        = {eid: i for i, eid in enumerate(road_edge_ids)}
    road_edge_info   = {e["id"]: e for e in static["road_edges"]}

    # Junction features: (N_j, 4) = [x_norm, y_norm, is_internal, degree]
    xs = np.array([j["x"] for j in static["junctions"]], dtype=np.float32)
    ys = np.array([j["y"] for j in static["junctions"]], dtype=np.float32)
    x_scale = max(xs.max() - xs.min(), 1.0)
    y_scale = max(ys.max() - ys.min(), 1.0)
    x_norm  = (xs - xs.min()) / x_scale
    y_norm  = (ys - ys.min()) / y_scale
    is_int  = np.array([1.0 if j["type"] == "internal" else 0.0
                        for j in static["junctions"]], dtype=np.float32)
    # degree = len(incoming) + len(outgoing)
    degree = np.array(
        [len(j.get("incoming", [])) + len(j.get("outgoing", [])) for j in static["junctions"]],
        dtype=np.float32,
    )
    degree_norm = degree / max(degree.max(), 1.0)
    junction_feats = np.stack([x_norm, y_norm, is_int, degree_norm], axis=1)  # (N_j, 4)

    # Road edge static features: (N_re, 3) = [length_norm, speed_norm, lanes_norm]
    max_len   = max(e["length"] for e in static["road_edges"])
    max_speed = max(e["speed"]  for e in static["road_edges"])
    re_static_feats = np.array(
        [[e["length"] / max(max_len, 1.0),
          e["speed"]  / max(max_speed, 1.0),
          e["num_lanes"] / 3.0]
         for e in static["road_edges"]],
        dtype=np.float32,
    )  # (N_re, 3)

    # Road edge connectivity: (2, N_re) COO
    re_src = np.array([j_id2idx.get(e["from"], 0) for e in static["road_edges"]], dtype=np.int64)
    re_dst = np.array([j_id2idx.get(e["to"],   0) for e in static["road_edges"]], dtype=np.int64)

    return {
        "junction_ids":      junction_ids,
        "j_id2idx":          j_id2idx,
        "junction_feats":    junction_feats,       # (N_j, 4)
        "road_edge_ids":     road_edge_ids,
        "re_id2idx":         re_id2idx,
        "road_edge_info":    road_edge_info,
        "re_static_feats":   re_static_feats,      # (N_re, 3)
        "re_src":            re_src,               # (N_re,)
        "re_dst":            re_dst,               # (N_re,)
        "n_junctions":       len(junction_ids),
        "n_road_edges":      len(road_edge_ids),
        "max_len":           max_len,
        "max_speed":         max_speed,
    }


# ── Per-snapshot data ─────────────────────────────────────────────────────────

def load_snapshot_raw(path: Path) -> Optional[dict]:
    """Load a snapshot JSON; return None if empty or corrupt."""
    raw = path.read_text()
    if not raw.strip():
        return None
    return json.loads(raw)


def extract_road_dyn_features(snap: dict, si: dict) -> np.ndarray:
    """Return (N_re, 3) dynamic road-edge features for this snapshot.

    Columns: [avg_speed_norm, density_norm, edge_demand_norm].
    Edges absent from road_edges_dynamic get zeros.
    """
    N_re     = si["n_road_edges"]
    re_id2idx = si["re_id2idx"]
    max_spd   = si["max_speed"]
    feats     = np.zeros((N_re, 3), dtype=np.float32)
    for e in snap.get("road_edges_dynamic", []):
        idx = re_id2idx.get(e["id"])
        if idx is None:
            continue
        feats[idx, 0] = e["avg_speed"]  / max(max_spd, 1.0)
        feats[idx, 1] = e["density"]
        feats[idx, 2] = e["edge_demand"] / 10.0   # rough normalisation
    return feats


def build_hetero_graph(
    snap:       dict,
    labels_map: Dict[str, float],
    si:         dict,
    use_dynamic_edges: bool = True,
) -> Tuple[HeteroData, torch.Tensor, Dict[str, int], List[str]]:
    """Build a PyG HeteroData from a single snapshot.

    Returns:
        graph         HeteroData
        y_log         log-normalised ETA labels for labeled vehicles (N_v_labeled,)
        v_id2idx      vehicle-id → local vehicle index
        vehicle_ids   ordered list of vehicle IDs (all vehicles in snap)
    """
    j_id2idx  = si["j_id2idx"]
    re_id2idx = si["re_id2idx"]
    max_spd   = si["max_speed"]

    # ── Vehicle nodes ──────────────────────────────────────────────────────
    nodes = snap.get("nodes", [])
    if not nodes:
        return None, None, {}, []

    v_ids   = [n["id"] for n in nodes]
    v_id2idx = {vid: i for i, vid in enumerate(v_ids)}
    N_v = len(nodes)

    max_route_len = max((n.get("route_length_left", 0) for n in nodes), default=1.0)

    v_feats = np.array(
        [[n.get("speed", 0.0)              / max(max_spd, 1.0),
          n.get("acceleration", 0.0)       / 10.0,
          n.get("current_position", 0.0)   / 100.0,
          n.get("route_length_left", 0.0)  / max(max_route_len, 1.0),
          1.0 if n.get("vehicle_type", "passenger") == "passenger" else 0.0]
         for n in nodes],
        dtype=np.float32,
    )  # (N_v, 5)

    # route_left per vehicle (list of road-edge indices)
    route_left_idx = []
    for n in nodes:
        idx_list = [re_id2idx[eid] for eid in n.get("route_left", []) if eid in re_id2idx]
        route_left_idx.append(idx_list)

    # ── Road edges (junction → junction) ──────────────────────────────────
    # Subgraph: only edges in road_edges_dynamic + their junctions
    active_re_ids = set(e["id"] for e in snap.get("road_edges_dynamic", []))
    if not active_re_ids:
        # Fallback: use all road edges (SUMO with small graph)
        active_re_idxs = np.arange(si["n_road_edges"], dtype=np.int64)
    else:
        active_re_idxs = np.array(
            [re_id2idx[eid] for eid in active_re_ids if eid in re_id2idx], dtype=np.int64
        )

    # Active junctions = endpoints of active road edges
    re_src_all = si["re_src"]
    re_dst_all = si["re_dst"]
    active_j_idxs_set = set(re_src_all[active_re_idxs].tolist()) | set(re_dst_all[active_re_idxs].tolist())
    active_j_idxs = np.array(sorted(active_j_idxs_set), dtype=np.int64)

    # Local junction index remapping
    j_global2local = {g: l for l, g in enumerate(active_j_idxs)}
    N_j = len(active_j_idxs)

    j_feats = si["junction_feats"][active_j_idxs]  # (N_j, 4)

    # Road edge src/dst in LOCAL junction indices
    re_src_local = np.array([j_global2local[g] for g in re_src_all[active_re_idxs]], dtype=np.int64)
    re_dst_local = np.array([j_global2local[g] for g in re_dst_all[active_re_idxs]], dtype=np.int64)

    # Road edge features: static + dynamic
    re_static  = si["re_static_feats"][active_re_idxs]  # (N_are, 3)
    re_dyn_all = extract_road_dyn_features(snap, si)
    re_dyn     = re_dyn_all[active_re_idxs]              # (N_are, 3)
    re_feats   = np.concatenate([re_static, re_dyn], axis=1)  # (N_are, 6)

    # ── Dynamic edges (junction ↔ vehicle, vehicle ↔ vehicle) ─────────────
    j2v_src, j2v_dst = [], []
    v2j_src, v2j_dst = [], []
    vv_src,  vv_dst  = [], []

    if use_dynamic_edges:
        for de in snap.get("dynamic_edges", []):
            etype = de.get("edge_type", 0)
            frm, to = de["from"], de["to"]
            if etype == 1:   # junction → vehicle
                j_g = j_id2idx.get(frm)
                v_l = v_id2idx.get(to)
                if j_g is not None and j_g in j_global2local and v_l is not None:
                    j2v_src.append(j_global2local[j_g])
                    j2v_dst.append(v_l)
            elif etype == 2:  # vehicle → junction
                v_l = v_id2idx.get(frm)
                j_g = j_id2idx.get(to)
                if v_l is not None and j_g is not None and j_g in j_global2local:
                    v2j_src.append(v_l)
                    v2j_dst.append(j_global2local[j_g])
            elif etype == 3:  # vehicle ↔ vehicle
                v_s = v_id2idx.get(frm)
                v_d = v_id2idx.get(to)
                if v_s is not None and v_d is not None:
                    vv_src.append(v_s); vv_dst.append(v_d)

    # ── Build HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    data["junction"].x  = torch.from_numpy(j_feats)             # (N_j, 4)
    data["vehicle"].x   = torch.from_numpy(v_feats)             # (N_v, 5)

    data["junction", "road", "junction"].edge_index = torch.tensor(
        np.stack([re_src_local, re_dst_local]), dtype=torch.long
    )
    data["junction", "road", "junction"].edge_attr = torch.from_numpy(re_feats)  # (N_are, 6)

    # Store metadata for later use
    data["vehicle"].route_left_idx  = route_left_idx
    data["vehicle"].v_ids           = v_ids
    data["junction"].active_j_idxs  = active_j_idxs  # global→local mapping
    data["junction"].active_re_idxs = active_re_idxs  # for temporal lookup

    if use_dynamic_edges:
        def _ei(src, dst):
            if src:
                return torch.tensor([src, dst], dtype=torch.long)
            return torch.zeros((2, 0), dtype=torch.long)

        data["junction", "j2v", "vehicle"].edge_index = _ei(j2v_src, j2v_dst)
        data["vehicle",  "v2j", "junction"].edge_index = _ei(v2j_src, v2j_dst)
        data["vehicle",  "vv",  "vehicle"].edge_index  = _ei(vv_src, vv_dst)

    # ── Labels ────────────────────────────────────────────────────────────
    # Return raw ETA in seconds; log-normalisation applied in training loop
    y_raw = torch.tensor(
        [labels_map.get(vid, float("nan")) for vid in v_ids], dtype=torch.float32
    )
    # Mask: only vehicles with valid labels
    labeled_mask = ~torch.isnan(y_raw)

    data["vehicle"].y_raw        = y_raw
    data["vehicle"].labeled_mask = labeled_mask

    return data
