# Model training & validation matrix

**Datasets (short names)**

| ID | Dataset | Use |
|----|---------|-----|
| **SUMO** | Simulation graph snapshots (dynamic traffic graph). | Primary controlled domain; ablations + optional compact baselines. |
| **Porto-G** | Porto taxi converted to the **same graph representation** as SUMO. | Ablations + DSTRA-GNN full model in baseline/ablation tables on real data. |
| **Porto-T** | Porto taxi **raw trajectory** (polyline / DeepTTE-style inputs + tabular features). | Trajectory-native baselines (DeepTTE, GBDT rows, etc.). |

**Protocol:** For every **learned** row: fit on **train**, tune / early-stop on **validation**, report **test** only in the paper (mean ± std over seeds, same as current manuscript intent).

---

## Part A — Ablations (DSTRA-GNN variants)

Train + validate + test **each** variant on **both** SUMO and Porto-G (same code paths, different dataloaders).

| # | Model / variant | SUMO | Porto-G | Notes |
|---|-----------------|------|---------|--------|
| A1 | `base_graph` | train, val, **test** | train, val, **test** | Static road graph only; no dynamic edges; no temporal; no route. |
| A2 | `dynamic_graph` | train, val, **test** | train, val, **test** | + dynamic edges; no temporal; no route. |
| A3 | `route_aware_graph` | train, val, **test** | train, val, **test** | + route + dynamic edges; no temporal. |
| A4 | `temporal_base` | train, val, **test** | train, val, **test** | + temporal (static-road edges only); no dynamic edges; no route. |
| A5 | `temporal_dynamic` | train, val, **test** | train, val, **test** | + temporal + dynamic edges; no route. |
| A6 | `temporal_route_aware` | train, val, **test** | train, val, **test** | Full model (reference for Part B). |

**Deliverable:** Ablation **Table 1** — rows A1–A6 × metrics on SUMO and Porto-G (test).

---

## Part B — External baselines

### B1 — Core (recommended minimum for reviewers)

Train + validate + test on **Porto-T** (and on **SUMO** only where the baseline is defined on your SUMO features — optional second column for simulation).

| # | Baseline | Porto-T | SUMO | Notes |
|---|----------|---------|------|--------|
| B1 | **AVG** (historical / OD-style average, MetaTTE-style) | train, val, **test** | train, val, **test** | No gradient steps; still evaluate on val for parity of splits. |
| B2 | **LR** (taxicab geometry / OD + time → duration) | train, val, **test** | train, val, **test** | Light sklearn baseline. |
| B3 | **GBM** (LightGBM / XGBoost on trip + time + geometry features) | train, val, **test** | train, val, **test** | Strong tabular ceiling. |
| B4 | **Route-sum** (sum of remaining edge times: limit and/or train-only hist. speeds) | N/A‡ | train, val, **test** | On Porto-G if edges exist; else define on map-matched path in Porto-T. Prefer **same leakage rules** as paper (train-only stats). |
| B5 | **DeepTTE** (or official / faithful reimplementation) | train, val, **test** | — | Trajectory-native; **Porto-T** primary. |
| B6 | **MetaTTE** (full method; e.g. MetaTTE-GRU or best variant from paper) | train, val, **test** | — | Named TTE SOTA line; **Porto-T**. Code: [maxwang967/MetaTTE](https://github.com/maxwang967/MetaTTE) (TensorFlow). Use **same chronological split** as paper if you want replication / comparison to their reported Porto numbers. |
| B7 | **TEMP** (neighbor-trip historical average) | train, val, **test** | — | As reported in MetaTTE's Porto comparison ladder; same chronological split. |
| B8 | **WDR-style** (wide + deep + recurrent on handcrafted + segment features) | train, val, **test** | — | Heavy engineering; cite KDD 2018 WDR; also in MetaTTE's Porto ladder. |
| B9 | **STNN** | train, val, **test** | — | Spatio-temporal neural baseline reported on Porto in MetaTTE; faithful reimpl. if no official code. |

‡ Route-sum is **not** “train” in the DL sense; **fit** historical speeds on train only, then score val/test.

**B1–B3 on SUMO** (results in `results/{avg,lr,gbm}_sumo/metrics.json`): per-(snapshot, vehicle) remaining-trip table built from `current_x/y` → `destination_x/y` (planar OD geometry), `route_length_left` (polyline-distance analogue), and `step` (hour/day-of-week), same framing as B4/B10. Train at stride 10 (~1.45M rows), val/test at stride 1 (~7.4M rows each, matching B4/B10 `n_pairs`). Test MAE/RMSE/MAPE: AVG 516.87s/1228.94/82.43%, LR 1063.82s/1895.94/287.40%, GBM 454.32s/1237.78/35.99% — vs. route-sum 875.19s/2194.34/63.01% and DCRNN 808.13s/1989.70/79.67%.

**Deliverable:** Baseline **Table 2** — **B1–B10** on **Porto** (T and/or G as per row); caption states input modality.

### B10 — Graph-native diffusion-conv baseline

| # | Baseline | Porto-G | SUMO | Notes |
|---|----------|---------|------|--------|
| B10 | **DCRNN** (diffusion-conv GRU forecaster + route-sum ETA) | train, val, **test** | train, val, **test** | Simplified 1-layer DCGRU encoder forecasts next-step `avg_speed` per road edge on the road-edge line graph (dual random-walk diffusion supports); ETA via route-sum over `route_left` using the forecast speed (falling back to B4's train-only historical mean speed for edges the model predicts as inactive). Genuine graph-native spatio-temporal baseline (vs. tabular/sequence B1–B9). |

### B2 — Extended (if time; strengthens “MetaTTE-style ladder”)

| # | Baseline | Porto-T | SUMO | Notes |
|---|----------|---------|------|--------|
| B11 | **MURAT** | train, val, **test** | — | Multi-task; heavier. |
| B12 | **Nei-TTE** | train, val, **test** | — | Graph+GRU on segments; needs road topology. |
| B13 | **Sequence-only** (Transformer or BiLSTM on path edge sequence from Porto-G or Porto-T) | train, val, **test** | optional on SUMO | Not in MetaTTE list but isolates “graph vs sequence” (reviewer #2). |

### B3 — Optional: published numbers only (no new training)

| # | Item | Dataset | Notes |
|---|------|---------|--------|
| B14 | **Published tables** (e.g. MetaTTE paper Porto columns) | Porto only | Use **only** if split + preprocessing + metric match yours; else appendix / discussion. Prefer **your** B6 retrain for the main table. |

---

## Summary counts (training jobs)

| Block | # trained models (typical) | Domains each |
|-------|---------------------------|--------------|
| **Ablations A1–A6** | 6 variants × 2 domains = **12** full training runs | SUMO + Porto-G |
| **Baselines B1–B9** | **9** methods × 1 domain = **9** (+ B1–B3 also trained on SUMO) | Porto-T (+ SUMO for B1–B3) |
| **Baseline B10 (DCRNN)** | **1** method × 2 domains = **2** | SUMO + Porto-G |
| **Baselines B11–B13** | add **0–3** each on Porto-T | optional |

---

## Row alignment for the two paper tables

| Paper table | Rows | Datasets shown |
|-------------|------|----------------|
| **Table 1 — Ablations** | A1–A6 | SUMO **test**; Porto-G **test** (same metrics). |
| **Table 2 — Baselines** | B1–B10 (+ optional B11–B14) | Primarily **Porto** (T for AVG/LR/GBM/DeepTTE/**MetaTTE**/**TEMP**/**WDR**/**STNN**; G or T for DSTRA-GNN **A6**; G+SUMO for **DCRNN** — label columns clearly). |

**Important:** If **Table 2** reports **DSTRA-GNN**, that row should be **A6 evaluated on the same Porto modality** as the table caption (recommend: **Porto-G** for apples-to-apples “graph ETA on real city,” and trajectory baselines on **Porto-T** in the same table only if you clearly label columns **“trajectory baselines (Porto-T)”** vs **“graph model (Porto-G)”**).

---

## Second real-world city (why there is no extra column)

Dense **GPS polylines** suitable for **map-matching → road-graph** ETA are **rarely open**: NYC TLC is trip-level (no polylines); T-Drive and similar are **too sparse** for your TTE/graph pipeline; Chengdu/DiDi mirrors are often **gated or dead links**. **Argument in the paper** (1 short limitation + data-availability paragraph) is the right response **together with** SUMO + Porto-G/Porto-T + strong baselines—**not** delaying the revision to hunt a phantom second open city.

### Strong argument vs more actions

| Approach | When it is enough |
|----------|-------------------|
| **Write it clearly** (limitation + why Porto + SUMO) | Almost always, **if** Table 2 has **retrained** comparators (B1–B6) + **test** metrics + **split** documentation. |
| **More actions** | Add only **high-yield** items: e.g. **replicate MetaTTE Porto split** (B6 + optional B13 check), **SUMO mini-baseline block** (appendix), **leakage audit** text—not another year on broken download links. |

**Verdict:** State the **data landscape argument firmly and briefly**; spend remaining effort on **experiments and tables**, not on proving a second open dense-trajectory city exists.

---

*Adjust variant names if your code uses different flags; keep A1–A6 aligned with `sections/experiments.tex` ablation table.*
