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
| B1 | **AVG** (historical / OD-style average, MetaTTE-style) | train, val, **test** | optional† | No gradient steps; still evaluate on val for parity of splits. |
| B2 | **LR** (taxicab geometry / OD + time → duration) | train, val, **test** | optional† | Light sklearn baseline. |
| B3 | **GBM** (LightGBM / XGBoost on trip + time + geometry features) | train, val, **test** | optional† | Strong tabular ceiling. |
| B4 | **Route-sum** (sum of remaining edge times: limit and/or train-only hist. speeds) | N/A‡ | train, val, **test** | On Porto-G if edges exist; else define on map-matched path in Porto-T. Prefer **same leakage rules** as paper (train-only stats). |
| B5 | **DeepTTE** (or official / faithful reimplementation) | train, val, **test** | — | Trajectory-native; **Porto-T** primary. |
| B6 | **MetaTTE** (full method; e.g. MetaTTE-GRU or best variant from paper) | train, val, **test** | — | Named TTE SOTA line; **Porto-T**. Code: [maxwang967/MetaTTE](https://github.com/maxwang967/MetaTTE) (TensorFlow). Use **same chronological split** as paper if you want replication / comparison to their reported Porto numbers. |

† **SUMO optional:** If you can build the **same feature vector** (or OD analogue) from SUMO exports, add SUMO columns for B1–B3; otherwise keep **Porto-T only** for Table 2 and put **SUMO baselines** in a small appendix block (AVG + GBM + route-sum minimum).

‡ Route-sum is **not** “train” in the DL sense; **fit** historical speeds on train only, then score val/test.

**Deliverable:** Baseline **Table 2** — **B1–B6** on **Porto** (T and/or G as per row); caption states input modality.

### B2 — Extended (if time; strengthens “MetaTTE-style ladder”)

| # | Baseline | Porto-T | SUMO | Notes |
|---|----------|---------|------|--------|
| B7 | **TEMP** (neighbor-trip historical average, if implementable) | train, val, **test** | — | As in MetaTTE refs. |
| B8 | **WDR-style** (wide + deep + recurrent on handcrafted + segment features) | train, val, **test** | — | Heavy engineering; cite KDD 2018 WDR. |
| B9 | **STNN** | train, val, **test** | — | If code or faithful reimpl. available. |
| B10 | **MURAT** | train, val, **test** | — | Multi-task; heavier. |
| B11 | **Nei-TTE** | train, val, **test** | — | Graph+GRU on segments; needs road topology. |
| B12 | **Sequence-only** (Transformer or BiLSTM on path edge sequence from Porto-G or Porto-T) | train, val, **test** | optional on SUMO | Not in MetaTTE list but isolates “graph vs sequence” (reviewer #2). |

### B3 — Optional: published numbers only (no new training)

| # | Item | Dataset | Notes |
|---|------|---------|--------|
| B13 | **Published tables** (e.g. MetaTTE paper Porto columns) | Porto only | Use **only** if split + preprocessing + metric match yours; else appendix / discussion. Prefer **your** B6 retrain for the main table. |

---

## Summary counts (training jobs)

| Block | # trained models (typical) | Domains each |
|-------|---------------------------|--------------|
| **Ablations A1–A6** | 6 variants × 2 domains = **12** full training runs | SUMO + Porto-G |
| **Baselines B1–B6** | **6** methods × 1 domain = **6** (+ optional SUMO duplicates for B1–B3) | Porto-T (+ optional SUMO) |
| **Baselines B7–B12** | add **0–6** each on Porto-T | optional |

---

## Row alignment for the two paper tables

| Paper table | Rows | Datasets shown |
|-------------|------|----------------|
| **Table 1 — Ablations** | A1–A6 | SUMO **test**; Porto-G **test** (same metrics). |
| **Table 2 — Baselines** | B1–B6 (+ optional B7–B13) | Primarily **Porto** (T for AVG/LR/GBM/DeepTTE/**MetaTTE**; G or T for DSTRA-GNN **A6** — label columns clearly). |

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
