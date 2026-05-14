# SNCS review — revision plan (digest)

Working copy of the Cursor plan for the Springer *SN Computer Science* reviews on **DSTRA-GNN-SNCS**. Manuscript paths are relative to this folder (e.g. [sections/methodology.tex](sections/methodology.tex)).

## Overview

- Consolidated **Reviewer #1** and **Reviewer #2** comments with exact quotes, plain-English “what to do,” and your author updates.
- **Author updates:** (1) **Porto taxi** converted to the **same representation** as SUMO; (2) **train DSTRA-GNN and SOTA baselines on both** SUMO and Porto, then report comparable metrics and tuning parity.

### Positioning vs literature (what “run SOTA on Porto” means)

Yes: you **retrain** (or faithfully reimplement) a **small set of named baselines** on **your** Porto trajectory setup—**same splits, same label definition, same inference-time features** as DSTRA-GNN—then report **test** MAE/RMSE/(w)MAPE **next to** your model. That is how you **position** against the literature under reviewer pressure, not by copying someone else’s table from a different preprocessing or task slice.

**Also run the same baselines on SUMO** so you keep one **model × domain** story: Porto answers “real trajectories,” SUMO answers “controlled scale and ablations.”

**Do not** claim parity with a paper’s published Porto number unless the **pipeline matches**; instead say “we retrained X on our Porto graph/trajectory representation.”

Pick **3–5** implementable lines (typical bundle): **GBDT** on trip/route features, **classical route-time sum**, **sequence model** (LSTM/Transformer on path), **one TTE/GNN reference** (e.g. DeepTTE-style or GCN+Transformer style), optionally **one heavier graph ETA** if inputs align.

### Results layout (two tables) — agreed structure

1. **Ablation table** — Only **DSTRA-GNN code-defined variants** (e.g. `base_graph` … `temporal_route_aware`). Report **test** metrics on **SUMO** and, separately, on **graph-converted Porto** (same protocol per domain). Caption states that rows isolate **dynamic edges / route / temporal** inside your architecture.

2. **Baseline table** — **External** methods (MetaTTE-style ladder: AVG, LR, GBM, DeepTTE, … as you implement). Same metric columns; **test** only; seeds as agreed. **May be Porto-only** if most baselines are **trajectory-native** (DeepTTE, GBM on trip features) and your main real-world claim is Porto; then **state in caption + text** that this table is **real-data Porto** only. For **SUMO**, still report **DSTRA-GNN vs at least AVG + GBM + route-sum** (or a compact baseline block in appendix) so reviewers do not read SUMO as “only ablations vs average.” Rows that only apply to **trajectory** Porto vs **graph** Porto should be clear in the caption or sub-columns.

The two tables answer different questions: (A) **component contribution**; (B) **competitiveness vs established estimators**.

## Revision checklist (todos)

| ID | Task |
|----|------|
| `dual-dataset-sota` | Build/train **DSTRA-GNN + SOTA baselines** on **SUMO** and **Porto** (same protocol: splits, window H, metrics, seeds); classical/GBM/sequence + cited SOTA where implementable; log hyperparameter budget; **test-set** tables for both domains |
| `porto-real-data` | Manuscript + rebuttal: Porto source citation; conversion pipeline (map match, snapshots, routes, features vs SUMO); dataset stats; limitations (taxi/Porto vs full mix) |
| `align-val-test` | One source of truth for **validation vs test**; test table; split/window rules; MAPE from predictions |
| `leakage-doc` | Dijkstra **average speed** + **edge_route_count** scope (train-only vs full); sensing/availability paragraph |
| `method-clarity` | Intuition for temporal global pool, MoE, static edges for time; define **route left splits**; relation-type / dynamic-edge discussion |
| `repro-refs` | Reproducibility appendix: features, hyperparams, splits, code+commit, compute, inference latency; fix bibliography DOIs/URLs |

---

# What the reviewers are asking for (plain English)

Each numbered item starts with an **exact quoted sentence or passage** from the review (Reviewer #1 = **R1**, Reviewer #2 = **R2**). After the quote, **“What this means for your revision”** spells out the practical request.

---

**1.** **R1:** “First, the evaluation relies entirely on a simulation-derived dataset generated using SUMO rather than real-world traffic data. While the dataset is large and detailed, the lack of validation on real-world benchmarks raises concerns about generalizability and practical deployment. The authors acknowledge this limitation briefly, but the manuscript would benefit from stronger justification of the simulation realism or additional experiments using publicly available datasets.”

**What this means for your revision:** They are not saying SUMO is forbidden; they want either (a) a stronger argument that the simulation matches reality well enough for the claims you make, and/or (b) extra experiments on at least one public real dataset (even if the setup is not identical), so readers can see behavior outside your simulator.

**Author status (your update):** You already created a **real dataset** by converting the **Porto taxi trajectory (ECML PKDD 2015 prediction challenge)** data—public release commonly indexed as [UCI ML Repository: Taxi Service Trajectory (Prediction Challenge, ECML PKDD 2015)](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)—into the **same representation** as your SUMO graphs. That directly targets (b) above.

**Still to do in the paper/rebuttal (not optional if you rely on it):** Name the source and license/access; describe the **conversion** (time discretization, map matching / road graph, how routes and dynamic edges are formed, feature parity with SUMO); define **chronological splits** and reporting (val vs test, same discipline as R2 asks for SUMO); report **numbers** (same model/baselines as feasible—at least your method + strongest baselines); add **one honest limitation** paragraph (taxi trips ≠ full traffic mix; Porto geography; possible differences in sensing features vs. simulation). Optionally keep SUMO as the large controlled study and Porto as **external validity**.

---

**2.** **R1:** “Second, the comparison with state-of-the-art methods is limited. Although the related work section discusses prior models such as DeepTTE, DCRNN, and DuETA, the experimental section does not include direct implementation-based comparisons with these methods. Instead, comparisons rely mainly on ablation variants and a simple average baseline. Including empirical comparisons with established benchmarks would provide stronger evidence of performance claims.”

**R2 (same issue, different wording):** “The experimental comparison appears to be mostly against an ‘average ETA’ baseline and internal ablations.” / “Without strong baselines, the reported ‘82% improvement’ is difficult to interpret: it may largely reflect a weak baseline rather than the method being state-of-the-art.”

**What this means for your revision:** Implement and report numbers for **external** baselines (classical route-time sums, gradient boosting on hand-crafted features, a sequence model, and ideally one spatiotemporal GNN), with a short statement that tuning effort is comparable. Related work alone is not enough; they want **tables with actual runs**.

**Author strategy (your update):** **Build and train** your model **and** the **SOTA / strong baselines** on **both** the **SUMO-based** dataset and the **real Porto** dataset (same representation for Porto). That gives a clean **model × domain** matrix: readers see (i) competitiveness vs named methods on your main benchmark, (ii) whether ordering holds on real trajectories, and (iii) that the large SUMO % gain is not an artifact of a single weak baseline on one simulator.

**Practical notes when executing:** Pick a **finite SOTA set** you can actually implement or adapt to your tensors (DeepTTE / DuETA-class TTE models, DCRNN or another STGNN if the graph/timeseries view fits, plus at least **GBM + route-sum** as R2 asked). Where a method’s original formulation does not match your inputs, state the **adaptation** (fair feature access, same train/val/test, same early-stopping rule). Report **test** results and **std across seeds** on **each** dataset; avoid claiming one aggregate “beats all papers everywhere.”

---

**3.** **R1:** “Third, the methodology section is detailed but sometimes difficult to follow due to dense technical descriptions and extensive notation. Some components, such as the temporal aggregation process and route encoding mechanism, would benefit from clearer explanation and intuitive justification. Additionally, the rationale for certain design choices, including the use of static road edges for temporal aggregation and the specific configuration of the Mixture-of-Experts architecture, is not sufficiently motivated.”

**What this means for your revision:** Add short intuitive paragraphs (and maybe one schematic sentence each): *why* temporal features live on static road edges, *why* pooling to one graph vector is reasonable (or what you lose), *why* MoE with your k—not just equations. They want a reader to understand the design without re-deriving everything.

---

**4.** **R1:** “Fourth, the paper claims substantial improvements (over 80%) compared to existing approaches, but the experimental context differs from prior studies in dataset structure, evaluation scope, and problem formulation. Therefore, these comparisons should be interpreted cautiously, and the manuscript should better clarify the limitations of cross-study performance comparisons.”

**What this means for your revision (updated given your dual-dataset SOTA plan):** R1’s concern is mainly about **comparing your headline % to numbers reported in other papers** on different benchmarks. Once you report **direct runs of established models on the same SUMO and Porto splits** as your method (see **dual-dataset-sota**), the core of comment **4 is addressed empirically**: competitiveness is shown **in your experimental context**, not by citing someone else’s table.

**Residual (light touch):** Keep **one short sentence** anywhere you still mention prior papers’ reported gains (e.g. “40–50%” in related work or discussion): those figures are **illustrative of the literature**, not a claim that your % is comparable to theirs. Avoid reframing your internal **% vs average baseline** as “beats DuETA everywhere.” Beyond that, you do **not** need a long “limitations of cross-study comparison” essay if the manuscript no longer leans on cross-study % matching as evidence.

**Author view (your note):** Treat comment **4 as largely superseded** by same-protocol **SOTA baselines on your data**; prioritize tables over defensive prose.

---

**5.** **R1:** “The reproducibility of the work could also be improved. While the authors mention that code and dataset generation tools are available, the manuscript does not clearly describe hyperparameter sensitivity, computational requirements, or training stability. Providing more implementation details would enhance transparency.”

**What this means for your revision:** Add concrete text (or an appendix) on hardware, training time, sensitivity or stability notes if any, and what you tried for important hyperparameters—not only “code exists.”

---

**6.** **R1:** “Finally, the manuscript would benefit from language refinement and minor editorial improvements. Some sections contain repetitive explanations, and the discussion occasionally overemphasizes contributions without sufficiently addressing limitations. Improving clarity and balance in presentation would strengthen the overall quality of the paper.”

**What this means for your revision:** Copy-edit pass: cut repetition, expand limitations, slightly dial back marketing tone in discussion.

---

**7.** **R2:** “Section 3.6 states that ‘All metrics … [are] averaged over vehicles in the test split,’ yet the Results section explicitly presents ‘Validation results’ and tables labeled ‘overall validation’. It is unclear whether the headline numbers are validation-only, test-only, or a mix.”

**R2 (concrete fix):** “Provide a dedicated test-set table (MAE/RMSE/MAPE + stratifications), and keep validation strictly for early stopping/model selection.” / “Explicitly document the exact split boundaries (time ranges/snapshot indices) and how you avoid boundary leakage with windowing (e.g., discarding the first H−1 snapshots of each split so windows don't overlap across splits).” / “Include confidence intervals/standard deviations across seeds (not just means).”

**What this means for your revision:** One clear story: **validation** = model picking; **test** = numbers in the main tables. Fix the contradiction between [sections/methodology.tex](sections/methodology.tex) and [sections/results.tex](sections/results.tex). State split date/index ranges and that windows do not straddle splits. Add std across seeds.

---

**8.** **R2:** “Routes are computed using Dijkstra with edge weights derived from edge length and average speed. It is unclear whether this ‘average speed’ is (a) a static speed limit proxy, (b) computed from training history only, or (c) computed using future data (which would leak).”

**R2 (concrete fix):** “Specify exactly how ‘average speed’ for route planning is computed and restrict it to training-only history (or a static speed limit) for route construction in validation/test.”

**What this means for your revision:** One explicit paragraph: which option (a/b/c) you use; if you used anything that peeked at val/test, rerun with a clean rule.

---

**9.** **R2:** “You include ‘edge route count’ (number of routes traversing each edge). If computed using all trips, it may encode future demand patterns that wouldn't be available at training time in a real deployment (or at least should be computed from training period only).”

**R2 (concrete fix):** “Compute ‘edge route count’ using training data only (or an explicit historical window) and report the impact via ablation.”

**What this means for your revision:** Define how the feature was built; if it used the full corpus, rebuild from train-only (or past-only) and optionally show an ablation.

---

**10.** **R2:** “The figure shows inputs ‘vehicle route left splits,’ but I could not find a definition of what these ‘splits’ are or how computed.” / “Define ‘route left splits’ precisely and show an ablation removing it.”

**What this means for your revision:** In text and/or caption, define “splits” (or rename to something self-explanatory). If it is a distinct feature, add/remove ablation.

---

**11.** **R2:** “Table 4 notes that ‘MAPE values are estimated where missing in logs,’ which is unusual and concerning for a primary result table.” / “Recompute MAPE directly from predictions/labels (avoid estimation); also consider sMAPE or wMAPE, given ETA can be near-zero and MAPE becomes unstable.”

**What this means for your revision:** Recompute all MAPE from stored predictions and ground truth; drop “estimated from logs.” Consider reporting wMAPE/sMAPE or robust metrics as R2 suggests.

---

**12.** **R2:** “In your ablations, adding dynamic edges without route features hurts (dynamic graph MAE 104.5s vs base graph 86.5s).” / “Add missing ablations to isolate effects: Route features with a static-only graph (no dynamic edges) … Treat relation types explicitly (e.g., relation-specific parameters or a hetero-GNN) … Provide diagnostics: edge-type attention weights, performance vs. vehicle density, and controlled experiments varying the interaction threshold.”

**What this means for your revision:** They doubt that “dynamic GNN” is the real story unless you show when/why vehicle–vehicle edges help. Add targeted ablations and/or hetero modeling and brief analysis—otherwise acknowledge that route+temporal dominate.

---

**13.** **R2:** “The DOI and/or URL of [1], [11], [14], [17], [18], [20], [21], [23], and [27] are wrong.” / “Section 2.2 refers to a DiDi 2016 dataset. However, the reference [7] does not specify a specific dataset.” / “The DOI is missing from [19], but it is https://doi.org/10.1007/978-3-032-06164-5_1”

**What this means for your revision:** Bibliography hygiene: correct links/DOIs and align the DiDi text with the actual reference.

---

**14.** **R2 (minor / reporting):** “Clarify the mismatch between ‘duration bins’ (Fig. 6; Table 3) and ‘route-length category’ language around figures/tables.” / “Report standard deviations across seeds in Tables 2/4, not only the mean.” / “The temporal module pools edge context by mean over all static edges; discuss why a global context is appropriate versus route- or location-conditioned context.” / “Provide the actual number of windows and average |V|, |E| per window/batch; batch size 2 is hard to interpret without knowing the unit.” / “If ‘trips: 22,767,090’ is correct, define ‘trip’ precisely (unique OD instance? per-vehicle per-day?); it seems high relative to unique vehicles.”

**What this means for your revision:** Editorial/clarity fixes: consistent terminology, extra table stats, one paragraph defending global temporal pooling (or acknowledging limitation), define “trip,” add |V|/|E|/window counts.

---

**15.** **R2:** “Even though the study is simulation-based, ETA models and route-aware dispatch can create distributional impacts … a fairness analysis across zones (A/B/C/H) would be valuable.” / “If deployed on real data, ‘vehicles on road count’ and interaction information raise privacy/surveillance concerns depending on sensing assumptions; the paper should clarify what is realistically observable and what is simulation-only.”

**What this means for your revision:** Short ethics/societal paragraph: fairness across zones, what sensors would be needed in real deployment vs what SUMO gives for free.

---

**16.** **R2 (technical suggestions, not always mandatory but signal expectations):** “Route left is encoded as edge IDs with mean pooling (order-invariant). This likely discards important sequential structure … A simple improvement: a GRU/Transformer over the edge-ID sequence …” / “Provide inference time per window (CPU/GPU) and scaling curves as the number of vehicles increases.” / “Dataset description/access … there's no clear way to obtain the dataset or fully reproduce the generation parameters.” / “Missing full feature definitions (28 node / 7 edge) …” / “Code: Not clearly provided in the manuscript text I saw; if it exists, link it prominently and include an exact commit hash.”

**What this means for your revision:** Optional model upgrades (sequence route encoder); **reproducibility pack**: data access, full feature list, code link + commit, latency/scaling numbers.

---

## Reviewer #2 “Questions for the authors” (1–10)

These align with items **7–12** and **16**: (1) val vs test numbers; (2) average speed for Dijkstra; (3) edge route count scope; (4) interaction-rule constants and sensitivity; (5) “route left splits”; (6) trips crossing splits; (7) shared vs relation-specific GNN parameters; (8) MoE vs single head and expert behavior; (9) MAPE recomputation; (10) runtime per window and scaling. Answer them in the **response letter** and mirror fixes in the **revised manuscript**.

---

## Suggested revision order

1. **Dual-dataset experiment campaign:** finish training **DSTRA-GNN + SOTA/strong baselines** on **SUMO and Porto** with a locked evaluation protocol; export test metrics + seed variance.
2. **Write experiments:** two-domain tables (and optional summary figure: relative rank / MAE); conversion + limitations for Porto; **fair comparison** wording; **brief** note only if you still cite other papers’ reported % gains (cross-benchmark numbers are illustrative, not matched to your protocol).
3. Align methodology/results: **test** table, splits, windows, MAPE from predictions, seed std (**both** datasets).
4. Leakage audit: Dijkstra speeds + `edge_route_count` construction; sensing paragraph (note domain differences: SUMO vs Porto observability).
5. Method intuition + dynamic-edge diagnostics + optional hetero/ablations.
6. References, definitions (trip, splits), tables/figures cleanup, prose pass.

---

*Generated for the DSTRA-GNN-SNCS revision; keep in sync with any updates to the Cursor plan if the strategy changes.*
