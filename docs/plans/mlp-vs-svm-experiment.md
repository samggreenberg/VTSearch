# MLP vs SVM: the definitive experiment

**Goal:** settle, with one rigorous experiment, whether VTSearch's production ranker (the small MLP in `vtscore/training/mlp.py`) should be replaced by an SVM — and if so, which kernel. The experiment measures **the correctness of VTSearch as a user actually experiences it**: votes cast in the order the app's Autopilot presents them, the production threshold-calibration path, and FPR + FNR over voting time against a held-out test set. Runtime (GPU training time per training size, GPU inference time per inference size) is measured alongside as a tiebreaker, not a decision driver.

**Scope is deliberately narrow:** image media type only, SigLIP only (`google/siglip-base-patch16-224`, 768-d, unit-norm vectors). No audio/video/text, no patch embedders, no region voting. Other media types can reuse the harness later if this experiment motivates it.

This plan is written to be executed on a GPU cluster. Everything up to "Open work items" is design; the work items at the bottom are the code changes that must land before the cluster run.

## Background: what already exists

The repo already contains most of the machinery, built for earlier rounds of this same question:

- **`vtscore/training/svm.py`** — a standalone SVM trainer (`train_svm`) with `linear` (LinearSVC) and `rbf` (SVC, `gamma="scale"`) kernels, inclusion-aware class weights mirroring the MLP's, and a `decision_sigmoid` score mode chosen specifically so its output contract matches `torch.sigmoid(mlp(X))` for ranking and threshold-finding. Deliberately not wired into production.
- **`vtscore/eval/label_curve.py`** — the MLP-vs-SVM label-count sweep with the `TRAINERS` registry (`mlp`, `svm_linear`, `svm_rbf`, `mlp_ens{3,5,7,10}`), a trainer-agnostic port of the production cross-calibration threshold (`_cross_calibrated_threshold`), and AUROC/AP/best-F1/`f1_at_xcal` + train/predict timing per cell. CLI: `python -m vtscore.eval.label_curve_main`.
- **`vtscore/eval/voting_iterations.py` + `vtscore/eval/al_strategies.py`** — the Autopilot-faithful voting simulation: text-sort (or known-good) seeding, 3-good / 4-bad initial phases, then a Hard (margin-to-threshold) / New (coverage-atlas diversity) interleave; per-step retrain + production-exact cross-calibration (`_train_and_calibrate` pins the fold RNG to `RandomState(42)` just like `cross_calibration_threshold_cached`); per-step `cost`/`fpr`/`fnr` against a held-out test split (`sim_fraction`, default 0.5). **This harness is currently MLP-hardwired** — `_train_and_calibrate` calls `train_model` directly, and `_score_pool` / `_evaluate_on_test` / `_blend_safe_threshold` assume a torch module.
- **Image demo datasets with ground truth** — `caltech101_*`, `caltech256_a`, `visual_genome_*` (multi-label), `vggface2_faces_*` (identity), each with per-category text queries in `vtscore/eval/config.py` (the queries a user would type). Measured item counts in `vtscore/datasets/demo_counts.py`.
- **GPU story** — `train_model` runs on CUDA with AMP when available (`get_torch_device()`); `vtscore/gpu_backends.py` gates optional cuML (used today for k-means/UMAP); `scripts/experiments/toponymy_image/` is a working SLURM stage-pipeline template (`prepare_dataset → … → summarize`, node-scratch env setup in `common.py`, `timed()` context manager).

What is missing, and what the work items below add: a trainer-pluggable voting simulation, more SVM kernels + a GPU (cuML) SVM backend, text-sort seed-score glue, prevalence control for the rare-event arm, per-step timing columns, a dedicated GPU timing microbenchmark, and the orchestrator + report generator.

## Why compare these two at all (design rationale)

The mechanical difference — hinge loss + convex quadratic program vs cross-entropy + Adam — is not the interesting part. (Convexity does mean the SVM has one deterministic answer per dataset, no seeds or early stopping, while the MLP has a small landscape of answers.) The real divide is what each model *assumes about the data*, and those assumptions predict where each should win.

**SVM: "only the boundary matters."** The SVM's worldview is geometric. It assumes the classes are (nearly) separable in the feature space it was handed, and that the decision surface should be determined **entirely by the hardest examples** — the support vectors. The 40 easy votes a user casts contribute literally nothing; they sit inside the margin and are discarded. Two consequences:

- *Extremely label-efficient when labels are clean.* Margin-based generalization bounds depend on the margin, not the dimension — genuinely reassuring at 20 votes in 768-d SigLIP space. This small-N/high-D sweet spot is exactly VTSearch's operating regime.
- *Brittle to label noise.* A mis-vote near the boundary doesn't get averaged away — it *becomes a support vector* and warps the surface. The slack parameter (C) softens this, but a fat-fingered vote hands the SVM disproportionate damage.

The kernel is a second, stronger assumption: it fixes what "similar" means *before seeing any labels*. RBF says "the concept is a local neighborhood in this space"; linear says "the concept is a half-space."

**MLP: "every example is evidence about a probability."** The MLP's worldview is statistical. It models P(good | embedding), and every example pulls on the surface in proportion to how wrong the model currently is about it. Consequences:

- *Graceful under noisy votes* — one bad label is outvoted by the bulk rather than becoming load-bearing.
- *It can carve a positive class that is a union of clusters.* Real rare-event concepts are often multi-modal ("damaged packaging" is not one blob in embedding space). A hidden layer handles that natively; a linear SVM cannot, and an RBF-SVM handles it only if the kernel bandwidth happens to match the cluster scale.
- Its scores are (uncalibrated but smooth) probabilities — the natural input to the threshold-calibration and Hard-phase machinery.

Worth remembering: VTSearch's MLP is regularized down to 8–32 hidden units with 0.5 dropout — in practice, close to a robust logistic regression with a small nonlinear twist. So "MLP vs *linear* SVM" is really "weighted cross-entropy vs hinge loss" on nearly the same model class; **RBF is the philosophically distinct candidate**. That is why kernels get their own screening stage rather than one default configuration each.

**The deepest split: what they say about unseen data.** An **RBF-SVM is a local method** — its decision function decays toward its bias far from the support vectors, so on data unlike anything it trained on it defaults to "not the thing." Conservative extrapolation. A **ReLU MLP (and a linear SVM) is piecewise-linear**, extending its trend into regions it has never seen with unearned confidence in either direction. For VTSearch this is not academic: the autopilot's New phase deliberately probes unexplored regions of the embedding space. RBF's conservatism protects FPR out there — but it can also refuse to surface a genuinely novel pocket of positives that a linear model would have extrapolated toward. That interaction (model philosophy × diversity sampling) is precisely why the definitive stage runs the closed autopilot loop rather than a static train/test sweep.

**Rules of thumb:**

| Situation | Favors |
|---|---|
| Few, clean labels; well-shaped embedding space; unimodal concept | SVM (especially linear) |
| Noisy/inconsistent votes | MLP |
| Multi-modal positive class ("union of clusters" concepts) | MLP, or RBF if bandwidth cooperates |
| Rare events where false alarms in unexplored regions are costly | RBF (local, conservative off-manifold) |
| Rare events hiding in *novel* regions | Linear/MLP (willing to extrapolate) |
| Growing label counts, large inference sets | MLP (fixed inference cost; kernel-SVM inference grows with support-vector count) |

The honest prior: at VTSearch's operating point (tens of votes, unit-norm contrastive embeddings, near-linearly-separable classes), MLP ≈ linear SVM, with RBF as the genuine wildcard — its locality could win on the rare-prevalence arms (better FPR) or lose (worse FNR on novel positives). The 1%-prevalence arms and the separately-reported FPR/FNR curves are designed to expose exactly that.

## Experiment design

### Fixed choices (pre-registered)

| Choice | Value | Rationale |
|---|---|---|
| Media / embedder | image, `siglip` (768-d) | The user-priority workflow; single-vector, no region logic |
| Workflow | `autopilot` strategy of `simulate_voting_iterations`, text-sort seeded | The standard user flow; seeds from each category's `EvalQuery` text via `embed_text_query` |
| Threshold path | trainer-agnostic cross-calibration (`calibrate_count=2`, `calibration_fraction=0.5`), `safe_thresholds=False`, `inclusion=0` | Production defaults (`settings_models.py` has `safe_thresholds: bool = False`; inclusion 0 → cost = FPR + FNR, exactly the target metric) |
| Held-out split | `sim_fraction=0.5` | Half the dataset is never voted on; all FPR/FNR are measured there |
| Vote budget | `max_steps=200` | Users rarely vote more; curves flatten well before this |
| Primary metrics | per-step `fpr`, `fnr`, `cost = fpr + fnr` on the held-out test set | What the user asked for; AP alone hides the operating point |
| Secondary metrics | AUROC + AP on the test set per step | Rank quality independent of thresholding, isolates "bad ranking" from "bad threshold" |
| Statistical unit | one (dataset, category, prevalence-arm, seed, trainer) trajectory | Paired across trainers by (dataset, category, prevalence-arm, seed) |
| Decision rule | see below | Pre-registered so the report can't be argued backwards |

**Closed-loop comparison, by design.** Autopilot picks the next vote using the *current* model's scores, so MLP and SVM trajectories diverge after the first retrain even at the same seed. That is intentional: the question is "which model makes *VTSearch* better," and VTSearch's vote order depends on the model. The same-seed pairing still shares the sim/test split and the seeding phase, which is what makes paired statistics meaningful.

**Known fidelity simplification (accepted):** the eval's Hard/New interleave alternates on step parity, while the live app transitions phases from the Smart/Stable/Span indicators (`autopilot-state.service.ts` / `labeling_progress.py`). The parity interleave is the established harness behavior and applies identically to every trainer, so it cannot bias the comparison; upgrading it to the indicator-driven state machine is explicitly out of scope.

### Datasets and the rare-event arms

Three datasets, chosen to span concept type; all have eval queries already defined in `vtscore/eval/config.py`:

| Dataset | Items | Concept type | Natural prevalence per category |
|---|---|---|---|
| `caltech101_m` | 838 | Clean object categories (easy leg) | ~3–10% |
| `caltech256_a` | 3 800 | Harder object categories, bigger haystack | ~3% |
| `visual_genome_m` | ~1 000s, multi-label | "Does this object appear anywhere" — closest to real search | varies widely per category |

Optionally `vggface2_faces_m` (identity matching, prevalence 40/1600 = 2.5%) as a fourth leg if budget allows — identity is a different geometry (tight single cluster) that flatters RBF locality.

**Prevalence arms.** Rare events are the stated use case, and FPR/FNR behave differently at 1% than at 8%. Each (dataset, category) runs at:

- **natural** — the category's as-loaded prevalence;
- **rare** — positives (in both sim and test pools, before splitting) randomly downsampled to **1% prevalence** (skip the arm when that leaves < 15 positives total, to keep test-set FNR estimable).

8 categories per dataset (chosen from the eval config's list to span easy→hard by text-sort AP), 10 seeds, so the definitive grid is:

```
3 datasets × 8 categories × ≤2 arms × 10 seeds × N_trainers trajectories, 200 votes each
```

### Stage A — kernel/hyperparameter screen (cheap, decides `N_trainers`)

Run the existing **`label_curve`** sweep (random balanced labels — not autopilot; this stage only prunes the SVM configuration space, it does not decide the winner) on `caltech256_a` + `visual_genome_m`, `label_counts=(5, 10, 20, 50, 100, 200)`, `seeds=(0..9)`, over the widened trainer grid:

- `mlp` (production baseline)
- `svm_linear` × C ∈ {0.03, 0.3, 1, 3, 30}
- `svm_rbf` × C ∈ {0.3, 1, 3, 30} × gamma ∈ {scale, 4×scale, ¼×scale}
- `svm_poly` (degree 2 and 3, C ∈ {0.3, 3})
- `svm_sigmoid` (C ∈ {0.3, 3})

Selection: keep the best config per kernel family by mean AUROC at n_labels ∈ {10, 20, 50} (the autopilot-relevant regime), then advance **`mlp` + linear + rbf + (poly and/or sigmoid only if within one Hanley–McNeil SE of rbf)** to Stage B. Expected `N_trainers` ≈ 4.

### Stage B — the definitive autopilot run

For each cell of the grid above, run the (trainer-pluggable) `simulate_voting_iterations` and record per step `t`: `n_good`, `n_bad`, `fpr`, `fnr`, `cost`, `auroc`, `average_precision`, `train_seconds`, `xcal_seconds`, `test_score_seconds`, `pool_score_seconds`, plus run-level `trainer`, `backend` (torch-cuda / cuml / sklearn-cpu), `device`, `prevalence_arm`.

**Analysis (pre-registered):**

- Curves: mean ± bootstrap-95%-CI of FPR and FNR (separately — not just cost) vs vote count `t`, faceted by dataset × prevalence arm, one line per trainer.
- Budget table: cost / FPR / FNR at t ∈ {25, 50, 100, 200}, and area-under-the-cost-curve (AULC) over t ∈ [8, 200] (start at 8 = first post-seed-phase step so the 1-good-1-bad noise band is excluded).
- Paired tests: Wilcoxon signed-rank on per-(dataset, category, arm, seed) AULC and cost@50 / cost@200, each SVM vs MLP; Holm correction across the SVM variants.
- **Decision rule:** an SVM variant *wins* if it beats the MLP's mean cost at **both** t=50 and t=200 on **≥ 2 of 3 datasets** with corrected p < 0.05, and its rare-arm FNR is no worse than the MLP's at t=50 (rare events are the point; a model that trades FNR for FPR at 1% prevalence loses even if cost ties). Runtime is a tiebreaker only: if quality is statistically indistinguishable, prefer the model with better GPU inference scaling. Anything else → keep the MLP, publish the curves, close the question.

### Stage C — GPU runtime microbenchmark (independent of Stage B)

Timing embedded in Stage B reflects tiny vote-regime fits; the user also wants scaling curves. A dedicated benchmark, GPU only:

- **Training time vs training-set size:** n_train ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384}, balanced labels, real SigLIP vectors sampled from `caltech256_a`+`places365_m` embeddings. Each trainer fits at each size.
- **Inference time vs inference-set size:** n_infer ∈ {10³, 10⁴, 10⁵, 10⁶}, model trained at n_train=100 (the realistic vote budget), batch scoring.
- Methodology: `torch.cuda.synchronize()` around every timed region; 2 warmup runs discarded; median of 7 repeats; report per-size medians + IQR; record GPU model, driver, torch/cuML versions.
- Backends: MLP = torch CUDA (existing AMP path). SVM = **cuML** (`cuml.svm.SVC` / `LinearSVC`), following the `gpu_backends.py` gating pattern, with a small-N score-parity cross-check against sklearn (assert Spearman rank correlation vs the sklearn scores > 0.99 at n_train ≤ 256) so the quality and timing runs are trusted to describe the same model. If cuML is unavailable on the cluster, SVM rows are labelled `sklearn-cpu` and the report says so loudly rather than silently comparing CPU to GPU.
- Expected shape worth confirming, not assuming: kernel-SVM inference scales with support-vector count (which grows with n_train) × n_infer, MLP inference is a fixed 2-layer matmul; kernel-SVM *training* is super-linear in n_train while the MLP's 200 full-batch epochs are near-flat. The crossover points are the deliverable.

### Compute budget

Per Stage B trajectory: ~200 steps × (1 final fit + 2 calibration-fold fits + test/pool scoring). MLP fit at ≤200 labels is sub-second on GPU; SVM fits are milliseconds. Estimate 3–5 min/trajectory ⇒ full grid 3×8×2×10×4 ≈ 1 920 trajectories ≈ **100–160 GPU-hours**, embarrassingly parallel across trajectories (SLURM array over (dataset, category, arm, seed), all trainers inside one job so they share the loaded dataset). Knobs if that's too rich: seeds 10→5, categories 8→5, `max_steps` 200→150 cuts it to ~25 GPU-hours. Stage A is < 2 GPU-hours; Stage C is < 1.

### The report

`docs/experiments/mlp-vs-svm/` gets the raw CSVs (one per stage) and a generated `REPORT.md`: the decision-rule verdict up top, then FPR/FNR curve figures, the budget table, the significance matrix, timing scaling curves, and a limitations section (closed-loop divergence, parity interleave, single embedder). The report generator is deterministic from the CSVs so the cluster run and the write-up can't drift.

## Open work items

Each item is independently shippable; the orchestrator item depends on all the others. Recommended implementer model in parentheses.

<!-- item-sep -->

- **Shared trainer registry + wider SVM grid** (Sonnet 5) — Extract the `TRAINERS` registry and its adapter helpers out of `vtscore/eval/label_curve.py` into a shared `vtscore/eval/trainers.py` importable by both sweeps. Extend `vtscore/training/svm.py` with `poly` (exposed `degree`) and `sigmoid` kernels and expose `C`/`gamma` through the registry via parameterized trainer names (e.g. `svm_rbf@C=3,gamma=scale`) or a factory API — whichever keeps the CSV `trainer` column self-describing. `label_curve` behavior must be byte-identical for the existing names.

<!-- item-sep -->

- **cuML GPU backend for the SVM trainer** (Opus 4.8 — silent numeric drift between backends is the regression risk) — Add a cuML path to `train_svm` following the `vtscore/gpu_backends.py` gating pattern (opt-in when a usable GPU + cuML exist, sklearn fallback, disable-after-failure). `decision_sigmoid` scoring only needs `decision_function`, which cuML's SVC/LinearSVC provide. Expose the chosen backend on `SVMClassifier` so result rows can record it. Include the sklearn-parity rank-correlation test (GPU-marked).

<!-- item-sep -->

- **Trainer-pluggable voting simulation** (Opus 4.8 — must not perturb the MLP path byte-for-byte) — Thread a `trainer: str` through `simulate_voting_iterations` / `run_voting_iterations_eval`: `_train_and_calibrate` dispatches through the shared registry and uses the trainer-agnostic cross-calibration port (moved to the shared module from `label_curve._cross_calibrated_threshold`); `_score_pool` / `_evaluate_on_test` / `_blend_safe_threshold` consume a `predict_fn` instead of a torch module; `ALContext.model` becomes `Optional[Any]` (`al_strategies` only checks `is not None`). Add `auroc` / `average_precision` per-step columns (reuse `label_curve`'s metric helpers). Guard with a test asserting `trainer="mlp"` reproduces today's rows exactly on a fixed synthetic source (`al_benchmark` provides one).

<!-- item-sep -->

- **Per-step timing + provenance columns** (Haiku 4.5) — Add `train_seconds`, `xcal_seconds`, `pool_score_seconds`, `test_score_seconds`, `backend`, `device` to the voting-iterations row schema (monotonic-clock timing already modeled by `label_curve.evaluate_one`). Update `visualize.plot_voting_iterations` tolerance for the new columns.

<!-- item-sep -->

- **Text-sort seed-score glue** (Haiku 4.5) — A helper that, given loaded medias + a dataset's `EvalQuery` list from `vtscore/eval/config.py`, embeds each query via `embed_text_query` and returns the `{dataset: {category: {media_id: cosine}}}` mapping `run_voting_iterations_eval(seed_scores=…)` expects. Unit-norm vectors make this a dot product over the stacked matrix.

<!-- item-sep -->

- **Prevalence control** (Sonnet 5) — A `target_prevalence: float | None` knob on `simulate_voting_iterations` (and the runner) that seeds-deterministically downsamples positives before the sim/test split, records the realized prevalence in each row, and refuses (returns no rows) below a minimum-positives floor (15). Applies to multi-label datasets via `media_is_positive`.

<!-- item-sep -->

- **GPU timing microbenchmark script** (Sonnet 5) — `python -m vtscore.eval.timing_benchmark`: the Stage C design above (sizes, sync/warmup/median-of-7 methodology, backend/device provenance), emitting a tidy CSV. Runnable on CPU for smoke-testing (marked rows), GPU-marked test for the CUDA path.

<!-- item-sep -->

- **Orchestrator + report generator** (Sonnet 5) — `scripts/experiments/mlp_vs_svm/` following the `toponymy_image` template: `stage_a_screen.py`, `stage_b_autopilot.py` (one SLURM array task per (dataset, category, arm, seed)), `stage_c_timing.py`, `summarize.py` (CSV → `REPORT.md` with the pre-registered analysis: bootstrap CIs, budget tables, Holm-corrected Wilcoxon matrix, decision-rule verdict), `queue_all.sh`, node-scratch env setup via `common.py`. Plots via matplotlib into the report directory.

<!-- item-sep -->

- **(Optional) vggface2 fourth leg** (Haiku 4.5) — Wire `vggface2_faces_m` into the Stage B config as an identity-matching leg; no new machinery, just configuration + report faceting.
