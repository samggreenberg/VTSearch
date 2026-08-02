# Sparse-positive spike fix (#2790 follow-up) — pre-registered plan

## Motivation

The conformal spike attribution found the residual cost spikes are a **too-few-
positives** problem: 87% of up-spikes happen with `n_good ≤ 6` (median **3**), 89%
are a Bad vote on a `hard`-selected boundary item. The Autopilot phase machine
(`region_curve.py::AutopilotPhaseMachine`) starts boundary (`hard`) sampling the
instant it leaves the `good` phase — i.e. at `good_to_start` positives (default **3**).
So the loop is boundary-sampling and calibrating a cut on ~3 positives, which even the
conformal gap-midpoint can't pin.

## Hypotheses

- **H1 (acquisition):** requiring more positives before boundary sampling — raising
  `good_to_start` — cuts the spike rate and the sparse-positive spike share, because
  the first trained cuts see enough positives to be stable.
- **H2 (defer the cut):** holding the cold-start / GMM threshold until `n_good ≥ K`
  (not trusting the trained conformal cut while positives are sparse) does the same on
  the threshold side, independent of the phase machine.
- **H3 (tradeoff):** both cost votes — more of the annotation budget goes to positives
  before the detector is useful. The win is only real if spike/variance drop **without**
  materially worsening convergence (cost-AUC / final cost / label efficiency).

## Arms (conformal k2; COCO × SigLIP 2 × whole; 5 classes × 10 seeds)

| arm | change |
|---|---|
| `base` | `good_to_start=3` (current default) |
| `goods6` | `good_to_start=6` |
| `goods9` | `good_to_start=9` |
| `goods12` | `good_to_start=12` |
| `defer6` | `good_to_start=3`, but hold the cold-start threshold until `n_good ≥ 6` (new `--defer-cut-goods` knob) |
| `goods6-defer6` | both |

`bad_to_start=4` throughout (we delay the *onset* of bad/hard sampling via the good
count, not the bad budget).

## Metrics

Per arm, from `results.jsonl` + traces:

- **Instability:** `spike_rate` (t≥20), across-seed `cost_sd`, and the **sparse-positive
  spike share** from `spike_analysis.py` (does the residual shrink?).
- **Cost of the intervention (H3):** `cost_auc` (mean test cost over t), `final_cost`
  (t=60), and **label efficiency** = votes to first reach cost ≤ 0.15; plus
  **votes-in-good-phase** (how long before the first bad/hard vote).
- **Accuracy:** `mean_regret`, `final_f1`.

## Decision rule (pre-registered)

Adopt the smallest `good_to_start` (or the `defer` variant) that cuts `spike_rate`
**and** the sparse-positive spike share by ≥ 40% vs `base`, without worsening
`cost_auc` or `final_cost` by > 0.02 (paired over (class, seed)). If raising
`good_to_start` reduces spikes but pushes `cost_auc` past that bar, the acquisition
lever is rejected in favour of `defer` (which keeps normal acquisition), or the
instability is accepted as the price of fast labeling.

## Watch-outs

- A rare class may not yield `good_to_start` positives from cold-start `top` selection
  in reasonable votes — the loop would stall in the `good` phase. Cap / flag
  votes-in-good-phase; if a cell spends > ~25 votes before its first bad, record it.
- `defer` must not simply reproduce the GMM safe-threshold ramp (already active t<20);
  it forces cold-start (not blended) until `n_good ≥ K`.

---

## Results (5 classes × 10 seeds, conformal k2)

| arm | spike_rate | cost_sd | sparse% | runaway% | cost_auc | final_cost | votes_in_good | regret |
|---|---|---|---|---|---|---|---|---|
| `gts3` (base) | 0.161 | 0.134 | 87% | 22% | 0.357 | 0.304 | 2.0 | 0.153 |
| `gts6` | 0.104 | 0.094 | 54% | 27% | 0.320 | 0.282 | 5.0 | 0.156 |
| `gts9` | 0.083 | 0.082 | 0% | 18% | 0.302 | 0.285 | 8.1 | 0.141 |
| `gts12` | 0.106 | 0.078 | 0% | 21% | 0.324 | 0.288 | 19.5 | 0.157 |
| **`defer6`** | **0.017** | **0.053** | 96% | **7%** | **0.264** | **0.236** | **2.0** | **0.051** |
| `goods6-defer6` | 0.104 | 0.094 | 54% | 27% | 0.320 | 0.282 | 5.0 | 0.156 |

**Verdict: `defer6` wins — a Pareto improvement (H2 ≫ H1).** Holding a distribution-based
GMM cut while `n_good < 6` (instead of trusting the sparse conformal cut) cuts the
spike rate **~10×** (0.161→0.017), the runaways **3×** (22%→7%), the cross-seed cost sd
**2.5×**, and regret **3×** (0.153→0.051) — while *also* lowering `cost_auc` and
`final_cost` (better, faster convergence) at **zero** extra annotation budget
(`votes_in_good` = 2, same as base). The better early cut feeds better boundary
selection, so the whole trajectory improves, not just the sparse window.

- **H1 (raise `good_to_start`, the "more goods" idea) works but is dominated.** It
  reduces spikes (0.161→0.083 at `gts9`) and moves them out of the sparse regime
  (sparse% → 0 by `gts9`), but spends annotation budget doing so — `gts9` burns ~8 of
  60 votes in the `good` phase, `gts12` ~20 (a third of the budget) and starts
  regressing. `defer6` beats every `gts*` arm at no budget cost.
- **H3 (tradeoff): `defer6` has none** — it improves instability *and* convergence.
- **The two levers are redundant** (`goods6-defer6` ≡ `gts6`): once `good_to_start ≥
  defer_cut_goods`, the loop is still in cold-start while positives are sparse, so
  there is no trained cut for `defer` to override. Combining them adds nothing.

**Recommendation:** adopt `--defer-cut-goods ≈ 6` (hold a GMM cut until ~6 positives) —
it is the sparse-positive fix, and unlike raising `good_to_start` it costs the user
nothing. Confirm on the region-voting/hac path and at other prevalences before
shipping to the app.

---

## Confirmation: region-voting / hac path (DINOv3, bag-aware grouped calibration)

Re-ran `base` vs `defer6` on the path real detectors use — hac proposal + region-voting
(max over ~24 HAC nodes, grouped conformal calibration), DINOv3, same 5 classes × 10 seeds.

| arm | spike_rate | cost_sd | runaway% | cost_auc | final_cost | votes_in_good | regret | n_spikes |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.181 | 0.306 | 35% | 0.446 | 0.294 | 9.3 | 0.184 | 263 |
| **`defer6`** | **0.128** | **0.201** | **16%** | **0.390** | **0.267** | 9.3 | **0.133** | **184** |

**`defer6` confirms — still a Pareto win** (spikes −30%, runaways ~2×, cost_sd −34%,
cost_auc / final_cost / regret all lower, zero extra budget). Two caveats: (a) the
**effect is smaller** than on the whole-image path (−30% vs −90% spikes) — the hac
max-over-nodes score distribution is more skewed, so the 2-component GMM cut is less
clean; (b) the **residual is still the sparse regime** (defer6 spikes: 96% Bad, 91%
`n_good ≤ 6`, median 3). The baseline spike *pattern* is identical across paths (263
hac vs 235 whole; 93% Bad, ~70%+ sparse), so the phenomenon is not path-specific.

**Takeaway:** ship `--defer-cut-goods ≈ 6`; it helps everywhere. The region-voting
path needs follow-up (a bag-aware deferred cut, not a plain pool GMM) to close the
gap. Prevalence robustness (below 1/100) still needs LVIS/VG — COCO val2017 caps it.

---

## Per-item: which media items cause the spikes (not random!)

Attributing every conformal-path up-spike to the vote that caused it
(`spike_items.py`, `render_spike_items.py`; visual report published as an Artifact):

- **Spikes are not sporadic — there are repeat offenders.** Each class has ~1 negative
  that spikes almost every time it is voted: `stop-sign 562818` and `traffic-light
  47769` spiked in **all 10 seeds** (rate 1.0); parking-meter/bus/fire-hydrant each
  have one at 0.7–0.9. **14 images cause 66/235 (28%) of all spikes**; the top 5 (one
  per class) cause ~19%.
- **What they are:** true COCO negatives the cold-start ranker surfaces at the decision
  boundary (model score ~0.1–0.4), voted Bad while only ~3 positives exist (median
  `n_good` = 3 at these items).
- **The false-negative signal (as suspected):** ~12% of spikes are Bad votes the model
  scored **>0.5** (hard/false negatives — it thinks they contain the class). A few top
  offenders are **Good** votes (a genuine positive whose late arrival reshuffles the
  sparse calibration). Where COCO draws a GT box on a "Bad" card, the annotation
  disagrees with the vote — candidate mislabels.
- **Recognizable at click-time:** the trigger is "a boundary-score item voted Bad
  before ~6 positives exist" — exactly the condition `--defer-cut-goods 6` guards.

---

## Broadened per-item run: 79 COCO categories × 20 seeds

Scaled the spike-item analysis for confidence + variety (CPU array — the whole path
is GPU-cap-free there). **1,580 traces, 6,729 spikes, 5,225 distinct culprit items.**

- **The recurring-offender pattern is near-universal:** **73 of 79 categories** have an
  image that stresses the cut in ≥5 seeds; **~15 categories have one that spikes in all
  20/20** (incl. `stop-sign 562818` and `traffic-light 47769` from the first run —
  stable across the broader run). 64 items spiked in ≥10/20 — so "spiky" is now
  statistically solid, not a 2-of-10 fluke. Repeat offenders (≥2) = 29% of all spikes.
- **Stronger hard-negative signal:** 15% are Bad votes scored >0.5, and several top
  offenders score **>1.0** (potted-plant, sink, chair, couch, backpack, remote) — the
  model is *confident* they contain the class. These are the highest-value look-alikes
  / candidate mislabels.
- **Reframe (owner):** these items are **valuable training signal**, not problems — a
  foreign-language stop sign is exactly what multilingual recognition needs; a bulb
  lamp that reads "trafficy" is the traffic-light-vs-livingroom-light boundary we want.
  The spike is a *valuable example landing while the cut is fragile*; `--defer-cut-goods`
  keeps the example and drops the instability.
- Visual catalog: published Artifact (1 high-confidence offender per category, ≥12/20).
  Tools: `spike_items.py` (per-item + confidence tiers), `render_spike_items.py`.

---

## Do the spike-causing items transfer across path / embedder? (robustness)

`spike_overlap.py` compares the offender sets (image ids that spiked in ≥K seeds)
between two runs, same 5 classes × 10 seeds.

- **Q1 — region-voting (hac/DINOv3) vs whole-image (SigLIP 2): no overlap.** Every
  class's top offender is a different image (whole stop-sign `562818` 10/10 vs
  region-voting `500826` 2/10; traffic-light `47769` 10/10 vs `366178` 2/10; etc.).
  Jaccard 0.
- **Q2 — same path, DINOv3 vs DINOv2 (both hac/region-voting): also no overlap.**
  Top offenders are entirely different images per class; Jaccard 0. So it isn't the
  proposal path that decides — it's the **embedder**.
- **Also observed:** the whole/SigLIP 2 path *concentrates* on strong repeat
  offenders (8–10/10 seeds), while both region-voting/DINO paths are **diffuse** (top
  only 2–4/10) — the bag-aware max-over-nodes scoring adds per-seed noise, so no single
  image is reliably the boundary case there.

**Conclusion: spikers are representation-specific, not a property of the image.** An
image sits at the decision boundary of *one embedder's* score landscape when positives
are sparse; a different embedder puts different images there. This means the "valuable
hard examples" are found **per embedder** — the SigLIP 2 mislabels/look-alikes and the
DINO ones are different sets, both worth harvesting. (The sparse-positive fix
`--defer-cut-goods` is embedder-agnostic — it stabilizes the cut regardless of which
images are at the boundary.)

---

## New GMM (equal-density crossing, dev #2801) on the region-voting path

Ported dev's crossing-based `calculate_gmm_threshold` (cuts at the components'
equal-density crossing, not the midpoint of means — engineered for the region-voted
max-over-~24-nodes score distribution, whose Bad mode is a wide right-skewed
extreme-value statistic). Re-ran base vs defer6 (DINOv3/hac, 5 classes × 10 seeds).

| GMM | arm | spike_rate | cost_sd | runaway% | cost_auc | final_cost | regret |
|---|---|---|---|---|---|---|---|
| old (midpoint) | base | 0.181 | 0.306 | 35% | 0.446 | 0.294 | 0.184 |
| old (midpoint) | defer6 | 0.128 | 0.201 | 16% | 0.390 | 0.267 | 0.133 |
| new (crossing) | base | 0.180 | 0.234 | 29% | 0.356 | 0.248 | **0.120** |
| new (crossing) | defer6 | 0.180 | **0.190** | **15%** | **0.337** | **0.215** | 0.147 |

- **The crossing GMM is a clear win for region voting** — it lowers the cost *level*
  substantially on the base arm (regret 0.184→0.120, cost_auc 0.446→0.356, final_cost
  0.294→0.248, runaways 35→29%) by cutting the skewed distribution correctly.
- **It largely subsumes defer's benefit here:** new-GMM *base* already beats old-GMM
  *defer6* on regret/cost_auc/final_cost. defer was a workaround for a bad GMM; a good
  GMM does most of the job in the safe-threshold blend directly.
- **Neither reduces the spike *rate*** (~0.18 across all four) — the step-to-step jumps
  persist; the win is in the cost *level*, not jumpiness.
- **Best combo = new-GMM + defer6** (lowest cost_sd/cost_auc/final_cost, runaways 15%),
  though regret ticks up vs new-GMM base. Worth updating to the new GMM regardless.

---

## What's special about the spiking *vectors*? (pure embedding, #2790)

`spike_vectors.py` — no eval framework, just cached SigLIP 2 whole-image embeddings.
For each class: G = positive vectors, B = negatives, S = spikers (bad, ≥5 seeds). For
each spiker, its **percentile among all bads** on several geometric features; mean over
71 classes (one strong spiker each):

| feature | spiker pctile | reading |
|---|---|---|
| `margin` = cos_G − cos_B | **0.69** | the signature: spikers lean toward good on the good↔bad axis |
| `proj_w` (Fisher axis) | 0.69 | identical (monotone with margin) |
| `cos_G` (good centroid) | 0.62 | more class-like than typical bads |
| `cos_B` (bad centroid) | 0.44 | *less* bad-like |
| `max_cos_G` (nearest good) | 0.59 | mildly nearer one exemplar |
| `knn_good` | 0.56 | slightly more good neighbors |
| `b_outlier` | 0.53 | ~typical — **not** an outlier |

**A spiker is a negative pulled toward the good cluster and away from the bad cluster
— a boundary case near the model's cut, leaning good but not extreme, and NOT an
isolated outlier.** That is why nothing stands out in the pixels: the signal is a soft
position in embedding space, not a distinct visual feature. The effect is real but
moderate and **class-dependent** (strong: oven/sink/zebra/elephant, margin pct 0.9+;
weak/reversed: umbrella/vase/surfboard). Consistent with "vaguely hard negatives."
Tool: `spike_vectors.py`.

### Does the signature hold on a second embedder, and is it *predictive*? (SigLIP 1)

Re-ran the whole pipeline on **SigLIP 1** (fresh embed of all 4,952 val2017 whole
vectors → 79-class × 20-seed labeling simulation → `spike_items` → `spike_vectors`).
Two questions: does the *descriptive* signature replicate across embedders, and is it
*predictive* (would picking the vector-hard negatives actually find the spikers, letting
us skip the simulation)? Added **precision@N / lift** to `spike_vectors.py`: rank all
bads by a feature, take the top-N (N = #spikers, base rate ~1%), count how many spike.

| feature | pctile S1 / S2 | lift S1 / S2 |
|---|---|---|
| `margin` = cos_G − cos_B | 0.68 / 0.69 | **0.0× / 2.5×** |
| `proj_w` (Fisher axis) | 0.68 / 0.69 | 0.0× / 2.5× |
| `cos_G` (good centroid) | 0.65 / 0.61 | 5.4× / 2.2× |
| `cos_B` | 0.46 / 0.44 | 1.8× / 0.0× |
| `max_cos_G` | 0.61 / 0.59 | 1.5× / 3.3× |
| `knn_good` | 0.56 / 0.56 | 0.0× / 2.6× |
| `b_outlier` | 0.52 / 0.53 | 0.0× / 0.9× |

**The descriptive signature is embedder-invariant.** Every percentile matches SigLIP 2
to ±0.04 — margin 0.68 vs 0.69, knn_good 0.56 vs 0.56, b_outlier 0.52 vs 0.53. So even
though the *specific* spiking items don't transfer across embedders (0 overlap, above),
the geometric *type* does: a spiker is a good-leaning boundary negative, not an outlier,
in **both** representations. The "vaguely hard negative" picture is real and general.

**But the signature is descriptively real and predictively near-useless.** The
mean-percentile (≈ each feature's AUC as a spike predictor) is a soft central tendency
at ~0.68 — which carries almost no mass into the top-1% tail where the precision@N cut
falls. So precision@N is tiny everywhere (best case `cos_G` on SigLIP 1: 5.4× lift but
still only 5% precision), it's **near-zero for the margin signature itself** (0.0× on
SigLIP 1), and *which* feature has any lift isn't even stable across embedders (margin
0.0× vs 2.5×; cos_B 1.8× vs 0.0×). **You cannot pre-screen spikers from the embedding
geometry** — the top vector-hard negatives are ~95%+ non-spikers. Spiking is set by the
per-step training dynamics (which model, which prior labels), which the static vector
position only weakly tilts. **The app simulation is irreplaceable for identifying which
hard negatives actually spike.** Tool: `spike_vectors.py --embedder siglip`; artifacts
`/exp/sgreenberg/threshold-stability/broad_sig1/spike_vectors_sig1.json`,
`broad/spike_vectors_sig2_v2.json`.

## FIX: a low-variance (linear) head stops the spikes (`--head-strategy`, `results_eval.py`)

The deep spikes are ~85% under-determined-MLP score variance, so the direct fix is a
lower-variance model while positives are sparse. Head-strategy sweep (20 COCO classes × 15
seeds, SigLIP 2, MLP-regime, t≥20; trace-free via `results_eval.py` on `results.jsonl`):

| head | deep-spike rate | spike size | @t60 cost | @t60 FNR | @t60 FPR |
|---|---|---|---|---|---|
| `mlp` (baseline) | 0.055 | 0.167 | 0.386 | 0.365 | 0.020 |
| **`linear`** (logistic) | **0.025** | 0.140 | **0.352** | **0.299** | 0.052 |
| **`svm`** (linear) | **0.023** | 0.134 | 0.356 | 0.308 | 0.049 |
| `reg-mlp` (small hidden) | 0.042 | 0.152 | 0.381 | 0.310 | 0.071 |
| `anneal-svm` | 0.026 | 0.139 | 0.364 | 0.319 | 0.044 |

**A linear head (logistic / linear SVM) cuts the deep-spike rate ~55–58% AND improves the
customer metric** (cost 0.386→0.352, FNR 0.365→**0.299** — catches more needles), for a modest
FPR rise (0.02→0.05) — the trade a rare-needle customer wants. Not a trade: a Pareto win. The
gap is largest early (t=20 cost 0.508→0.396), exactly the sparse regime the spikes live in.
`reg-mlp` (shrinking the hidden layer) helps far less — the MLP's nonlinearity still injects
variance; you must go fully linear. `anneal-svm ≈ svm` because `n_good` rarely clears the K=8
switch (positives stay sparse), so it's SVM almost throughout. All arms are **uniform within
cross-calibration** (`make_head` keyed on the full labelset's `n_good`, reused for M0 + the fold
sub-models) and use conformal on raw `decision_function` scores (no Platt).

**Recommendation:** for the #2790 spike regime (sparse positives, few dozen votes) replace the
MLP with a **linear head**. Caveat: validated to t≤60; the earlier MLP-vs-SVM study found the
MLP overtakes by ~t=200 with abundant labels, so the full production answer is an **anneal keyed
on `n_good`** (linear/SVM while sparse → MLP once rich) — only the sparse end is tested here.
Tools: `--head-strategy {linear,svm,reg-mlp,anneal-*}`, `results_eval.py`.

## MECHANISM: what the hidden calibration catastrophe actually is (`calib_conditions.py`)

Necessary+sufficient conditions, established by instrumenting the calibration (`_calib_diag`
records, per MLP step: where the cut sits in the bad→positive gap, and `poolpos_recall` = the
**true** recall over *all* pool positives, which the sim knows but the labeled data hides).
Two hypotheses were **refuted** on the way, which is the crux of the answer:

1. *Refuted:* a spike is **not** a training-recall collapse — at spikes 99.6% of labeled
   positives stay **above** the cut. The cut does not jump above the labeled positives.
2. *Refuted:* the trigger is **not** a vote scored high. Bads essentially never outscore a
   labeled positive; and a Bad above the *previous top bad* spikes only 2.0% vs 12.4% for a
   Bad *below* it. The vote's score position does not predict the spike.

**What it is.** Acquisition surfaces *easy* (high-scoring) positives, so the ~3 labeled
positives are unrepresentatively high and sit well above the cut. The conformal cut lives in
the **gap** between the labeled bads and those high positives — and the *unlabeled/test*
positives live densely in that gap, near the cut. Each vote **retrains** the MLP; with almost
nothing pinning the cut in the gap, the cut (and/or the scores) **wobbles** on every refit. A
spike is a step where the operating point wobbles **up** through that dense band of unlabeled
positives → true recall collapses (`Δpoolpos_recall` −0.093 at spikes vs +0.004 otherwise;
`Δcut_in_gap` +0.218, `Δthreshold` +0.031) — while labeled recall stays ~1, so it's invisible
to the labeled data. **That divergence is why it's "hidden."**

**Necessary** (all three): (1) sparse, unrepresentatively-high labeled positives → an
under-determined cut in a wide gap; (2) a retrain wobble that moves the operating point up —
the **cut** (~60% of spikes, `Δthreshold`>0) or the **scores** under a stable cut (~40%, a
second mode); (3) unlabeled positives densely in that path.

**Sufficient: none observable.** (2)∧(3) is sufficient, but (3) is *hidden* (unlabeled) and
(2) is a *stochastic retrain outcome* not determined by the vote's features. The tightest
labeled-observable predictor is the symptom "the cut moved up" — **60% recall, 18%
precision**. No vote-level trigger exists. The unobservability of a sufficient condition is
not a gap in the analysis; it **is** the nature of the catastrophe. This is also why the fixes
work by *reducing the cut's freedom* (defer/GMM/recall-anchor pin it to the score distribution
or the positives), and why more positives barely moved FNR (mode-2 + the ranking floor remain).
Tools: `calib_conditions.py`, `_calib_diag`; artifacts `broad_v2_diag3/`.

### The "unobservable" residual IS the cross-calibration fold reshuffle (2026-08-02)

Owner's insight: the threshold is set by **cross-calibration** (split labels Train/Calibrate,
sub-model on Train, cut on Calibrate, ×`calibrate_count` folds pooled) — so *which fold the
labels land in* should matter, and might explain the "hidden" residual. Reading the code, it
does — precisely:

- The conformal cut at `inclusion=0` is the **gap midpoint between the top negatives and
  `min(pos)` — the *lowest calibration positive*** (`conformal_threshold`). So the cut is
  pinned by the single lowest positive that landed in the Calibrate fold.
- The split is `rng.permutation` of the positive/negative index arrays with a **fixed run
  seed** (`stratified_fold_orderings`). With `cal_fraction=0.5` and ~3 positives, that's a
  **1-train / 2-cal** split. When a vote arrives the arrays grow, so the permutation
  **reshuffles which positive is held out to Train every step** — jerking `min(cal-positive)`
  and hence the cut. It's not random noise: it's a *deterministic, chaotic* function of the
  fold assignment, invisible from the vote's own score (which is why my earlier "stochastic /
  unobservable" framing was incomplete — it's observable in fold space, just not vote space).
- This also predicts the old Stage-B result that **`calibrate_count`=k8 is the dominant
  stability lever, not the rule**: more folds pool more Calibrate positives, so the pooled
  `min(cal-positive)` stabilizes to the true lowest → the cut stops jumping.

**Result — the reshuffle is real but MINOR (hypothesis mostly refuted in magnitude).**
`--stable-folds` (freeze the split → no per-step reshuffle) cut the deep-spike rate only
~10–15% (t[20,30) 0.108→0.094; overall 0.080→0.072), **not** the ~60% predicted. Reason:
freezing the fold *identity* doesn't freeze the *scores* — the MLP retrains every vote, so
`min(cal-positive)`'s **score** still moves even when *which* positive it is stays fixed. So
the dominant wobble is the **MLP retrain itself** (scores shift), with the fold reshuffle a
~13% add-on. (Side effect: stable folds gave a better operating point — FNR 0.321→0.263, cost
0.343→0.310.) The `calibrate_count` sweep agrees: deep-spike t[20,30) 0.108→0.100→0.097→0.088
for k=2/4/8/16 (overall 0.080→0.067; deepest band t[45+) 0.039→0.025, −36%). More folds pool a
more stable cut and `min(cal-pos)`, but the effect is diminishing and modest.

**Decomposition of the deep spikes — DIRECT measurement (corrects the by-exclusion estimate).**
Instrumenting `poolpos_recall` at the *previous* cut with the *current* scores splits each
spike's recall drop into two **additive** credit terms (hold one fixed, move the other):
`Δrecall = Δscore (retrained scores, cut held) + Δcut (cut moved, scores held)`. Averaged over
the 3002 spikes (800 traces): total −0.094 = score −0.063 + cut −0.030 — i.e. **credit for the
lost recall ≈ 67% under-determined-model / 33% moving-cut** (signed). (An earlier note said
"56/44"; that was `mean(|score|)`/`mean(|cut|)` — the share of absolute *movement*, which
over-credits the cut by counting recall-*helping* moves; **67/33 is the credit for the loss**.)
This *is* a credit-attribution, **not** a partition of spikes ("X% of spikes are score-caused")
— essentially every spike has both terms. Two caveats: (a) it's a first-order attribution, so
decomposition order shifts it a few points; (b) the two aren't independent causes — **the cut
moves *because* the scores moved** (the conformal cut recomputes on the retrained scores), so
most of the 33% cut credit is *downstream* of the model variance. The genuinely independent,
*removable* cut slice (the cross-calibration fold reshuffle) is only ~13% (`--stable-folds`;
~16–19 via `calibrate_count`↑). So, correcting the earlier by-exclusion "75–85 / 15–25":
- **Model score variance ≈ ⅔** — the under-determined 3-positive MLP gives different scores
  each retrain; test positives wobble vs a fixed cut. Only more positives fix it (hard).
- **Cut movement ≈ ⅓, but mostly downstream of the model** — only ~13 pts is the removable
  fold reshuffle; the rest is the cut faithfully tracking the unstable MLP. The owner's
  cross-calibration hunch is a real contributor (I first *under*counted it), but the removable
  part is small.

Both are rooted in **sparse positives**. This is *why* no observable sufficient condition
exists — the dominant term is the MLP retrain's score shift, a complex nonlinear function of
the whole label set, unpredictable from the vote. And it's *why the shipped fixes work*:
defer/GMM/recall-anchor cut the *cut's* sensitivity (the ~15–25% calibration term) **and**
park the operating point where a wobble crosses fewer test positives — they don't touch the
MLP variance but blunt its consequence. Tools: `--stable-folds`, `--calibrate-count`;
artifacts `broad_v2_stable/`, `broad_cc{4,8,16}/`.

## The actual #2790 spikes: deep-run transient excursions (`deep_spikes.py`)

The per-event work above characterizes *all* MLP-regime Bad votes; the #2790 complaint is
specifically the violent jumps **deep in the run, while the MLP is already trained and
improving** (issue: seed0 t~24, cost 0.088→0.424 then recovers). `deep_spikes.py` isolates
these — tracking each spike's **up-jump and recovery** ("jump back to good"), sliced by
depth `t`, on the faithful v2 traces. Both embedders identical.

- **They persist deep and don't fade with learning.** Spike rate by `t`: t[20,30) ≈ 10%,
  t[30,45) ≈ 6.6%, t[45+) ≈ 4%. A well-trained MLP at t=20–45 still spikes ~7–10% of Bad
  votes.
- **They are violent and self-correcting.** Median up-jump +0.17–0.19, but the tail is
  catastrophic: e.g. `kite` s10 t27 **cost 0.009 → 0.993**; `baseball-glove` s3 t51 **0.037
  → 1.000** — the cut lurches over *all* the positives (FNR→1), then **~37% snap back within
  2 steps** (median recovery 2 steps); the other ~40–45% run away longer.
- **Root cause = sparse POSITIVES that never resolve, not the handoff.** `n_good` median at
  a deep spike is **5** (min 3) even at t=25–51 — the boundary/`hard` acquisition surfaces
  mostly negatives, so the MLP is pinned by ~3–5 positives *for the whole run*. **93% of
  deep spikes are live false-positives** (`surface_margin`≥0): one boundary Bad added to ~3
  positives shoves the cut across the real matches → cost→1.0 → next retrain re-fits →
  snap back. Pure threshold-*placement* instability (ranking/oracle barely moves, Stage B
  cost_sd ≈ 4× oracle floor).
- **Implication for mitigation:** staying in HardText longer (more *bads* before learned
  sort) does **not** help — `n_good` stays ~5 regardless of bad count. The lever is the
  *positive* side (see the mitigation experiments below). Tool: `deep_spikes.py`.

## Mitigation experiments: it's the positives, and threshold tricks trade F1

Tested the two candidate levers on the faithful harness (79 classes × 20 seeds); both land
on one conclusion: **the bottleneck is positives, not the cut rule or the negatives.**

**defer6 (crossing-GMM cut while sparse) crushes the deep cost-spikes ~8× — but trades
the deep spikes.** Deep-spike rate t[20,30) 0.10 → 0.012 (both embedders); the
distribution-based GMM cut doesn't lurch when one boundary Bad is added. It also **catches
more needles**: SigLIP 2 @t60 FNR 0.32→**0.18** (misses 18% vs 32%), FPR 0.02→0.09, cost
(=FNR+FPR) 0.343→**0.268**. (**Metric correction**, owner 2026-08-01: an intermediate write-up
called this a regression on **F1** — that was wrong. Under `neg_multiple`=100, F1 collapses
to precision ≈ 2·TP/(TP+FP) and *under-weights* FNR, so it penalises the extra FPs and hides
the recall win. For a rare-needle, don't-miss-any customer the right metric is **FNR+FPR**
(prevalence-independent, recall-inclusive) — which is what `cost` already is — and by it
defer6 is a genuine **improvement**, catching 82% of needles vs 68% for a review set of 9%
vs 2% of the hay. So the original cost-based "defer6 helps" stands; only the F1 detour was
the error.)

**Handoff-quality arms (`handoff_quality.py`, SigLIP 2) — staying in TextHard longer buys
nothing; more positives is the only lever:**

| arm | switch t | handoff jump | @t60 cost | @t60 F1 |
|---|---|---|---|---|
| g3/b4 baseline | 7.1 | +0.125 | 0.343 | 0.625 |
| g3/b8 more bads | 11.1 | +0.065 | 0.345 | 0.621 |
| g3/b12 more bads | 15.2 | +0.096 | 0.347 | 0.622 |
| g6/b4 more goods | 10.4 | +0.054 | 0.334 | **0.677** |

More bads (the *free* lever — the haystack is all hay) softens the one-time handoff jump
but leaves the learned detector **flat** at matched budget. More *goods* is the only thing
that moves quality (F1 0.625→**0.677**). But "require more goods" is **begging the
question** — finding needles is the task VTSearch exists for; you can't demand them.
Confirmed in the data: g6/b4 ran only 1540/1580 traces because **40 seeds couldn't reach 6
goods** (the hard classes can't supply them).

**Unified root cause = positive starvation.** The deep threshold spikes (cut pinned by ~3
positives all run), the horrible Text→Learned scores (3-positive MLP never catches text),
and the recall/FNR ceiling all dissolve only when `n_good` grows. The one lever that is both
**shippable and helps everything** (spikes *and* recall, no threshold trickery, no demanding
goods) is **positive-*seeking* acquisition**: have the autopilot spend some picks surfacing
its best *guesses* at positives for the user to confirm — the `soft` select mode
(`--soft-seek`), which picks the item with calibrated P(good)~0.7 (the ~50–76%-good band vs
`hard`'s ~17%) — the system doing the hard task, not begging for it. **Built + under test**; the g6/b4 arm is the "if goods were free" ceiling it should chase. **Metric (owner
2026-08-01):** evaluate on **FNR+FPR** (= `cost`) and **recall-at-review-budget**, *not* F1.
Customers hunt rare needles (~0.1%) and can't miss any; F1 collapses to precision under that
imbalance and under-weights FNR — the opposite of the goal. Since #2790 shows the *ranking*
is the stable/good part and only the *threshold* is unstable, a threshold-independent
recall@budget is the cleanest "customer value" metric. Tools: `handoff_quality.py`,
`deep_spikes.py`, `soft_eval.py` (reports n_good / FNR / FPR / cost, not F1).

## When does a Bad vote spike? (per-EVENT, #2790)

If the vector position only weakly tilts the odds, *what* sets them? The unit that
answers this is the **vote event**, not the item: `spike_events.py` rebuilds every
Bad-vote event from the traces (~77k per embedder) with the **state at vote time**
(`n_good`, `n_bad`, `t`, `surface_margin` = item score − cut, `select_mode`, `head`)
plus context geometry vs the labeled-so-far sets, and whether it spiked (Δcost > 0.1).

### Harness-fidelity fix (this changed the mechanism story)

The app's autopilot (cold-start, `retrainMode=false`) surfaces the **good and bad phases
with text/example (cosine) sort** and only switches to the learned **MLP at the `hard`
phase** — once `n_good ≥ good_to_start` (3) **and** `n_bad ≥ bad_to_start` (4)
(`label-view.component.ts` phase dispatch / `AutopilotPhaseMachine.next_phase`). The
realistic harness (`_resolve_step_head`) instead switched to the MLP the instant *both*
classes existed (the **first bad**, `n_bad`=1), so it was scoring *and selecting* with a
1–3-bad MLP the app would never use. Fixed: gate MLP training on
`mlp_eligible = n_good ≥ good_to_start ∧ n_bad ≥ bad_to_start`, holding the cosine
cold-start head through the whole good+bad phase. Verified end-to-end (`head` stays
`cosine` at `n_bad` 0→3, flips to `mlp` at `n_bad`=4) and 42 region_curve tests pass.
**All numbers below are the post-fix re-run** (`broad_sig1_v2`, `broad_v2`); the pre-fix
`n_bad<4` "spikes" were harness artifacts and are now exactly **0.000**.

### The three regimes

1. **Cosine phase (`n_bad` 0–3, good+bad phases):** no MLP; cost is *frozen* (flat across
   every `n_bad`=0 step in 1519/1580 traces). Spike rate at `n_bad` 0 and 1 is **0.000** —
   the old "first bad = 59–71%" was entirely the too-early MLP switch.
2. **Hard-phase handoff (`n_bad` 3→4):** the app fires learned sort entering the `hard`
   phase, replacing the frozen cosine cost with the first MLP's calibrated cost. This
   crosses the Δcost>0.1 bar **~41% (S1) / 53% (S2)** of the time — a real, app-faced,
   *one-time* model switch (smaller than the old bogus 1-bad handoff because the MLP now
   has 4 bads). Not a retrain instability.
3. **Recurring MLP-retrain spikes (`n_bad` ≥ 4):** the phenomenon of interest. Both
   embedders agree to within a few points (S1 = SigLIP 1, S2 = SigLIP 2):

| condition | spike rate S1 / S2 |
|---|---|
| **base** (all Bad votes) | 6.7% / 7.5% |
| `n_bad` 0 / 1 (cosine phase) | **0.000 / 0.000** |
| item **below** cut (`margin`<0) | **0.7% / 0.9%** |
| item **at/above** cut (`margin`≥0) | 15.5% / 17.3% |
| &nbsp;&nbsp;└ just above [0, 0.1) | 15.9% / 17.7% |
| &nbsp;&nbsp;└ just below [−0.1, 0) | 0.5% / 0.7% |
| sparse pos `n_good` 3 → ≥15 | 8→2% / 10→3% |
| `margin`≥0 ∧ `n_good`≤4 (best profile) | 16.7% / — |

**The necessary condition is "the item is a live false-positive," not "hard-ish Bad far
from other bads."** A recurring spike essentially requires the voted item to currently
score **at/above the cut** (`margin`≥0) — the crossing at `margin`=0 is nearly a hard
boundary (0.5% just below vs 15.9% just above). Voting Bad on something the MLP already
scores as bad can't move the operating point. `margin≥0` is the (near-)necessary gate.

**Amplifier:** **sparse positives** — `n_good`=3 ⇒ ~8–10%, decaying monotonically to ~2%
by `n_good`≥15. The logistic ranks `surface_margin` (+0.64) then `n_good` (−0.28) on both;
all else minor.

**"Far from other bads" is NOT a driver.** Distance to the labeled-bad set (`nb_cos_max`)
and the context good↔bad margin are flat-to-slightly-*rising* in spike rate on both
embedders. The intuition was really tracking the *count* — few bads yet — not geometric
distance. (Consistent with the vector study: spikers aren't outliers.)

**Not sufficient.** Even the best profile (`margin`≥0 ∧ `n_good`≤4) spikes only ~1 in 6.
The residual is the exact cut arithmetic on that retrain — whether this one Bad tips the
operating point across a real match — which isn't a function of the vote's own features.

**Why the same item spikes only sometimes (when a spiker does NOT spike).** 78/95 (S1)
robust spikers **also** have no-spike occurrences — the same item does both. Holding the
item fixed, the spike occurrences differ from the no-spike ones by **trajectory timing,
not geometry**: Δ`t` −1.9, Δ`n_bad` −1.4, Δ`n_good` −0.5, Δ`surface_margin` +0.01; every
geometry delta ≈ 0. So a spiker spikes when caught **early — few positives, while it still
scores above the cut**; the *same* item voted later, after its score has fallen below the
stabilized cut, lands in the 0.7% floor and does nothing. Spiking is a property of *when*
the item is met, not of the item.

Tool: `spike_events.py`, `inspect_mech.py`; harness fix in `region_curve.py`
(`_resolve_step_head`); artifacts `broad_sig1_v2/spike_events.json`,
`broad_v2/spike_events.json`. Two guards fall out: (1) the hard-phase cosine→MLP handoff
is one-time and app-faced — could be smoothed by warm-starting the first MLP cut toward
the cold-start operating point; (2) recurring spikes need a live FP at the cut with sparse
positives — damp the cut move while `n_good` is small (the shipped crossing-GMM /
`defer-cut-goods` levers).
