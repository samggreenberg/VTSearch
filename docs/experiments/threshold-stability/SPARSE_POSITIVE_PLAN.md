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
