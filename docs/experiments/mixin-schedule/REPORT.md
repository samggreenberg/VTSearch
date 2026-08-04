# How safe should safe-thresholds be? (#2841)

**Status: Phase 1 (screen) complete and reported below. Phase 2 (A/B) running;
its section and the final verdict are not yet written.**

## The question

`safe_thresholds` blends two candidate decision thresholds: a **GMM cut** fitted
on the score distribution (needs no labels, never wild) and a **cross-calibration
cut** (conformal, uses the labels, unreliable when there are few). #2799 settled
*whether* to blend and turned it on for everyone. It did not touch the schedule
inside the blend — **how much GMM, for how long** — which had been one
hard-coded line since it was written:

```python
x-cal weight = clip((n_labels - 6) / 14, 0, 1)   # pure GMM ≤6, pure x-cal ≥20
```

Three independent choices are baked into that line, and none had ever been
measured: the **endpoints** (6, 20), the **shape** (linear), and the
**statistic** it ramps on (total labels). A fourth question — whether a weighted
average is even the right combiner — could not be expressed in it at all.

The issue's framing was that pure GMM forever is presumably bad ("that's
probably not better than ignoring the learned threshold entirely, right?").
**The data says otherwise, and then says something more interesting than
either.**

## What was built

`vtscore/training/blend_schedules.py` — a registry of 18 named schedules across
five families, so each buried choice becomes an arm:

| Family | Schedules | What it varies |
|---|---|---|
| controls | `pure_gmm`, `pure_xcal` | the two ends of the axis |
| A. endpoints | `prod`, `fast`, `slow`, `vslow`, `early`, `late` | when the handoff starts and finishes |
| B. shape | `convex`, `concave`, `step`, `logistic` | the curve between the endpoints |
| C. statistic | `rare`, `pos` | ramps on the **rarer class** / positive count instead of the total |
| D. cap | `cap80`, `cap50` | never hand off completely |
| E. corridor | `corridor`, `corridor_ramp` | **clamp** x-cal between the GMM component means instead of averaging |

Family C exists because the binding constraint on conformal calibration is the
rarer class, not the total: a 19-bad/1-good labelset has 20 labels and one
positive, and #2790 traced the deep threshold spikes to exactly that starvation.
Family E exists because a weighted average taxes *every* x-cal cut to defend
against the rare wild one, whereas the pathology safe-thresholds actually fixes
is wild (#2788's cold-start "admit nothing" cuts).

`prod` is pinned **bit-identical** to the historical ramp by a 0..80
parametrized test, so the machinery is a no-op for users until a default flips.

### Fidelity: the app and the framework disagreed, and it hid two bugs

Both app training paths hard-coded *"skip cross-calibration below 6 votes"*;
the eval harness never skipped. That was safe only because the production weight
happens to be 0 there — a coincidence, not a guarantee. Deriving the skip from
the schedule (`xcal_is_discarded`) makes the two agree by construction, and any
schedule that trusts the learned cut earlier now automatically stops skipping.
Two real defects fell out:

- **The two app paths left different placeholders** (`NO_GOOD_THRESHOLD` on the
  Find path, `0.5` on the vote/labelset path). Normally discarded — but when the
  GMM fit degenerates, `blend_gmm_threshold` falls back to that placeholder, so
  the same collection would admit *nothing* through one path and *everything
  scoring ≥0.5* through the other. Both now admit nothing, the safe reading of
  "no threshold was ever computed".
- **An off-by-one against its own rationale**: the guard skipped *below* 6 but
  paid for two 200-epoch fold fits at *exactly* 6, where the weight is already
  zero. No user ever saw the difference — the app's first trained detector
  appears at 7 votes — but it was pure waste.

### COCO, from cache

The issue asks for VG **and** COCO, and COCO is not a VTSearch demo dataset. It
does not need to be: the #2790 sweep already embedded all 4952 COCO-2017-val
images, and the boxes are staged flat beside them, so `build_coco_pickle.py`
joins the two into an ordinary media pickle and every existing stage runs on it
unchanged. Nothing is re-embedded.

**One honest limit.** That cache stores each image's whole vector and its HAC
region vectors but **not** the raw 14×14 patch grid that `max_patch` pools over,
so COCO can only serve the **binary-voting** arm; a COCO region arm would need a
genuine re-embed. The builder refuses patch embedders outright rather than
emitting a pickle that would silently score as whole-image while being reported
as a region arm.

## Method

Two phases, because the blended threshold **feeds acquisition**: Autopilot's
Hard phase picks the item nearest the decision boundary, so two schedules label
different items and their trajectories diverge.

- **Phase 1 — screen.** One run on the production trajectory, with every
  schedule re-cut counterfactually at each step (one extra metric row each). All
  schedules see the same model, the same step, and the same held-out test
  scores, so they are exactly paired. Cheap — many schedules for the price of
  one run — but structurally blind to acquisition feedback.
- **Phase 2 — A/B.** One full independent trajectory per schedule, paired per
  cell. This is the verdict; the screen only decides who gets to run.

**Arms.** Region voting = VG × `dinov3_patch` × `max_patch` (the production
region path live decisions read). Binary voting = VG × `siglip` and COCO ×
`siglip`/`siglip2` × `whole_image`. Reported **separately, never pooled** — the
issue allows them to want different curves, and pooling would hide exactly that.

**Metric.** Inclusion-weighted `cost` (= FPR + FNR at inclusion 0), averaged over
the **7–20 vote window**: 7 is the app's first trained-detector step, 20 is where
the production ramp ends.

**Scale.** 1008 cells, 23 VG categories (scale-banded) + 19 COCO categories,
12 seeds, 30 steps, the production **linear** head. 42 cells emitted no rows
(rare small-object categories; deterministic and pre-vote, so symmetric across
schedules).

**Fidelity check, run before anything is reported:** `prod`'s counterfactual row
must reproduce the threshold and cost the run actually used. It did, to
**0.000e+00 over 26,142 paired rows**. Every other schedule row comes off the
same code path, so this is what licenses the rest.

## Phase 1 result: the ranking is monotone in "how long you keep the GMM"

Region voting (n = 254 paired cells, baseline `prod` cost 0.5376):

| schedule | cost | d_cost | % cells improved | p (Wilcoxon) | d_fnr | d_fpr |
|---|---|---|---|---|---|---|
| `pure_gmm` | 0.4839 | **−0.0537** | 83.5 | 9.7e-27 | −0.0738 | +0.0201 |
| `vslow` | 0.4875 | −0.0501 | 85.0 | 3.1e-30 | −0.0599 | +0.0098 |
| `slow` | 0.4981 | −0.0395 | 86.6 | 4.3e-32 | −0.0430 | +0.0035 |
| `late` | 0.4987 | −0.0389 | 86.2 | 8.4e-31 | −0.0465 | +0.0076 |
| `corridor_ramp` | 0.5144 | −0.0232 | 67.3 | 1.1e-17 | −0.0298 | +0.0066 |
| `cap50` | 0.5162 | −0.0214 | 86.2 | 1.1e-34 | −0.0205 | **−0.0009** |
| `convex` | 0.5197 | −0.0179 | 82.7 | 9.3e-28 | −0.0221 | +0.0042 |
| `rare` | 0.5230 | −0.0146 | 57.9 | 7.4e-06 | −0.0159 | +0.0014 |
| `pos` | 0.5289 | −0.0087 | 55.5 | 0.0087 | −0.0116 | +0.0029 |
| `cap80` | 0.5335 | −0.0041 | 76.8 | 1.3e-25 | −0.0039 | **−0.0002** |
| `corridor` | 0.5355 | −0.0021 | 44.9 | 0.24 (n.s.) | −0.0079 | +0.0057 |
| `logistic` | 0.5417 | +0.0042 | 33.1 | 4.5e-11 | +0.0004 | +0.0037 |
| `early` | 0.5497 | +0.0121 | 21.7 | 6.7e-27 | +0.0137 | −0.0016 |
| `step` | 0.5537 | +0.0161 | 22.8 | 1.6e-20 | +0.0047 | +0.0114 |
| `concave` | 0.5584 | +0.0208 | 16.9 | 1.6e-31 | +0.0216 | −0.0007 |
| `fast` | 0.5786 | +0.0410 | 11.8 | 4.7e-35 | +0.0385 | +0.0025 |
| `pure_xcal` | 0.6012 | +0.0636 | 13.0 | 3.4e-35 | +0.0606 | +0.0030 |

Binary voting (n = 694 paired cells, baseline `prod` cost 0.4931):

| schedule | cost | d_cost | % cells improved | p (Wilcoxon) | d_fnr | d_fpr |
|---|---|---|---|---|---|---|
| `rare` | 0.4621 | **−0.0310** | 86.0 | 6.8e-74 | −0.0592 | +0.0282 |
| `pos` | 0.4622 | −0.0308 | 85.9 | 1.6e-72 | −0.0591 | +0.0283 |
| `vslow` | 0.4624 | −0.0307 | 76.2 | 6.3e-50 | −0.0827 | +0.0520 |
| `slow` | 0.4640 | −0.0291 | 82.4 | 1.1e-68 | −0.0578 | +0.0287 |
| `late` | 0.4691 | −0.0239 | 75.4 | 2.8e-49 | −0.0656 | +0.0417 |
| `pure_gmm` | 0.4694 | −0.0236 | 69.3 | 2.5e-25 | −0.1060 | +0.0824 |
| `corridor` | 0.4714 | −0.0217 | 69.0 | 6.3e-34 | −0.0313 | +0.0096 |
| `corridor_ramp` | 0.4728 | −0.0202 | 72.8 | 1.5e-40 | −0.0601 | +0.0398 |
| `cap50` | 0.4752 | −0.0179 | 87.5 | 4.4e-89 | −0.0211 | +0.0032 |
| `convex` | 0.4826 | −0.0105 | 70.0 | 3.1e-33 | −0.0354 | +0.0249 |
| `cap80` | 0.4896 | −0.0035 | 75.5 | 2.2e-72 | −0.0034 | **−0.0001** |
| `early` | 0.5001 | +0.0070 | 30.3 | 2.5e-33 | +0.0228 | −0.0158 |
| `logistic` | 0.5013 | +0.0082 | 22.8 | 8.5e-57 | −0.0037 | +0.0119 |
| `concave` | 0.5090 | +0.0159 | 19.5 | 5.3e-63 | +0.0350 | −0.0190 |
| `step` | 0.5222 | +0.0291 | 15.3 | 1e-80 | +0.0005 | +0.0286 |
| `fast` | 0.5336 | +0.0405 | 11.5 | 3e-91 | +0.0596 | −0.0190 |
| `pure_xcal` | 0.5588 | +0.0657 | 11.2 | 1.7e-92 | +0.1011 | −0.0354 |

Three things read straight off these tables:

1. **`pure_xcal` is the worst schedule on both arms** (+0.064 / +0.066). That is
   safe-thresholds OFF, and it independently reproduces #2799's verdict on a
   different grid — a free replication.
2. **The issue's premise is inverted.** Pure GMM forever is not bad. On region
   voting it was the single **best** schedule (−0.0537), and on binary voting it
   still beat the incumbent. Every schedule that hands off *faster* than
   production (`fast`, `step`, `concave`, `early`) lost.
3. **Production sits in the wrong half of its own family.** Ten of seventeen
   schedules beat it.

## …but most of that is not calibration, it is a lower cut

Every schedule at the top of those tables has a strongly negative `d_fnr` and a
positive `d_fpr`. That is the signature of simply **cutting lower**, and at
inclusion 0 the cost weights are (1, 1), so trading a lot of FNR for a little
FPR scores as a win. It is a real win at the shipped operating point — but #2790
flagged exactly this trap, so the same paired cells were re-scored under
asymmetric weights.

Region voting, `d_cost` under each weighting:

| schedule | fpr ×1 (shipped) | fpr ×2 | fpr ×4 | fnr ×2 |
|---|---|---|---|---|
| `pure_gmm` | −0.0537 | −0.0336 | **+0.0066** | −0.1276 |
| `vslow` | −0.0501 | −0.0403 | −0.0208 | −0.1100 |
| `slow` | −0.0395 | −0.0360 | −0.0291 | −0.0825 |
| `cap50` | −0.0214 | **−0.0222** | **−0.0240** | −0.0419 |
| `cap80` | −0.0041 | −0.0043 | −0.0047 | −0.0080 |
| `corridor` | −0.0021 | +0.0036 | +0.0150 | −0.0100 |

Binary voting:

| schedule | fpr ×1 (shipped) | fpr ×2 | fpr ×4 | fnr ×2 |
|---|---|---|---|---|
| `rare` | −0.0310 | −0.0028 | **+0.0537** | −0.0902 |
| `vslow` | −0.0307 | +0.0213 | **+0.1253** | −0.1133 |
| `slow` | −0.0291 | −0.0004 | +0.0571 | −0.0869 |
| `pure_gmm` | −0.0236 | +0.0588 | **+0.2235** | −0.1297 |
| `corridor` | −0.0217 | −0.0121 | +0.0071 | −0.0529 |
| `cap50` | −0.0179 | −0.0147 | **−0.0084** | −0.0390 |
| `cap80` | −0.0035 | −0.0036 | −0.0037 | −0.0069 |

`pure_gmm`'s apparent win **flips** on both arms — catastrophically on binary
(+0.2235). So do `rare`, `pos`, `slow` and `vslow` on binary. The schedules that
improve at **every** weighting are the **cap family**, plus `slow` on region only.

The sharpest version of this: on region voting, **`cap50` and `cap80` are the
only schedules that improve *both* error types** — `cap50` at
d_fnr −0.0205 *and* d_fpr −0.0009, a Pareto move with no trade at all. Every
other winner buys FNR with FPR (`pure_gmm` gives up 0.27 FPR per FNR gained on
region, 0.78 on binary; `cap50` gives up 0.15 on binary and *nothing* on region).

**Method note, stated plainly:** this re-weighting was added *after* seeing the
first table, prompted by the FNR/FPR pattern — it was not pre-registered. And it
is a **scoring** sensitivity, not a simulation of a different Inclusion setting:
moving the Inclusion knob changes the conformal rule itself, not just the
weights. It answers "is this win an artefact of a symmetric metric", which is
the question it was added for, and nothing more.

## Reading so far

The two questions come apart:

- *"Should the handoff be slower?"* — At the shipped operating point, yes,
  dramatically. But the gain is mostly permissiveness, and a user who cares more
  about false alarms than misses would see it reverse. That makes it a
  **product/operating-point decision**, not a pure calibration improvement.
- *"How safe should safe be?"* — The weighting-independent answer is **never
  fully hand off**. A permanent GMM share is the one change that improves
  calibration rather than relocating the operating point, and on the production
  region path it improves both error types at once.

`cap50` was not in the pre-registered promotion list (which ranked on the
shipped metric alone), so it was **added** as an extra A/B arm rather than
quietly substituted; the pre-registered arms all still run.

## Phase 2 — A/B trajectories

_Running. Nine arms (`prod`, `pure_gmm`, `vslow`, `slow`, `rare`, `pos`,
`cap50`, `cap80`, `corridor`), 16 seeds, 1344 cells each, full independent
trajectories so acquisition feedback is included._

## Provenance

- Screen: `/exp/sgreenberg/mixin-2841/results-screen/` (`REPORT_screen.md`,
  `screen_deltas.csv`, `screen_sensitivity.csv`).
- A/B: `/exp/sgreenberg/mixin-2841/results-ab/<arm>/`.
- Cached embeddings reused, nothing re-embedded: VG + Caltech pickles from the
  Max-Patch run (`/exp/sgreenberg/max-patch/datadir/embeddings/`), COCO from the
  #2790 sweep cache (`/exp/sgreenberg/threshold-stability/cache/regions/coco/`).
- Analyzer self-test (`selftest_analyze_mixin.py`) recovers planted effects,
  their magnitudes and signs, a null, an opposite-per-mode split, the window
  exclusion, and the fidelity abort.
