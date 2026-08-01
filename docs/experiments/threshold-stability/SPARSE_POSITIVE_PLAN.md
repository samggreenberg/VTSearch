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
