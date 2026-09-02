# Provenance-partitioned calibration — keep manual-review votes out of the threshold's calibration set

**Status:** Design (pre-registered). Measurement first; the production change is
gated on the experiment's decision rules below.

## Background

The threshold's calibration votes are not drawn from the haystack distribution —
they are chosen by the detector's own sort, the most biased sampler available.
The selection-bias study
([`docs/experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md`](../experiments/2026-07-27-inclusion-knob/SELECTION-BIAS.md))
measured what that costs, and the answer splits cleanly by **labeling flow**:

- **Autopilot votes are safe.** Margin sampling (Hard phase) pulls calibration
  positives *below* the population's, so the conformal cut lands conservatively
  low; the New phase adds atlas diversity. FNR excess converges to ~0.004 by 100
  votes — the same rate as exchangeable random voting.
- **Manual top-of-list review votes are poison.** The `toplist` arm (a fair
  model of a user voting down a visible sorted result list — the Find-and-verify
  flow) biases calibration positives *high*, the dangerous direction: FNR excess
  **grows** with votes, 0.254 → 0.410 from 12 → 100, and Inclusion 0 recall
  falls to 0.474.

The textbook repair — propensity-**weighted** conformal quantiles — is shelved in
[`inclusion-calibration-bias.md`](inclusion-calibration-bias.md) because weights
*estimated* from tens of votes are high-variance, and its pre-registered trigger
is "a genuinely exploitative labeling flow." That flow already exists (manual
list review), but the cheaper repair is not weighting: it is **partitioning**.
Every vote's surfacing context is *known at click time* (which Autopilot phase
requested it, or that the user was reviewing a sorted list) — no propensity
needs estimating. Train on all votes; calibrate only on the votes whose
surfacing context the study measured as trustworthy.

## Design decisions (settled)

- **Provenance taxonomy.** `surfaced_by ∈ {autopilot:good, autopilot:bad,
  autopilot:hard, autopilot:new, list_review, seed_example, unknown}` plus
  context scalars (`score_at_vote`, `rank_at_vote`, `sort_kind`). Recorded per
  vote at click time — the surfacing model is gone by the next retrain, so this
  is *not re-derivable* and must be persisted (scalars only; the No Persisted
  Vectors rule is untouched). Storage: a namespaced key in
  `LabeledElement.metadata`, which already round-trips through labelset
  export/import. Recording is issue #2850 and ships independently of everything
  below.
- **Eligibility default for unattributed votes.** Legacy votes and imported
  labelsets carry no provenance (`unknown`) and stay **calibration-eligible**.
  The partition only bites when a session demonstrably mixes flows; it must not
  silently gut calibration for every existing labelset.
- **Partition point.** The fold split, not the training set.
  `compute_fold_orderings` / `_grouped_folds` (`vtscore/training/thresholds/conformal.py`)
  gain a per-vote calibration-eligibility mask threaded from the detector glue
  (`vtscore/detectors/training.py`, where votes become `X_list`/`y_list`):
  ineligible votes are pinned to the **Train** side of every fold split and
  never appear in a calibration ordering. Fold models still see every vote, so
  ranking quality is untouched by construction.
- **Starvation fallback.** If the eligible subset cannot support a stratified
  split (fewer than 2 eligible positives or 2 eligible negatives on the
  calibration side), the partition disables itself and the unpartitioned rule
  runs — a biased calibration beats no calibration, and cold start must not
  regress.

## Pre-registered experiment

**Harness:** extend `scripts/experiments/inclusion_knob/run_autopilot_sweep.py`,
which already drives the repo's own `vtscore.eval.al_strategies` selector, and
reuse the `toplist` policy from `run_selection_sweep.py` (batches of 8 off the
current sort, matching page-at-a-time manual review). New knob: a **mixed vote
stream** that interleaves autopilot-selected votes with toplist bursts at a
configured fraction, tagging each simulated vote with its provenance.

**Arms.** Vote-stream mix (autopilot : list-review) ∈ {100:0, 85:15, 70:30,
50:50, 0:100} × calibration policy ∈ {`all-votes` (status quo),
`partitioned` (eligible-only + starvation fallback)}. Same evaluation frame as
the selection-bias study: AG News one-vs-rest + synthetic separability arms,
4 seeds, checkpoints at 12/24/50/100 votes, sim/test halves, production
conformal rule with the safe-blend on.

**Metrics.** Per (arm, seed, checkpoint), paired across calibration policies on
identical vote streams: `fnr_excess` at Inclusion 0–3 (the load-bearing
number), threshold − oracle-cut, precision and recall at Inclusion 0, and the
fraction of cells where the starvation fallback engaged.

**Hypotheses.**

- **H1:** In mixed streams (≥15% list-review), `partitioned` cuts FNR excess at
  50–100 votes by at least half of the toplist-induced excess, and the gain
  grows with the manual fraction.
- **H2:** In the 100:0 stream the two policies are near-identical (every vote is
  eligible; any difference is fold-split noise).
- **H3:** `partitioned` does not pay for its recall in precision: threshold −
  oracle moves toward zero, not merely downward.
- **H4:** The starvation fallback keeps the 12-vote checkpoint no worse than
  `all-votes` in every mix.

**Decision rules.**

1. H1 and H2 hold → ship the partition (mask + fallback) to production and
   scope `docs/ML.md`'s Inclusion-budget caveat to unattributed votes.
2. H1 holds but H3 fails (recall bought with precision) → the partition is a
   knob, not a default; report and stop.
3. H1 fails → the shelved weighted-conformal item stays shelved, this plan's
   production item is dropped, and the negative result is recorded in the
   report. Vote-provenance recording (#2850) stays shipped regardless — it is
   the enabler for any future repair and for measuring real (non-simulated)
   per-user vote-order bias.
4. Whatever the outcome, write
   `docs/experiments/provenance-calibration/REPORT.md` with the paired tables.

## Open work

<!-- item-sep -->

- [ ] #2850 — Record vote surfacing provenance in `LabeledElement.metadata`
  (Sonnet 5)

<!-- item-sep -->

- **Mixed-flow harness arm + measurement.** The `run_autopilot_sweep.py`
  vote-stream mix knob, provenance tagging in the simulated stream, the
  `partitioned` calibration policy behind a harness flag, and the paired
  analysis per the pre-registered metrics and decision rules above. CPU-cheap
  (same scale as the selection-bias sweep).

<!-- item-sep -->

- **Production calibration partition (gated on decision rule 1).** The
  eligibility mask through `compute_fold_orderings` / `_grouped_folds` with the
  starvation fallback, sourced from recorded provenance in the detector glue;
  cache-key participation (the mask changes the fold orderings, so it must
  enter `_calibration_cache_key`); tests in `tests_lib/detectors/`.

<!-- item-sep -->
