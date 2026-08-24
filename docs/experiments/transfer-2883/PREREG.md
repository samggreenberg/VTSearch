# Pre-registration: is `transfer` a bias or a variance? (issue #2883)

**Written before the run.** Nothing below may be edited after the cells array is
submitted; the findings go in a separate `REPORT.md`. This line has now produced
two wrong-but-plausible readings at full power (#2836 and #2846, both mis-sized
by the theory bench in the same direction), so a prediction recorded afterwards
is not a prediction.

## BLUF — the prediction

**Most of the `+0.037` "transfer" term is not a cost anyone pays. It is a
variance measured against an optimistic reference point.**

Concretely, on the production arm (`visual_genome_m/dinov3_patch/max_patch`),
ramp window 6–20:

| # | claim | pre-registered bar |
|---|---|---|
| H1 | `transfer` is a **variance**, not a bias | `symmetry` = \|mean\|/mean_abs **< 0.10** in threshold units, and **below all three sibling terms** |
| H2 | the reference point is **optimistic**, and that is the majority of the term | `optimism` > 2 SE, and `optimism_share` **> 0.5** |
| H3 | it scales like an **estimation error** | `a + b/m` fits with median R² **> 0.8**, slope **> 0** at > 2 SE; **`n_pos` fits better than `m`** |
| H4 | `pooled_sim_oracle` is **not a bound** on test loss | at least one regularised reading beats it, **p < 0.01**; specifically **`smooth` wins and `bag` does not** |

**If all four land, the follow-up is not a better cut or a better fit — it is
more positives, and a decomposition whose reference point is honest.** If H2 and
H4 both fail, `transfer` is a real irreducible cost, `family_headroom_exhausted`
stands as written, and the whole calibration line is genuinely finished.

## What changed since #2883 was written (2026-08-07)

Checked before designing, because two of the issue's premises had moved.

**Confirmed.** The cut axis is dead. #2881 — the one live strand the issue
carves out — ran in #3130 and came back a **negative** on both arms (+0.0069,
p = 0.001 region; +0.0058, p = 0.014 whole-image), against its own
pre-registered null. Nothing has reopened it.

**Corrected.** The issue quotes `transfer` = **+0.0407**, from a decomposition
#3187 later found to be aggregated with a per-column `mean()` that dropped
NaN-linked steps independently per term. Re-running today's fixed
`analyze_cut.py` over #3130's 552 cells (this branch, no new compute) gives the
corrected production-arm ramp row:

| term | corrected | as published |
|---|---:|---:|
| `prior_loss` | +0.0114 | +0.0111 |
| `identification` | **−0.0057** | −0.0057 |
| `misspecification` | +0.0129 | +0.0129 |
| **`transfer`** | **+0.0372** | +0.0389 |
| total | **+0.0557** | +0.0595 |
| residual | **0.0** (exactly) | — |

`transfer` still dominates at **67 %**, so #2883's aim is unchanged and #3187's
expectation ("65–68 % either way") is discharged. Two facts nobody had seen:
**478 of 3921 steps (12 %) lack a complete oracle chain**, and the residual is
now exactly zero rather than silently absorbing dropped rows.

**Stale.** `launch_cut.sh` pins `CALIB_HEAD=linear`. `PRODUCTION_HEAD` is
`linear_svm` since #3198; `preflight.sh` check 12 blocks the stale pin. This run
uses `linear_svm`.

**Wrong as stated.** #2883's central justification — and
`decisions.family_headroom_exhausted`, which #2884 mechanised from it — is that
`pooled_sim_oracle` *"bounds every rule that picks a threshold from that sim
set"*. It bounds every rule's loss **on the sim set**. It is not a bound on
**test** loss, which is what every table in this line reports, because the
empirical minimiser overfits the sample it minimises over. That is H4, and it is
not a hypothetical: the repo has already recorded the same estimator being beaten
on test twice.

- #3116 found `rule_inefficiency` — the trained cut minus a calibration-set
  oracle, scored on test — **negative in every row** of #2897, and named the
  cause: *"the oracle is overfitting a handful of scores."*
- This study's own `cost_identification` is **−0.0057**: the label-reading
  `supervised` cut already **loses on test** to the unsupervised `priorfree`.

A term whose siblings go negative is not measuring the cost of an assumption.

## Why the term cannot be what its name says

`D_sim` and `D_test` are one random partition of a single pool
(`_split_media_ids`, `sim_fraction = 0.5` → 2096 / 2097 medias) scored by one
model. The two score samples are draws from the same distribution. **There is no
distribution to transfer across**, so the term can only be estimation error.

The corrected table already says so in threshold units, and this is the
observation the study is built on:

| term | mean | mean_abs | \|mean\|/mean_abs |
|---|---:|---:|---:|
| `prior_loss` | +0.0154 | 0.0154 | **1.00** |
| `identification` | −0.0176 | 0.0183 | **0.96** |
| `misspecification` | +0.0057 | 0.0191 | 0.30 |
| **`transfer`** | **+0.0003** | **0.0168** | **0.016** |

Three terms move the cut the same way every time. `transfer` moves it 0.017
either way and nets to nothing — while costing +0.037, because the cost curve is
convex near its minimum and the reference is a sample minimum. That is a
variance, and variance has different remedies from misspecification.

## The design

Everything is a **re-cut of the same per-step models**: no extra training, no
GPU, no new embeddings. It rides the existing cells array.

**Subsampling happens at cut time, not by changing `sim_fraction`.** This is the
one design decision the study turns on. Re-running with a smaller `sim_fraction`
would shrink the sim set and *grow the test set* at the same time, moving the
estimator and the reference point together — the exact confound #3116 names
("re-decompose against a reference that does not move with the arm"). Cutting a
seeded subsample of the sim pairs leaves the test set, the vote trajectory and
the per-step model **bit-identical** across levels, so the only thing that moves
is the number of labelled sim scores.

**Three estimates of one reference point**, because the quantity the
decomposition needs — the population optimum `C(τ*)` — has never been measured:

| estimator | what it is |
|---|---|
| `oracle_cut` (today's `oracle_cost`) | the sample minimum — a **lower** bound |
| `honest_test_oracle` (K = 5, cross-fitted) | cut and cost on disjoint folds — an **upper** bound |
| the `a + b/m` intercept | uses neither bound's sample |

`transfer` is therefore reported as a **bracket**, not a point.

**Arms.** Unchanged from #3130 so the contrast is direct: `visual_genome_m`,
23 categories × 12 seeds × 30 steps, `dinov3_patch/max_patch` (production region
voting) plus the `siglip/whole_image` single-vector control. 552 cells, cpu
partition. Pile at `/expscratch/$USER/vts-cache`.

## The known threat to H2, stated in advance

`honest_test_oracle` picks its cut on 4/5 of the test set, so it is a *worse*
cut than the full-sample one and its cost includes that penalty. **The measured
`optimism` is therefore an upper estimate of the reference artefact**, and
`optimism_share > 0.5` could be met by the penalty rather than by the artefact.

This is why H3's intercept is in the design and not decoration: it estimates the
same reference from the sim-side curve, with no cross-fitting penalty in it. If
the intercept and the cross-fitted reference agree, the bracket is real. **If
they disagree by more than the bracket's own width, H2 is reported as
unresolved** — not as a win — and the honest reading is that the reference point
needs a better estimator before anyone re-decomposes anything.

## What is measured but cannot win

The label-free `bagfit_mid` / `bagfit_priorfree` arms bag the *mixture fit*
rather than the labelled cost curve. They read no labels, so unlike everything
else here they could ship — and they are in `SWEEP_ONLY`, excluded from
`best_by_cost`, `best_vs_production` and the ship gate.

#2883 item 1 asks for the characterisation **before** a remedy. A remedy that
wins in the very run that diagnoses the disease is the wrong-but-plausible
result this line has already paid for twice. If they look good, that is an
argument for a pre-registered follow-up, not a ship.

## Out of scope

- **The Gumbel crossing.** #2846 is closed; `gumbel_any_*` gets no more spend.
- **The tail-α cut.** #2881 closed negative in #3130.
- **Any new cut rule at all.** This study attacks the fit and the reference
  point; a cut rule would be the axis three studies have already exhausted.
- **The theory bench as evidence.** #2883 item 3: it has mis-sized this family
  twice at full power. Synthetic data is used here only to pin the estimators'
  behaviour in `tests_lib/detectors/test_transfer_rules.py` — a selftest of the
  code, never a result.

## Commitments

- If H1–H4 all land: `family_headroom_exhausted` gets a caveat naming what it
  actually measures, the decomposition gets an honest reference, and the
  follow-up is **positives**, not cuts.
- If H2 and H4 both fail: `transfer` is real and irreducible, the flag stands,
  and this line closes. That is a publishable negative and it will be written up
  as one.
- The one result that reopens rather than settles: a **flat learning curve with
  a large intercept** — that would mean the term is neither variance nor
  reference artefact, and something in the sim/test split is not what the code
  says it is.

Refs #2883, refs #3187, refs #3116, refs #2884, refs #2879, refs #2836.
